#!/usr/bin/env python3
"""MPNN embedding generation and management module.

This module provides the MPNNEmbeddings dataclass and functions for
generating, saving, and loading neural network embeddings from protein
structures using the MPNN (Message Passing Neural Network) architecture.

Key components:
- MPNNEmbeddings: Dataclass for storing per-residue embeddings
- from_pdb: Generate embeddings from a PDB or CIF file
- from_npz: Load pre-computed embeddings from NumPy archive
- _embed_pdb: Internal function for MPNN embedding computation

Embeddings are 64-dimensional vectors computed for each residue,
capturing structural and sequence features for alignment.

Supported file formats:
- PDB (.pdb): Standard Protein Data Bank format
- mmCIF (.cif): Macromolecular Crystallographic Information File format
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import haiku as hk
import jax
import numpy as np
from Bio.PDB import MMCIFParser, PDBParser
from Bio.PDB.Structure import Structure
from jax import numpy as jnp

from sabr import constants, model, util

LOGGER = logging.getLogger(__name__)

# Constants for CB position calculation (standard protein geometry)
_CB_BOND_LENGTH = 1.522  # C-CA bond length in Angstroms
_CB_BOND_ANGLE = 1.927  # N-CA-CB angle in radians (~110.5 degrees)
_CB_DIHEDRAL = -2.143  # N-CA-C-CB dihedral angle in radians


@dataclass(frozen=True)
class MPNNInputs:
    """Input data for MPNN embedding computation.

    Contains backbone coordinates and residue information extracted
    from a PDB or CIF structure file.

    Attributes:
        coords: Backbone coordinates [1, N, 4, 3] (N, CA, C, CB).
        mask: Binary mask for valid residues [1, N].
        chain_ids: Chain identifiers (all ones) [1, N].
        residue_indices: Sequential residue indices [1, N].
        residue_ids: List of residue ID strings.
        sequence: Amino acid sequence as one-letter codes.
    """

    coords: np.ndarray
    mask: np.ndarray
    chain_ids: np.ndarray
    residue_indices: np.ndarray
    residue_ids: List[str]
    sequence: str


def _np_norm(
    x: np.ndarray, axis: int = -1, keepdims: bool = True
) -> np.ndarray:
    """Compute Euclidean norm of vector with numerical stability."""
    eps = 1e-8
    return np.sqrt(np.square(x).sum(axis=axis, keepdims=keepdims) + eps)


def _np_extend(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    length: float,
    angle: float,
    dihedral: float,
) -> np.ndarray:
    """Compute 4th coordinate given 3 coordinates and internal geometry."""

    def normalize(x: np.ndarray) -> np.ndarray:
        return x / _np_norm(x)

    bc = normalize(b - c)
    n = normalize(np.cross(b - a, bc))

    d = c + (
        length * np.cos(angle) * bc
        + length * np.sin(angle) * np.cos(dihedral) * np.cross(n, bc)
        + length * np.sin(angle) * np.sin(dihedral) * (-n)
    )
    return d


def _compute_cb(
    n_coords: np.ndarray, ca_coords: np.ndarray, c_coords: np.ndarray
) -> np.ndarray:
    """Compute CB (C-beta) coordinates from backbone atoms."""
    return _np_extend(
        c_coords,
        n_coords,
        ca_coords,
        _CB_BOND_LENGTH,
        _CB_BOND_ANGLE,
        _CB_DIHEDRAL,
    )


def _get_structure(file_path: str) -> Structure:
    """Parse a structure file (PDB or CIF format)."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".cif":
        parser = MMCIFParser(QUIET=True)
        LOGGER.debug(f"Using MMCIFParser for {file_path}")
    elif suffix == ".pdb":
        parser = PDBParser(QUIET=True)
        LOGGER.debug(f"Using PDBParser for {file_path}")
    else:
        raise ValueError(
            f"Unrecognized file format: {suffix}. Expected .pdb or .cif"
        )

    return parser.get_structure("structure", file_path)


def _get_inputs_mpnn(file_path: str, chain: str | None = None) -> MPNNInputs:
    """Extract coordinates, residue info, and sequence from a PDB or CIF file.

    This function provides the same interface as
    softalign.Input_MPNN.get_inputs_mpnn but uses Biopython for parsing,
    enabling support for both PDB and CIF formats. It also extracts the
    amino acid sequence during parsing using the AA_3TO1 dictionary.

    Args:
        file_path: Path to the structure file (.pdb or .cif).
        chain: Chain identifier to extract. If None, uses first chain.

    Returns:
        MPNNInputs containing backbone coordinates and residue information.

    Raises:
        ValueError: If the specified chain is not found.
    """
    structure = _get_structure(file_path)

    # Get the first model
    struct_model = structure[0]

    # Find the target chain
    target_chain = None
    if chain is not None:
        for ch in struct_model:
            if ch.id == chain:
                target_chain = ch
                break
        if target_chain is None:
            available = [ch.id for ch in struct_model]
            raise ValueError(
                f"Chain '{chain}' not found in {file_path}. "
                f"Available chains: {available}"
            )
    else:
        # Use first chain
        target_chain = list(struct_model.get_chains())[0]
        LOGGER.info(f"No chain specified, using first chain: {target_chain.id}")

    # Extract residue data
    coords_list = []
    ids_list = []
    seq_list = []

    for residue in target_chain.get_residues():
        # Skip heteroatoms (water, ligands, etc.)
        hetflag = residue.get_id()[0]
        if hetflag.strip():
            continue

        # Check if all backbone atoms are present
        try:
            n_coord = residue["N"].get_coord()
            ca_coord = residue["CA"].get_coord()
            c_coord = residue["C"].get_coord()
        except KeyError:
            # Skip residues missing backbone atoms
            continue

        # Extract one-letter amino acid code (X for unknown residues)
        resname = residue.get_resname()
        one_letter = constants.AA_3TO1.get(resname, "X")
        seq_list.append(one_letter)

        # Compute CB position
        cb_coord = _compute_cb(
            n_coord.reshape(1, 3),
            ca_coord.reshape(1, 3),
            c_coord.reshape(1, 3),
        ).reshape(3)

        # Store coordinates [N, CA, C, CB]
        residue_coords = np.stack(
            [n_coord, ca_coord, c_coord, cb_coord], axis=0
        )
        coords_list.append(residue_coords)

        # Generate residue ID string
        res_id = residue.get_id()
        resnum = res_id[1]
        icode = res_id[2].strip()
        if icode:
            id_str = f"{resnum}{icode}"
        else:
            id_str = str(resnum)
        ids_list.append(id_str)

    if not coords_list:
        raise ValueError(
            f"No valid residues found in chain '{chain}' of {file_path}"
        )

    # Stack all coordinates
    coords = np.stack(coords_list, axis=0)  # [N, 4, 3]

    # Filter out any residues with NaN coordinates
    valid_mask = ~np.isnan(coords).any(axis=(1, 2))
    coords = coords[valid_mask]
    ids_list = [ids_list[i] for i in range(len(ids_list)) if valid_mask[i]]
    seq_list = [seq_list[i] for i in range(len(seq_list)) if valid_mask[i]]

    n_residues = coords.shape[0]

    # Create output arrays with batch dimension
    mask = np.ones(n_residues)
    chain_ids = np.ones(n_residues)
    residue_indices = np.arange(n_residues)

    sequence = "".join(seq_list)
    LOGGER.info(
        f"Extracted {n_residues} residues from chain '{target_chain.id}' "
        f"in {file_path}"
    )

    # Add batch dimension to match softalign output format
    return MPNNInputs(
        coords=coords[None, :],  # [1, N, 4, 3]
        mask=mask[None, :],  # [1, N]
        chain_ids=chain_ids[None, :],  # [1, N]
        residue_indices=residue_indices[None, :],  # [1, N]
        residue_ids=ids_list,
        sequence=sequence,
    )


@dataclass(frozen=True)
class MPNNEmbeddings:
    """Per-residue embedding tensor and matching residue identifiers.

    Can be instantiated from either:
    1. A PDB file (via from_pdb function)
    2. An NPZ file (via from_npz function)
    3. Direct construction with embeddings data
    """

    name: str
    embeddings: np.ndarray
    idxs: List[str]
    stdev: Optional[np.ndarray] = None
    sequence: Optional[str] = None

    def __post_init__(self) -> None:
        if self.embeddings.shape[0] != len(self.idxs):
            raise ValueError(
                f"embeddings.shape[0] ({self.embeddings.shape[0]}) must match "
                f"len(idxs) ({len(self.idxs)}). "
                f"Error raised for {self.name}"
            )
        if self.embeddings.shape[1] != constants.EMBED_DIM:
            raise ValueError(
                f"embeddings.shape[1] ({self.embeddings.shape[1]}) must match "
                f"constants.EMBED_DIM ({constants.EMBED_DIM}). "
                f"Error raised for {self.name}"
            )

        n_rows = self.embeddings.shape[0]
        processed_stdev = self._process_stdev(self.stdev, n_rows)
        object.__setattr__(self, "stdev", processed_stdev)

        LOGGER.debug(
            f"Initialized MPNNEmbeddings for {self.name} "
            f"(shape={self.embeddings.shape})"
        )

    def _process_stdev(
        self, stdev: Optional[np.ndarray], n_rows: int
    ) -> np.ndarray:
        """Process and validate stdev, returning a properly shaped array."""
        if stdev is None:
            return np.ones_like(self.embeddings)

        stdev = np.asarray(stdev)

        if stdev.ndim == 1:
            if stdev.shape[0] != constants.EMBED_DIM:
                raise ValueError(
                    f"1D stdev must have length {constants.EMBED_DIM}, "
                    f"got {stdev.shape[0]}"
                )
            return np.broadcast_to(stdev, (n_rows, constants.EMBED_DIM)).copy()

        if stdev.ndim == 2:
            if stdev.shape[1] != constants.EMBED_DIM:
                raise ValueError(
                    f"stdev.shape[1] ({stdev.shape[1]}) must match "
                    f"constants.EMBED_DIM ({constants.EMBED_DIM})"
                )
            if stdev.shape[0] == 1:
                return np.broadcast_to(
                    stdev, (n_rows, constants.EMBED_DIM)
                ).copy()
            if stdev.shape[0] < n_rows:
                raise ValueError(
                    f"stdev rows fewer than embeddings rows are not allowed: "
                    f"stdev rows={stdev.shape[0]}, embeddings rows={n_rows}"
                )
            if stdev.shape[0] > n_rows:
                return stdev[:n_rows, :].copy()
            return stdev

        raise ValueError(
            f"stdev must be 1D or 2D array compatible with embeddings, "
            f"got ndim={stdev.ndim}"
        )

    def save(self, output_path: str) -> None:
        """
        Save MPNNEmbeddings to an NPZ file.

        Args:
            output_path: Path where the NPZ file will be saved.
        """
        output_path_obj = Path(output_path)
        np.savez(
            output_path_obj,
            name=self.name,
            embeddings=self.embeddings,
            idxs=np.array(self.idxs),
            stdev=self.stdev,
            sequence=self.sequence if self.sequence else "",
        )
        LOGGER.info(f"Saved embeddings to {output_path_obj}")


def _embed_pdb(
    pdbfile: str, chains: str, max_residues: int = 0
) -> MPNNEmbeddings:
    """Return MPNN embeddings for chains in pdbfile using SoftAlign.

    Args:
        pdbfile: Path to the PDB file.
        chains: Chain identifier(s) to embed.
        max_residues: Maximum number of residues to embed. If 0, embed all.

    Returns:
        MPNNEmbeddings for the specified chain.
    """
    LOGGER.info(f"Embedding PDB {pdbfile} chain {chains}")
    e2e_model = model.create_e2e_model()
    if len(chains) > 1:
        raise NotImplementedError(
            f"Only single chain embedding is supported. "
            f"Got {len(chains)} chains: '{chains}'. "
            f"Please specify a single chain identifier."
        )
    inputs = _get_inputs_mpnn(pdbfile, chain=chains)
    embeddings = e2e_model.MPNN(
        inputs.coords, inputs.mask, inputs.chain_ids, inputs.residue_indices
    )[0]
    if len(inputs.residue_ids) != embeddings.shape[0]:
        raise ValueError(
            f"IDs length ({len(inputs.residue_ids)}) does not match embeddings "
            f"rows ({embeddings.shape[0]})"
        )

    ids = inputs.residue_ids
    sequence = inputs.sequence

    if max_residues > 0 and len(ids) > max_residues:
        LOGGER.info(
            f"Truncating embeddings from {len(ids)} to {max_residues} residues"
        )
        embeddings = embeddings[:max_residues]
        ids = ids[:max_residues]
        sequence = sequence[:max_residues]

    return MPNNEmbeddings(
        name="INPUT_PDB",
        embeddings=embeddings,
        idxs=ids,
        stdev=jnp.ones_like(embeddings),
        sequence=sequence,
    )


def from_pdb(
    pdb_file: str,
    chain: str,
    max_residues: int = 0,
    params_name: str = "CONT_SW_05_T_3_1",
    params_path: str = "softalign.models",
    random_seed: int = 0,
) -> MPNNEmbeddings:
    """
    Create MPNNEmbeddings from a PDB file.

    Args:
        pdb_file: Path to input PDB file.
        chain: Chain identifier to embed.
        max_residues: Maximum residues to embed. If 0, embed all.
        params_name: Name of the model parameters file.
        params_path: Package path containing the parameters file.
        random_seed: Random seed for JAX.

    Returns:
        MPNNEmbeddings for the specified chain.
    """
    model_params = util.read_softalign_params(
        params_name=params_name, params_path=params_path
    )
    key = jax.random.PRNGKey(random_seed)
    transformed_embed_fn = hk.transform(_embed_pdb)

    result = transformed_embed_fn.apply(
        model_params, key, pdb_file, chain, max_residues
    )

    LOGGER.info(
        f"Computed embeddings for {pdb_file} chain {chain} "
        f"(length={result.embeddings.shape[0]})"
    )
    return result


def from_npz(npz_file: str) -> MPNNEmbeddings:
    """
    Create MPNNEmbeddings from an NPZ file.

    Args:
        npz_file: Path to the NPZ file to load.

    Returns:
        MPNNEmbeddings object loaded from the file.
    """
    input_path = Path(npz_file)
    data = np.load(input_path, allow_pickle=True)

    name = str(data["name"])
    idxs = [str(idx) for idx in data["idxs"]]

    sequence = None
    if "sequence" in data:
        seq_str = str(data["sequence"])
        sequence = seq_str if seq_str else None

    embedding = MPNNEmbeddings(
        name=name,
        embeddings=data["embeddings"],
        idxs=idxs,
        stdev=data["stdev"],
        sequence=sequence,
    )
    LOGGER.info(
        f"Loaded embeddings from {input_path} "
        f"(name={name}, length={len(idxs)})"
    )
    return embedding
