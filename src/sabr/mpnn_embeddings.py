#!/usr/bin/env python3
"""MPNN embedding generation and management module.

This module provides the MPNNEmbeddings dataclass and functions for
generating, saving, and loading neural network embeddings from protein
structures using the MPNN (Message Passing Neural Network) architecture.

Key components:
- MPNNEmbeddings: Dataclass for storing per-residue embeddings
- from_pdb: Generate embeddings from a PDB or CIF file
- from_chain: Generate embeddings from a BioPython Chain object
- from_npz: Load pre-computed embeddings from NumPy archive

Embeddings are 64-dimensional vectors computed for each residue,
capturing structural and sequence features for alignment.

Supported file formats:
- PDB (.pdb): Standard Protein Data Bank format
- mmCIF (.cif): Macromolecular Crystallographic Information File format
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from Bio.PDB import Chain, MMCIFParser, PDBParser
from Bio.PDB.Structure import Structure

from sabr import constants, jax_backend

LOGGER = logging.getLogger(__name__)


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


def _compute_cb(
    n_coords: np.ndarray, ca_coords: np.ndarray, c_coords: np.ndarray
) -> np.ndarray:
    """Compute CB (C-beta) coordinates from backbone atoms.

    Uses standard protein geometry constants to calculate the CB position
    from N, CA, and C backbone atom coordinates.

    Args:
        n_coords: N atom coordinates [1, 3] or [3].
        ca_coords: CA atom coordinates [1, 3] or [3].
        c_coords: C atom coordinates [1, 3] or [3].

    Returns:
        CB coordinates with same shape as input.
    """
    eps = 1e-8

    def normalize(x: np.ndarray) -> np.ndarray:
        norm = np.sqrt(np.square(x).sum(axis=-1, keepdims=True) + eps)
        return x / norm

    # Compute CB position using internal geometry
    # a=C, b=N, c=CA for the extension calculation
    bc = normalize(n_coords - ca_coords)
    n = normalize(np.cross(n_coords - c_coords, bc))

    length = constants.CB_BOND_LENGTH
    angle = constants.CB_BOND_ANGLE
    dihedral = constants.CB_DIHEDRAL

    cb = ca_coords + (
        length * np.cos(angle) * bc
        + length * np.sin(angle) * np.cos(dihedral) * np.cross(n, bc)
        + length * np.sin(angle) * np.sin(dihedral) * (-n)
    )
    return cb


def _parse_structure(file_path: str) -> Structure:
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


def _extract_inputs_from_chain(
    target_chain: Chain.Chain, source_name: str = ""
) -> MPNNInputs:
    """Extract coordinates, residue info, and sequence from a Chain object.

    This is the core extraction logic used by both file-based and
    structure-based input functions.

    Args:
        target_chain: BioPython Chain object to extract from.
        source_name: Source identifier for logging (file path or "structure").

    Returns:
        MPNNInputs containing backbone coordinates and residue information.

    Raises:
        ValueError: If no valid residues are found.
    """
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
            f"No valid residues found in chain '{target_chain.id}'"
            + (f" of {source_name}" if source_name else "")
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
    log_msg = f"Extracted {n_residues} residues from chain '{target_chain.id}'"
    if source_name:
        log_msg += f" in {source_name}"
    LOGGER.info(log_msg)

    # Add batch dimension to match softalign output format
    return MPNNInputs(
        coords=coords[None, :],  # [1, N, 4, 3]
        mask=mask[None, :],  # [1, N]
        chain_ids=chain_ids[None, :],  # [1, N]
        residue_indices=residue_indices[None, :],  # [1, N]
        residue_ids=ids_list,
        sequence=sequence,
    )


def _get_inputs(
    source: Union[str, Chain.Chain], chain: str | None = None
) -> MPNNInputs:
    """Extract MPNN inputs from a file path or Chain object.

    Args:
        source: Either a file path (str) or BioPython Chain object.
        chain: Chain identifier to extract (only used for file paths).

    Returns:
        MPNNInputs containing backbone coordinates and residue information.
    """
    if isinstance(source, str):
        structure = _parse_structure(source)
        source_name = source
        struct_model = structure[0]

        if chain is not None:
            for ch in struct_model:
                if ch.id == chain:
                    return _extract_inputs_from_chain(ch, source_name)
            available = [ch.id for ch in struct_model]
            raise ValueError(
                f"Chain '{chain}' not found in {source_name}. "
                f"Available chains: {available}"
            )
        else:
            target_chain = list(struct_model.get_chains())[0]
            LOGGER.info(
                f"No chain specified, using first chain: {target_chain.id}"
            )
            return _extract_inputs_from_chain(target_chain, source_name)
    else:
        # source is a Chain object
        return _extract_inputs_from_chain(source, "")


@dataclass(frozen=True)
class MPNNEmbeddings:
    """Per-residue embedding tensor and matching residue identifiers.

    Can be instantiated from either:
    1. A PDB file (via from_pdb function)
    2. A BioPython Chain (via from_chain function)
    3. An NPZ file (via from_npz function)
    4. Direct construction with embeddings data
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


def from_pdb(
    pdb_file: str,
    chain: str,
    residue_range: Tuple[int, int] = (0, 0),
    params_name: str = "mpnn_encoder",
    params_path: str = "sabr.assets",
    random_seed: int = 0,
) -> MPNNEmbeddings:
    """
    Create MPNNEmbeddings from a PDB file.

    Args:
        pdb_file: Path to input PDB file (.pdb or .cif).
        chain: Chain identifier to embed.
        residue_range: Tuple of (start, end) residue numbers in PDB numbering
            (inclusive). Use (0, 0) to embed all residues.
        params_name: Name of the model parameters file.
        params_path: Package path containing the parameters file.
        random_seed: Random seed for reproducibility.

    Returns:
        MPNNEmbeddings for the specified chain.
    """
    LOGGER.info(f"Embedding PDB {pdb_file} chain {chain}")

    if len(chain) > 1:
        raise NotImplementedError(
            f"Only single chain embedding is supported. "
            f"Got {len(chain)} chains: '{chain}'. "
            f"Please specify a single chain identifier."
        )

    # Parse structure and extract inputs
    inputs = _get_inputs(pdb_file, chain=chain)

    # Create backend and compute embeddings
    backend = jax_backend.EmbeddingBackend(
        params_name=params_name,
        params_path=params_path,
        random_seed=random_seed,
    )

    embeddings = backend.compute_embeddings(
        coords=inputs.coords,
        mask=inputs.mask,
        chain_ids=inputs.chain_ids,
        residue_indices=inputs.residue_indices,
    )

    if len(inputs.residue_ids) != embeddings.shape[0]:
        raise ValueError(
            f"IDs length ({len(inputs.residue_ids)}) does not match embeddings "
            f"rows ({embeddings.shape[0]})"
        )

    ids = inputs.residue_ids
    sequence = inputs.sequence

    # Filter by residue range if specified
    start_res, end_res = residue_range
    if residue_range != (0, 0):
        # Find indices where residue numbers fall within range
        keep_indices = []
        for i, res_id in enumerate(ids):
            try:
                res_num = int(res_id)
                if start_res <= res_num <= end_res:
                    keep_indices.append(i)
            except ValueError:
                # Skip residues with non-numeric IDs (e.g., insertion codes)
                continue

        if keep_indices:
            LOGGER.info(
                f"Filtering to residue range {start_res}-{end_res}: "
                f"{len(keep_indices)} of {len(ids)} residues"
            )
            embeddings = embeddings[keep_indices]
            ids = [ids[i] for i in keep_indices]
            sequence = "".join(sequence[i] for i in keep_indices)
        else:
            LOGGER.warning(f"No residues found in range {start_res}-{end_res}")

    result = MPNNEmbeddings(
        name="INPUT_PDB",
        embeddings=embeddings,
        idxs=ids,
        stdev=np.ones_like(embeddings),
        sequence=sequence,
    )

    LOGGER.info(
        f"Computed embeddings for {pdb_file} chain {chain} "
        f"(length={result.embeddings.shape[0]})"
    )
    return result


def from_chain(
    chain: Chain.Chain,
    residue_range: Tuple[int, int] = (0, 0),
    params_name: str = "mpnn_encoder",
    params_path: str = "sabr.assets",
    random_seed: int = 0,
) -> MPNNEmbeddings:
    """
    Create MPNNEmbeddings from a BioPython Chain object.

    This function enables in-memory structure processing without requiring
    the structure to be saved to disk first.

    Args:
        chain: BioPython Chain object.
        residue_range: Tuple of (start, end) residue numbers in PDB numbering
            (inclusive). Use (0, 0) to embed all residues.
        params_name: Name of the model parameters file.
        params_path: Package path containing the parameters file.
        random_seed: Random seed for reproducibility.

    Returns:
        MPNNEmbeddings for the chain.
    """
    LOGGER.info(f"Embedding chain {chain.id}")

    # Extract inputs from chain
    inputs = _get_inputs(chain)

    # Create backend and compute embeddings
    backend = jax_backend.EmbeddingBackend(
        params_name=params_name,
        params_path=params_path,
        random_seed=random_seed,
    )

    embeddings = backend.compute_embeddings(
        coords=inputs.coords,
        mask=inputs.mask,
        chain_ids=inputs.chain_ids,
        residue_indices=inputs.residue_indices,
    )

    if len(inputs.residue_ids) != embeddings.shape[0]:
        raise ValueError(
            f"IDs length ({len(inputs.residue_ids)}) does not match embeddings "
            f"rows ({embeddings.shape[0]})"
        )

    ids = inputs.residue_ids
    sequence = inputs.sequence

    # Filter by residue range if specified
    start_res, end_res = residue_range
    if residue_range != (0, 0):
        keep_indices = []
        for i, res_id in enumerate(ids):
            try:
                res_num = int(res_id)
                if start_res <= res_num <= end_res:
                    keep_indices.append(i)
            except ValueError:
                continue

        if keep_indices:
            LOGGER.info(
                f"Filtering to residue range {start_res}-{end_res}: "
                f"{len(keep_indices)} of {len(ids)} residues"
            )
            embeddings = embeddings[keep_indices]
            ids = [ids[i] for i in keep_indices]
            sequence = "".join(sequence[i] for i in keep_indices)
        else:
            LOGGER.warning(f"No residues found in range {start_res}-{end_res}")

    result = MPNNEmbeddings(
        name="INPUT_CHAIN",
        embeddings=embeddings,
        idxs=ids,
        stdev=np.ones_like(embeddings),
        sequence=sequence,
    )

    LOGGER.info(
        f"Computed embeddings for chain {chain.id} "
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

    sequence = str(data["sequence"]) or None if "sequence" in data else None

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
