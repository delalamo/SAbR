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
from typing import FrozenSet, List, Optional, Tuple

import numpy as np
from Bio.PDB import Chain

from sabr import constants
from sabr.embeddings.backend import EmbeddingBackend
from sabr.embeddings.inputs import get_inputs
from sabr.util import detect_backbone_gaps, validate_array_shape

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class MPNNEmbeddings:
    """Per-residue embedding tensor and matching residue identifiers.

    Can be instantiated from either:
    1. A PDB file (via from_pdb function)
    2. A BioPython Chain (via from_chain function)
    3. An NPZ file (via from_npz function)
    4. Direct construction with embeddings data

    Attributes:
        name: Identifier for the embedding source.
        embeddings: Per-residue embeddings array of shape [N, EMBED_DIM].
        idxs: List of residue ID strings matching embedding rows.
        sequence: Amino acid sequence as one-letter codes (optional).
        gap_indices: FrozenSet of row indices where structural gaps occur.
            Each index i means there is a gap AFTER residue i (between
            residue i and i+1). None if gap detection was not performed.
    """

    name: str
    embeddings: np.ndarray
    idxs: List[str]
    sequence: Optional[str] = None
    gap_indices: Optional[FrozenSet[int]] = None

    def __post_init__(self) -> None:
        validate_array_shape(
            self.embeddings,
            0,
            len(self.idxs),
            "embeddings",
            "len(idxs)",
            f"Error raised for {self.name}",
        )
        validate_array_shape(
            self.embeddings,
            1,
            constants.EMBED_DIM,
            "embeddings",
            "constants.EMBED_DIM",
            f"Error raised for {self.name}",
        )

        LOGGER.debug(
            f"Initialized MPNNEmbeddings for {self.name} "
            f"(shape={self.embeddings.shape})"
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
            sequence=self.sequence if self.sequence else "",
        )
        LOGGER.info(f"Saved embeddings to {output_path_obj}")


def _compute_gap_indices(
    coords: np.ndarray,
    keep_indices: Optional[List[int]] = None,
) -> Optional[FrozenSet[int]]:
    """Compute structural gap indices from backbone coordinates.

    Args:
        coords: Full backbone coordinates [1, N, 4, 3] or [N, 4, 3].
        keep_indices: If provided, only consider these residue indices.
            An empty list results in None (no residues to analyze).

    Returns:
        FrozenSet of gap indices after filtering, or None if < 2 residues.
    """
    # Normalize batch dimension - detect_backbone_gaps also handles this,
    # but we need 3D coords to do keep_indices slicing
    if coords.ndim == 4 and coords.shape[0] == 1:
        coords = coords[0]

    if keep_indices is not None:
        if len(keep_indices) < 2:
            return None
        coords = coords[keep_indices]

    if coords.shape[0] < 2:
        return None

    return detect_backbone_gaps(coords)


def _create_embeddings(
    inputs,
    name: str,
    residue_range: Tuple[int, int],
    params_name: str,
    params_path: str,
    random_seed: int,
) -> MPNNEmbeddings:
    """Create MPNNEmbeddings from extracted inputs.

    This is a shared helper for from_pdb and from_chain.

    Args:
        inputs: MPNNInputs with coordinates and residue info.
        name: Identifier for the embedding source.
        residue_range: Tuple of (start, end) residue numbers (inclusive).
            Use (0, 0) to embed all residues.
        params_name: Name of the model parameters file.
        params_path: Package path containing the parameters file.
        random_seed: Random seed for reproducibility.

    Returns:
        MPNNEmbeddings for the structure.
    """
    backend = EmbeddingBackend(
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
    keep_indices = None

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
            keep_indices = None

    # Compute gap indices from backbone coordinates
    gap_indices = _compute_gap_indices(inputs.coords, keep_indices)

    return MPNNEmbeddings(
        name=name,
        embeddings=embeddings,
        idxs=ids,
        sequence=sequence,
        gap_indices=gap_indices,
    )


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

    inputs = get_inputs(pdb_file, chain=chain)
    result = _create_embeddings(
        inputs,
        "INPUT_PDB",
        residue_range,
        params_name,
        params_path,
        random_seed,
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

    inputs = get_inputs(chain)
    result = _create_embeddings(
        inputs,
        "INPUT_CHAIN",
        residue_range,
        params_name,
        params_path,
        random_seed,
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
        sequence=sequence,
    )
    LOGGER.info(
        f"Loaded embeddings from {input_path} "
        f"(name={name}, length={len(idxs)})"
    )
    return embedding
