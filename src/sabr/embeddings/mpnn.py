#!/usr/bin/env python3
"""MPNN embedding generation and management module.

This module provides the QueryEmbeddings dataclass and functions for
generating, saving, and loading neural network embeddings from protein
structures using the MPNN (Message Passing Neural Network) architecture.

Key components:
- QueryEmbeddings: Dataclass for storing per-residue query embeddings
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
from typing import FrozenSet, List, Optional

import numpy as np
from Bio.PDB import Chain

from sabr.alignment.gaps import detect_backbone_gaps
from sabr.embeddings.backend import EmbeddingBackend
from sabr.embeddings.inputs import get_inputs
from sabr.structure.residues import ResidueRange

LOGGER = logging.getLogger(__name__)
QUERY_SCHEMA = "query-v1"
EMBED_DIM = 64


def _validate_array_shape(
    array: np.ndarray,
    dim: int,
    expected_size: int,
    array_name: str,
    size_name: str,
    context: str = "",
) -> None:
    actual_size = array.shape[dim]
    if actual_size != expected_size:
        msg = (
            f"{array_name}.shape[{dim}] ({actual_size}) must match "
            f"{size_name} ({expected_size})."
        )
        if context:
            msg += f" {context}"
        raise ValueError(msg)


@dataclass(frozen=True)
class QueryEmbeddings:
    """Per-residue query embedding tensor and matching residue identifiers.

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
        _validate_array_shape(
            self.embeddings,
            0,
            len(self.idxs),
            "embeddings",
            "len(idxs)",
            f"Error raised for {self.name}",
        )
        _validate_array_shape(
            self.embeddings,
            1,
            EMBED_DIM,
            "embeddings",
            "EMBED_DIM",
            f"Error raised for {self.name}",
        )

        LOGGER.debug(
            f"Initialized QueryEmbeddings for {self.name} "
            f"(shape={self.embeddings.shape})"
        )

    def save(self, output_path: str) -> None:
        """
        Save query embeddings to an NPZ file.

        Args:
            output_path: Path where the NPZ file will be saved.
        """
        np.savez(
            Path(output_path),
            schema=QUERY_SCHEMA,
            name=self.name,
            embeddings=self.embeddings,
            idxs=np.array(self.idxs),
            sequence=self.sequence if self.sequence else "",
        )
        LOGGER.info(f"Saved embeddings to {output_path}")


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
    residue_range: ResidueRange | None,
    params_name: str,
    params_path: str,
    random_seed: int,
    backend: EmbeddingBackend | None = None,
) -> QueryEmbeddings:
    """Create query embeddings from extracted inputs.

    This is a shared helper for from_pdb and from_chain.

    Args:
        inputs: MPNNInputs with coordinates and residue info.
        name: Identifier for the embedding source.
        residue_range: Tuple of (start, end) residue numbers (inclusive).
            Pass None to embed all residues.
        params_name: Name of the model parameters file.
        params_path: Package path containing the parameters file.
        random_seed: Random seed for reproducibility.

    Returns:
        QueryEmbeddings for the structure.
    """
    if backend is None:
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

    if residue_range is not None:
        keep_indices = [
            i for i, res_id in enumerate(ids) if residue_range.contains(res_id)
        ]

        if keep_indices:
            LOGGER.info(
                f"Filtering to residue range {residue_range}: "
                f"{len(keep_indices)} of {len(ids)} residues"
            )
            embeddings = embeddings[keep_indices]
            ids = [ids[i] for i in keep_indices]
            sequence = "".join(sequence[i] for i in keep_indices)
        else:
            LOGGER.warning(f"No residues found in range {residue_range}")
            keep_indices = None

    # Compute gap indices from backbone coordinates
    gap_indices = _compute_gap_indices(inputs.coords, keep_indices)

    return QueryEmbeddings(
        name=name,
        embeddings=embeddings,
        idxs=ids,
        sequence=sequence,
        gap_indices=gap_indices,
    )


def from_pdb(
    pdb_file: str,
    chain: str,
    residue_range: ResidueRange | None = None,
    params_name: str = "mpnn_encoder",
    params_path: str = "sabr.assets",
    random_seed: int = 0,
    backend: EmbeddingBackend | None = None,
) -> QueryEmbeddings:
    """
    Create query embeddings from a PDB file.

    Args:
        pdb_file: Path to input PDB file (.pdb or .cif).
        chain: Chain identifier to embed.
        residue_range: Tuple of (start, end) residue numbers in PDB numbering
            (inclusive). Pass None to embed all residues.
        params_name: Name of the model parameters file.
        params_path: Package path containing the parameters file.
        random_seed: Random seed for reproducibility.

    Returns:
        QueryEmbeddings for the specified chain.
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
        backend,
    )

    LOGGER.info(
        f"Computed embeddings for {pdb_file} chain {chain} "
        f"(length={result.embeddings.shape[0]})"
    )
    return result


def from_chain(
    chain: Chain.Chain,
    residue_range: ResidueRange | None = None,
    params_name: str = "mpnn_encoder",
    params_path: str = "sabr.assets",
    random_seed: int = 0,
    backend: EmbeddingBackend | None = None,
) -> QueryEmbeddings:
    """
    Create query embeddings from a BioPython Chain object.

    This function enables in-memory structure processing without requiring
    the structure to be saved to disk first.

    Args:
        chain: BioPython Chain object.
        residue_range: Tuple of (start, end) residue numbers in PDB numbering
            (inclusive). Pass None to embed all residues.
        params_name: Name of the model parameters file.
        params_path: Package path containing the parameters file.
        random_seed: Random seed for reproducibility.

    Returns:
        QueryEmbeddings for the chain.
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
        backend,
    )

    LOGGER.info(
        f"Computed embeddings for chain {chain.id} "
        f"(length={result.embeddings.shape[0]})"
    )
    return result


def from_npz(npz_file: str) -> QueryEmbeddings:
    """
    Create query embeddings from an NPZ file.

    Args:
        npz_file: Path to the NPZ file to load.

    Returns:
        QueryEmbeddings object loaded from the file.
    """
    input_path = Path(npz_file)
    data = np.load(input_path, allow_pickle=True)
    required = {"schema", "name", "embeddings", "idxs", "sequence"}
    missing = required - set(data.files)
    if missing:
        missing_keys = ", ".join(sorted(missing))
        raise ValueError(f"Embedding file {input_path} is missing keys: {missing_keys}")
    schema = str(data["schema"])
    if schema != QUERY_SCHEMA:
        raise ValueError(
            f"Embedding file {input_path} uses unsupported schema {schema!r}."
        )
    sequence = str(data["sequence"])
    embedding = QueryEmbeddings(
        name=str(data["name"]),
        embeddings=data["embeddings"],
        idxs=[str(x) for x in data["idxs"]],
        sequence=sequence or None,
    )
    LOGGER.info(
        f"Loaded embeddings from {npz_file} "
        f"(name={embedding.name}, length={len(embedding.idxs)})"
    )
    return embedding
