#!/usr/bin/env python3
"""Embedding generation for SAbR.

This package provides the embedding functionality for SAbR including:
- MPNNEmbeddings: Dataclass for per-residue embeddings
- EmbeddingBackend: JAX/Haiku backend for embedding computation
- Input extraction from PDB/CIF files and BioPython chains
"""

from sabr.embeddings.backend import EmbeddingBackend
from sabr.embeddings.inputs import (
    MPNNInputs,
    compute_cb,
    extract_from_biopython_chain,
    extract_from_gemmi_chain,
    get_inputs,
)
from sabr.embeddings.mpnn import (
    MPNNEmbeddings,
    from_chain,
    from_npz,
    from_pdb,
)

__all__ = [
    # Main classes
    "MPNNEmbeddings",
    "MPNNInputs",
    "EmbeddingBackend",
    # Functions
    "from_pdb",
    "from_chain",
    "from_npz",
    "get_inputs",
    "compute_cb",
    "extract_from_gemmi_chain",
    "extract_from_biopython_chain",
]
