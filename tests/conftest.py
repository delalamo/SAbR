"""Shared test fixtures and utilities for SAbR tests."""

from typing import Any, Dict, List, Optional

import numpy as np
from Bio import SeqIO


class DummyResult:
    """Mock alignment result for testing."""

    def __init__(self, alignment: np.ndarray, chain_type: str) -> None:
        self.alignment = alignment
        self.chain_type = chain_type


class DummyEmbeddings:
    """Mock embeddings object for testing."""

    def __init__(
        self,
        name: str,
        embeddings: np.ndarray,
        idxs: List[str],
        stdev: Optional[np.ndarray] = None,
        sequence: Optional[str] = None,
    ) -> None:
        self.name = name
        self.embeddings = embeddings
        self.idxs = idxs
        self.stdev = stdev
        self.sequence = sequence


def create_dummy_aligner(
    alignment: np.ndarray,
    chain_type: str,
    captured_kwargs: Optional[Dict[str, Any]] = None,
) -> Any:
    """Create a mock aligner that returns predetermined results.

    Args:
        alignment: The alignment matrix to return.
        chain_type: The chain type to return.
        captured_kwargs: Optional dict to capture kwargs passed to __call__.

    Returns:
        A DummyAligner class that can be instantiated.
    """

    class DummyAligner:
        def __call__(self, input_data: Any, **kwargs: Any) -> DummyResult:
            if captured_kwargs is not None:
                captured_kwargs.update(kwargs)
            return DummyResult(alignment, chain_type)

    return DummyAligner


def create_dummy_from_pdb(n_residues: int = 100) -> Any:
    """Create a mock from_pdb function.

    Args:
        n_residues: Number of residues for the dummy embeddings.

    Returns:
        A function that returns DummyEmbeddings.
    """

    def dummy_from_pdb(
        pdb_file: str, chain: str, max_residues: int = 0, **kwargs: Any
    ) -> DummyEmbeddings:
        # Extract real sequence from PDB file to match alignment expectations
        sequence = None
        for record in SeqIO.parse(pdb_file, "pdb-atom"):
            if record.id.endswith(chain):
                sequence = str(record.seq).replace("X", "")
                break
        if sequence is None:
            # Fallback to dummy sequence if chain not found
            sequence = "A" * n_residues

        actual_n_residues = len(sequence)
        if max_residues > 0:
            actual_n_residues = min(actual_n_residues, max_residues)
            sequence = sequence[:max_residues]

        return DummyEmbeddings(
            name=f"{pdb_file}_{chain}",
            embeddings=np.zeros((actual_n_residues, 64)),
            idxs=[str(i) for i in range(actual_n_residues)],
            stdev=np.ones((actual_n_residues, 64)),
            sequence=sequence,
        )

    return dummy_from_pdb
