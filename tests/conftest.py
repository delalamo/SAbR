"""Shared test fixtures and utilities for SAbR tests."""

from typing import Any, Dict, List, Optional

import numpy as np


class DummyResult:
    """Mock alignment result for testing."""

    def __init__(self, alignment: np.ndarray, species: str) -> None:
        self.alignment = alignment
        self.species = species


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
    species: str,
    captured_kwargs: Optional[Dict[str, Any]] = None,
) -> Any:
    """Create a mock aligner that returns predetermined results.

    Args:
        alignment: The alignment matrix to return.
        species: The species name to return.
        captured_kwargs: Optional dict to capture kwargs passed to __call__.

    Returns:
        A DummyAligner class that can be instantiated.
    """

    class DummyAligner:
        def __call__(self, input_data: Any, **kwargs: Any) -> DummyResult:
            if captured_kwargs is not None:
                captured_kwargs.update(kwargs)
            return DummyResult(alignment, species)

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
        return DummyEmbeddings(
            name=f"{pdb_file}_{chain}",
            embeddings=np.zeros((n_residues, 64)),
            idxs=[str(i) for i in range(n_residues)],
            stdev=np.ones((n_residues, 64)),
            sequence="A" * n_residues,
        )

    return dummy_from_pdb
