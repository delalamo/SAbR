"""Shared test fixtures and utilities for SAbR tests."""

from importlib import resources
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gemmi
import numpy as np
import pytest
from Bio import PDB, SeqIO

DATA_PACKAGE = "tests.data"


def resolve_data_path(filename: str) -> Path:
    """Resolve a test data file path."""
    return Path(resources.files(DATA_PACKAGE) / filename)


# Shared fixtures dictionary with test data paths and metadata
FIXTURES = {
    "8_21": {
        "pdb": resolve_data_path("8_21_renumbered.pdb"),
        "chain": "A",
        "alignment": resolve_data_path("8_21_renumbered_alignment.npz"),
        "embeddings": resolve_data_path("8_21_renumbered_embeddings.npz"),
        "min_deviations": 0,
        "max_deviations": 0,
    },
    "5omm": {
        "pdb": resolve_data_path("5omm_imgt.pdb"),
        "chain": "C",
        "alignment": resolve_data_path("5omm_imgt_alignment.npz"),
        "embeddings": resolve_data_path("5omm_imgt_embeddings.npz"),
        "min_deviations": 5,
        "max_deviations": 200,
    },
    "test_heavy_chain": {
        "pdb": resolve_data_path("test_heavy_chain.pdb"),
        "chain": "F",
        "alignment": resolve_data_path("test_heavy_chain_alignment.npz"),
        "embeddings": resolve_data_path("test_heavy_chain_embeddings.npz"),
        "min_deviations": 0,
        "max_deviations": 25,
    },
}


def load_alignment_fixture(path: Path) -> Tuple[np.ndarray, str]:
    """Load an alignment fixture from disk."""
    if not path.exists():
        pytest.skip(f"Missing alignment fixture at {path}")
    data = np.load(path, allow_pickle=True)
    alignment = data["alignment"]
    chain_type = data["chain_type"].item()
    return alignment, chain_type


def extract_residue_ids_from_pdb(
    pdb_path: Path, chain: str
) -> List[Tuple[str, int, str]]:
    """Extract residue IDs from a PDB file path using Gemmi."""
    structure = gemmi.read_structure(str(pdb_path))
    return extract_residue_ids_from_gemmi_structure(structure, chain)


def extract_residue_ids_from_gemmi_structure(
    structure: gemmi.Structure, chain: str
) -> List[Tuple[str, int, str]]:
    """Extract residue IDs from a Gemmi structure."""
    residues = []
    model = structure[0]
    for ch in model:
        if ch.name != chain:
            continue
        for res in ch:
            if res.het_flag != "A":  # Skip non-amino acid residues
                continue
            # Map Gemmi het_flag to BioPython format: 'A' -> ' '
            hetflag = " " if res.het_flag == "A" else "H_" + res.name
            resnum = res.seqid.num
            icode = res.seqid.icode if res.seqid.icode.strip() else " "
            residues.append((hetflag, resnum, icode))
    return residues


def extract_residue_ids_from_structure(
    structure: PDB.Structure.Structure, chain: str
) -> List[Tuple[str, int, str]]:
    """Extract residue IDs from a BioPython structure."""
    residues = []
    for res in structure[0][chain]:
        hetflag, resseq, icode = res.get_id()
        if hetflag.strip():
            continue
        residues.append((hetflag, resseq, icode))
    return residues


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
        sequence: Optional[str] = None,
        gap_indices: Optional[frozenset] = None,
    ) -> None:
        self.name = name
        self.embeddings = embeddings
        self.idxs = idxs
        self.sequence = sequence
        self.gap_indices = gap_indices


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
        pdb_file: str,
        chain: str,
        residue_range: tuple = (0, 0),
        **kwargs: Any,
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
        start_res, end_res = residue_range
        if residue_range != (0, 0):
            # Filter by residue range
            actual_n_residues = min(actual_n_residues, end_res - start_res + 1)
            sequence = sequence[: end_res - start_res + 1]

        return DummyEmbeddings(
            name=f"{pdb_file}_{chain}",
            embeddings=np.zeros((actual_n_residues, 64)),
            idxs=[str(i) for i in range(actual_n_residues)],
            sequence=sequence,
        )

    return dummy_from_pdb
