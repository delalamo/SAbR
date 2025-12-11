import tempfile
from pathlib import Path

import pytest

from sabr import util

TEST_DATA_DIR = Path(__file__).parent / "data"


def create_minimal_pdb(chain_id: str, sequence: str) -> str:
    """Create a minimal PDB file content with a given chain."""
    pdb_lines = [
        "HEADER    TEST STRUCTURE",
        f"ATOM      1  CA  ALA {chain_id}   1       "
        f"0.000   0.000   0.000  1.00  0.00           C  ",
    ]
    return "\n".join(pdb_lines)


def test_fetch_sequence_from_pdb_chain_not_found():
    """Test ValueError when chain is not found in PDB file."""
    pdb_content = create_minimal_pdb("A", "ACDEFG")

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".pdb", delete=False
    ) as f:
        f.write(pdb_content)
        pdb_path = f.name

    try:
        with pytest.raises(ValueError) as excinfo:
            util.fetch_sequence_from_pdb(pdb_path, "Z")

        msg = str(excinfo.value)
        assert "Chain Z not found" in msg
        assert pdb_path in msg
    finally:
        Path(pdb_path).unlink()


def test_fetch_sequence_from_pdb_filters_x_residues():
    """Test that X residues are removed from the sequence."""
    pdb_path = TEST_DATA_DIR / "test_x_residues.pdb"
    sequence = util.fetch_sequence_from_pdb(str(pdb_path), "A")
    # UNK residues are typically represented as X and should be filtered
    assert "X" not in sequence
    # Should have ALA and GLY at minimum
    assert len(sequence) >= 2


def test_fetch_sequence_from_pdb_empty_file():
    """Test behavior with empty PDB file."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".pdb", delete=False
    ) as f:
        f.write("")
        pdb_path = f.name

    try:
        with pytest.raises(ValueError) as excinfo:
            util.fetch_sequence_from_pdb(pdb_path, "A")

        # BioPython may raise "Empty file." for empty PDB files
        error_msg = str(excinfo.value)
        assert "Chain A not found" in error_msg or "Empty file" in error_msg
    finally:
        Path(pdb_path).unlink()


def test_fetch_sequence_from_pdb_nonexistent_file():
    """Test behavior when PDB file doesn't exist."""
    with pytest.raises((FileNotFoundError, ValueError)):
        util.fetch_sequence_from_pdb("/nonexistent/path/file.pdb", "A")


def test_fetch_sequence_from_pdb_multiple_chains():
    """Test extraction from PDB with multiple chains."""
    pdb_path = TEST_DATA_DIR / "test_multi_chain.pdb"

    # Should find chain A
    seq_a = util.fetch_sequence_from_pdb(str(pdb_path), "A")
    assert len(seq_a) >= 1

    # Should find chain B
    seq_b = util.fetch_sequence_from_pdb(str(pdb_path), "B")
    assert len(seq_b) >= 1

    # Should find chain C
    seq_c = util.fetch_sequence_from_pdb(str(pdb_path), "C")
    assert len(seq_c) >= 1

    # Chain D should not be found
    with pytest.raises(ValueError):
        util.fetch_sequence_from_pdb(str(pdb_path), "D")
