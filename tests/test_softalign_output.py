import numpy as np
import pytest

from sabr.alignment import SoftAlignOutput


def test_softalignoutput_valid_creation():
    """Test successful creation with matching shapes."""
    alignment = np.ones((3, 4), dtype=int)
    idxs1 = ["a", "b", "c"]
    idxs2 = ["1", "2", "3", "4"]

    output = SoftAlignOutput(
        alignment=alignment,
        score=1.5,
        sim_matrix=None,
        chain_type="mouse_H",
        idxs1=idxs1,
        idxs2=idxs2,
    )

    assert output.alignment.shape == (3, 4)
    assert output.score == pytest.approx(1.5)
    assert output.chain_type == "mouse_H"
    assert output.idxs1 == idxs1
    assert output.idxs2 == idxs2
    assert output.sim_matrix is None


def test_softalignoutput_with_sim_matrix():
    """Test creation with non-None sim_matrix."""
    alignment = np.ones((2, 2), dtype=int)
    sim_matrix = np.array([[0.8, 0.2], [0.3, 0.9]])
    idxs1 = ["x", "y"]
    idxs2 = ["1", "2"]

    output = SoftAlignOutput(
        alignment=alignment,
        score=2.0,
        sim_matrix=sim_matrix,
        chain_type=None,
        idxs1=idxs1,
        idxs2=idxs2,
    )

    assert output.sim_matrix is not None
    assert output.sim_matrix.shape == (2, 2)
    assert output.chain_type is None


def test_softalignoutput_alignment_idxs1_mismatch():
    """Test ValueError when alignment.shape[0] != len(idxs1)."""
    alignment = np.ones((3, 4), dtype=int)
    idxs1 = ["a", "b"]  # Only 2, but alignment has 3 rows
    idxs2 = ["1", "2", "3", "4"]

    with pytest.raises(ValueError) as excinfo:
        SoftAlignOutput(
            alignment=alignment,
            score=1.0,
            sim_matrix=None,
            chain_type="test",
            idxs1=idxs1,
            idxs2=idxs2,
        )

    msg = str(excinfo.value)
    assert "alignment.shape[0] (3) must match len(idxs1) (2)" in msg


def test_softalignoutput_alignment_idxs2_mismatch():
    """Test ValueError when alignment.shape[1] != len(idxs2)."""
    alignment = np.ones((3, 4), dtype=int)
    idxs1 = ["a", "b", "c"]
    idxs2 = ["1", "2"]  # Only 2, but alignment has 4 columns

    with pytest.raises(ValueError) as excinfo:
        SoftAlignOutput(
            alignment=alignment,
            score=1.0,
            sim_matrix=None,
            chain_type="test",
            idxs1=idxs1,
            idxs2=idxs2,
        )

    msg = str(excinfo.value)
    assert "alignment.shape[1] (4) must match len(idxs2) (2)" in msg


def test_softalignoutput_empty_alignment():
    """Test with empty alignment and empty idxs."""
    alignment = np.array([], dtype=int).reshape(0, 0)
    idxs1 = []
    idxs2 = []

    output = SoftAlignOutput(
        alignment=alignment,
        score=0.0,
        sim_matrix=None,
        chain_type=None,
        idxs1=idxs1,
        idxs2=idxs2,
    )

    assert output.alignment.shape == (0, 0)
    assert output.idxs1 == []
    assert output.idxs2 == []


def test_softalignoutput_single_element():
    """Test with minimal 1x1 alignment."""
    alignment = np.array([[1]], dtype=int)
    idxs1 = ["a"]
    idxs2 = ["1"]

    output = SoftAlignOutput(
        alignment=alignment,
        score=1.0,
        sim_matrix=None,
        chain_type="test",
        idxs1=idxs1,
        idxs2=idxs2,
    )

    assert output.alignment.shape == (1, 1)
    assert output.idxs1 == ["a"]
    assert output.idxs2 == ["1"]


def test_softalignoutput_numpy_array():
    """Test that regular numpy arrays also work (not just jax)."""
    alignment = np.ones((2, 3), dtype=int)
    idxs1 = ["x", "y"]
    idxs2 = ["1", "2", "3"]

    output = SoftAlignOutput(
        alignment=alignment,
        score=1.5,
        sim_matrix=None,
        chain_type="test",
        idxs1=idxs1,
        idxs2=idxs2,
    )

    assert output.alignment.shape == (2, 3)


def test_softalignoutput_negative_score():
    """Test that negative scores are allowed."""
    alignment = np.ones((2, 2), dtype=int)
    idxs1 = ["a", "b"]
    idxs2 = ["1", "2"]

    output = SoftAlignOutput(
        alignment=alignment,
        score=-5.0,
        sim_matrix=None,
        chain_type="test",
        idxs1=idxs1,
        idxs2=idxs2,
    )

    assert output.score == pytest.approx(-5.0)
