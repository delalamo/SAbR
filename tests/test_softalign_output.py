import numpy as np
import pytest

from sabr.alignment.soft_aligner import AlignmentResult
from sabr.types import ChainType


def test_alignment_result_valid_creation():
    alignment = np.ones((3, 4), dtype=int)

    output = AlignmentResult(
        alignment=alignment,
        score=1.5,
        selected_chain_type=ChainType.HEAVY,
        sim_matrix=None,
    )

    assert output.alignment.shape == (3, 4)
    assert output.score == pytest.approx(1.5)
    assert output.selected_chain_type is ChainType.HEAVY
    assert output.sim_matrix is None


def test_alignment_result_with_sim_matrix():
    alignment = np.ones((2, 2), dtype=int)
    sim_matrix = np.array([[0.8, 0.2], [0.3, 0.9]])

    output = AlignmentResult(
        alignment=alignment,
        score=2.0,
        selected_chain_type=ChainType.KAPPA,
        sim_matrix=sim_matrix,
    )

    assert output.sim_matrix is not None
    assert output.sim_matrix.shape == (2, 2)
    assert output.selected_chain_type is ChainType.KAPPA


def test_alignment_result_requires_two_dimensional_alignment():
    with pytest.raises(ValueError, match="two-dimensional"):
        AlignmentResult(
            alignment=np.ones(3),
            score=1.0,
            selected_chain_type=ChainType.LAMBDA,
            sim_matrix=None,
        )


def test_alignment_result_negative_score():
    output = AlignmentResult(
        alignment=np.ones((2, 2), dtype=int),
        score=-5.0,
        selected_chain_type=ChainType.HEAVY,
        sim_matrix=None,
    )

    assert output.score == pytest.approx(-5.0)
