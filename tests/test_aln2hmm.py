import numpy as np
import pytest

from sabr import aln2hmm


def test_alignment_matrix_to_state_vector_basic():
    matrix = np.array([[1, 0], [0, 1]], dtype=int)

    states, b_start, a_end = aln2hmm.alignment_matrix_to_state_vector(matrix)

    assert states == [((1, "m"), 0)]
    assert b_start == 0
    assert a_end == 1


def test_alignment_matrix_to_state_vector_requires_2d():
    with pytest.raises(ValueError):
        aln2hmm.alignment_matrix_to_state_vector(np.ones(3))
