import numpy as np
import pytest

from sabr.alignment.validation import validate_alignment_matrix
from sabr.errors import AlignmentError


def test_validate_alignment_matrix_rejects_duplicate_row_assignments():
    matrix = np.array([[1, 1], [0, 0]], dtype=int)

    with pytest.raises(AlignmentError, match="multiple reference positions"):
        validate_alignment_matrix(matrix)


def test_validate_alignment_matrix_allows_insertions_in_same_column():
    matrix = np.array([[1, 0], [1, 0], [0, 1]], dtype=int)

    validate_alignment_matrix(matrix)


def test_validate_alignment_matrix_rejects_non_monotonic_paths():
    matrix = np.zeros((2, 4), dtype=int)
    matrix[0, 3] = 1
    matrix[1, 1] = 1

    with pytest.raises(AlignmentError, match="monotonic"):
        validate_alignment_matrix(matrix)
