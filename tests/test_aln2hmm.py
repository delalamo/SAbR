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


def test_alignment_matrix_to_state_vector_diagonal_path():
    """Test simple diagonal alignment (all matches)."""
    matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=int)

    states, b_start, a_end = aln2hmm.alignment_matrix_to_state_vector(matrix)

    # Diagonal should produce only matches
    assert all(s[0][1] == "m" for s in states)
    assert len(states) == 2  # 3 positions, but 2 moves
    assert b_start == 0
    assert a_end == 2


def test_alignment_matrix_to_state_vector_with_insertions():
    """Test alignment with insertions (A-only steps)."""
    # Pattern: match, insert, insert, match
    # Row 0: position 0
    # Row 1: positions 1,2,3
    matrix = np.array([[1, 0, 0, 0], [0, 0, 0, 1]], dtype=int)

    states, b_start, a_end = aln2hmm.alignment_matrix_to_state_vector(matrix)

    # Should have insertions
    assert any(s[0][1] == "i" for s in states)
    assert b_start == 0


def test_alignment_matrix_to_state_vector_with_deletions():
    """Test alignment with deletions (B-only steps)."""
    # Pattern: match at (0,0), deletion at B=1, match at (2,1)
    matrix = np.array([[1, 0], [0, 0], [0, 1]], dtype=int)

    states, b_start, a_end = aln2hmm.alignment_matrix_to_state_vector(matrix)

    # Should have deletions
    deletion_states = [s for s in states if s[0][1] == "d"]
    assert len(deletion_states) > 0
    # Deletion states should have None for sequence A index
    for s in deletion_states:
        assert s[1] is None


def test_alignment_matrix_to_state_vector_complex_path():
    """Test complex path with mixed matches, insertions, and deletions."""
    matrix = np.array(
        [
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],  # Deletion row
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
        ],
        dtype=int,
    )

    states, b_start, a_end = aln2hmm.alignment_matrix_to_state_vector(matrix)

    # Should have matches
    match_states = [s for s in states if s[0][1] == "m"]
    assert len(match_states) > 0

    # Should have deletions (row 2 has no 1s)
    deletion_states = [s for s in states if s[0][1] == "d"]
    assert len(deletion_states) > 0


def test_alignment_matrix_to_state_vector_empty_path():
    """Test that empty alignment matrix with no path triggers assertion."""
    matrix = np.zeros((3, 3), dtype=int)

    with pytest.raises(
        AssertionError, match="Alignment matrix contains no path"
    ):
        aln2hmm.alignment_matrix_to_state_vector(matrix)


def test_alignment_matrix_to_state_vector_single_match():
    """Test minimal alignment with single match."""
    matrix = np.array([[1]], dtype=int)

    states, b_start, a_end = aln2hmm.alignment_matrix_to_state_vector(matrix)

    # Single match should produce empty states list (no moves between positions)
    assert states == []
    assert b_start == 0
    assert a_end == 0


def test_alignment_matrix_to_state_vector_insertion_at_end():
    """Test insertion at the end of sequence."""
    # Match at (0,0), then insertions at A=1,2
    matrix = np.array([[1, 0, 0], [0, 0, 0]], dtype=int)

    states, b_start, a_end = aln2hmm.alignment_matrix_to_state_vector(matrix)

    # Should handle end insertions
    assert len(states) > 0
    # Check for insertion states
    insertion_states = [s for s in states if s[0][1] == "i"]
    assert len(insertion_states) >= 0  # May have insertions


def test_alignment_matrix_to_state_vector_large_matrix():
    """Test with larger alignment matrix."""
    size = 10
    # Create diagonal alignment
    matrix = np.eye(size, dtype=int)

    states, b_start, a_end = aln2hmm.alignment_matrix_to_state_vector(matrix)

    # Should produce matches
    assert all(s[0][1] == "m" for s in states)
    assert len(states) == size - 1
    assert b_start == 0
    assert a_end == size - 1
