import numpy as np
import pytest

from sabr import aln2hmm


def test_alignment_matrix_to_state_vector_basic():
    matrix = np.array([[1, 0], [0, 1]], dtype=int)

    states, b_start, a_end = aln2hmm.alignment_matrix_to_state_vector(matrix)

    assert len(states) == 1
    assert states[0].residue_number == 1
    assert states[0].insertion_code == "m"
    assert states[0].mapped_residue == 0
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
    # Matrix is transposed before processing, so insertions occur when
    # the transposed matrix has multiple 1s in same row
    # After transpose: row 0 will have positions at columns 0,1,2
    matrix = np.array([[1, 0], [1, 0], [1, 0], [0, 1]], dtype=int)

    states, b_start, a_end = aln2hmm.alignment_matrix_to_state_vector(matrix)

    # Should have insertions
    assert any(s[0][1] == "i" for s in states)
    assert b_start == 0


def test_alignment_matrix_to_state_vector_with_deletions():
    """Test alignment with deletions (B-only steps)."""
    # Matrix is transposed before processing, so deletions occur when
    # the transposed matrix has multiple 1s in same column
    # After transpose: column 0 will have positions at rows 0,1,2
    matrix = np.array([[1, 1, 1, 0], [0, 0, 0, 1]], dtype=int)

    states, b_start, a_end = aln2hmm.alignment_matrix_to_state_vector(matrix)

    # Should have deletions
    deletion_states = [s for s in states if s[0][1] == "d"]
    assert len(deletion_states) > 0
    # Deletion states should have None for sequence A index
    for s in deletion_states:
        assert s[1] is None


def test_alignment_matrix_to_state_vector_complex_path():
    """Test complex path with mixed matches, insertions, and deletions."""
    # Matrix is transposed before processing
    # Create a path with both deletions and matches:
    # After transpose, we want: (0,0), (1,0), (2,0), (3,1), (4,2)
    # This means deletions from (0,0) to (2,0), then diagonal matches
    matrix = np.array(
        [
            [1, 1, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
        ],
        dtype=int,
    )

    states, b_start, a_end = aln2hmm.alignment_matrix_to_state_vector(matrix)

    # Should have matches
    match_states = [s for s in states if s[0][1] == "m"]
    assert len(match_states) > 0

    # Should have deletions
    deletion_states = [s for s in states if s[0][1] == "d"]
    assert len(deletion_states) > 0


def test_alignment_matrix_to_state_vector_empty_path():
    """Test that empty alignment matrix with no path raises ValueError."""
    matrix = np.zeros((3, 3), dtype=int)

    with pytest.raises(ValueError, match="Alignment matrix contains no path"):
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
    # Matrix is transposed, multiple columns in same row req'd
    # After transpose: row 0 should have multiple positions
    matrix = np.array([[1, 0], [1, 0], [1, 0], [0, 1]], dtype=int)

    states, b_start, a_end = aln2hmm.alignment_matrix_to_state_vector(matrix)

    # Should handle insertions
    assert len(states) > 0
    # Check for insertion states
    insertion_states = [s for s in states if s[0][1] == "i"]
    assert len(insertion_states) > 0


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
