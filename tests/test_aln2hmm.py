import numpy as np
import pytest

from sabr.alignment import aln2hmm


def test_alignment_matrix_to_state_vector_basic():
    """Test basic diagonal alignment starting at column 0."""
    matrix = np.array([[1, 0], [0, 1]], dtype=int)

    output = aln2hmm.alignment_matrix_to_state_vector(matrix)

    # New algorithm produces one state per matched column
    assert len(output.states) == 2
    # First state: column 0 -> IMGT position 1, seq 0
    assert output.states[0].residue_number == 1
    assert output.states[0].insertion_code == "m"
    assert output.states[0].mapped_residue == 0  # offset = 0
    # Second state: column 1 -> IMGT position 2, seq 1
    assert output.states[1].residue_number == 2
    assert output.states[1].insertion_code == "m"
    assert output.states[1].mapped_residue == 1
    assert output.imgt_start == 0
    assert output.imgt_end == 2  # end = seq_end + 1 + offset


def test_alignment_matrix_to_state_vector_requires_2d():
    with pytest.raises(ValueError):
        aln2hmm.alignment_matrix_to_state_vector(np.ones(3))


def test_alignment_matrix_to_state_vector_diagonal_path():
    """Test simple diagonal alignment (all matches)."""
    matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=int)

    output = aln2hmm.alignment_matrix_to_state_vector(matrix)

    # Diagonal should produce only matches, one per column
    assert all(s.insertion_code == "m" for s in output.states)
    assert len(output.states) == 3  # 3 positions = 3 states
    assert output.imgt_start == 0
    assert output.imgt_end == 3


def test_alignment_matrix_to_state_vector_with_insertions():
    """Test alignment with insertions (multiple rows in same column)."""
    # After transpose: column 0 has rows 0,1,2 (match, inserts)
    # Column 1 has row 3 (match)
    matrix = np.array([[1, 0], [1, 0], [1, 0], [0, 1]], dtype=int)

    output = aln2hmm.alignment_matrix_to_state_vector(matrix)

    # Should have insertions
    insertion_states = [s for s in output.states if s.insertion_code == "i"]
    assert len(insertion_states) == 2  # rows 1,2 are insertions at col 0
    assert output.imgt_start == 0


def test_alignment_matrix_to_state_vector_with_deletions():
    """Test alignment with deletions (gap in column sequence)."""
    # After transpose: row 0 at col 0, row 1 at col 3
    # Columns 1 and 2 have no matches = deletions
    matrix = np.array([[1, 0, 0, 0], [0, 0, 0, 1]], dtype=int)

    output = aln2hmm.alignment_matrix_to_state_vector(matrix)

    # Should have deletions for columns 1 and 2
    deletion_states = [s for s in output.states if s.insertion_code == "d"]
    assert len(deletion_states) == 2
    # Deletion states should have None for mapped_residue
    for s in deletion_states:
        assert s.mapped_residue is None


def test_alignment_matrix_to_state_vector_complex_path():
    """Test complex path with mixed matches, insertions, and deletions."""
    # After transpose:
    # Column 0: row 0 (match)
    # Column 1: no rows (delete)
    # Column 2: no rows (delete)
    # Column 3: rows 1,2,3 (match + 2 insertions)
    matrix = np.array(
        [
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
        ],
        dtype=int,
    )

    output = aln2hmm.alignment_matrix_to_state_vector(matrix)

    # Should have matches
    match_states = [s for s in output.states if s.insertion_code == "m"]
    assert len(match_states) == 2  # column 0 and column 3

    # Should have deletions
    deletion_states = [s for s in output.states if s.insertion_code == "d"]
    assert len(deletion_states) == 2  # columns 1 and 2

    # Should have insertions
    insertion_states = [s for s in output.states if s.insertion_code == "i"]
    assert len(insertion_states) == 2  # rows 2,3 at column 3


def test_alignment_matrix_to_state_vector_empty_path():
    """Test that empty alignment matrix with no path raises ValueError."""
    matrix = np.zeros((3, 3), dtype=int)

    with pytest.raises(ValueError, match="Alignment matrix contains no path"):
        aln2hmm.alignment_matrix_to_state_vector(matrix)


def test_alignment_matrix_to_state_vector_single_match():
    """Test minimal alignment with single match."""
    matrix = np.array([[1]], dtype=int)

    output = aln2hmm.alignment_matrix_to_state_vector(matrix)

    # Single match should produce one state
    assert len(output.states) == 1
    assert output.states[0].residue_number == 1
    assert output.states[0].insertion_code == "m"
    assert output.states[0].mapped_residue == 0
    assert output.imgt_start == 0
    assert output.imgt_end == 1


def test_alignment_matrix_to_state_vector_insertion_at_end():
    """Test insertion at the end of sequence."""
    # After transpose: col 0 has rows 0,1,2 (match + inserts), col 1 has row 3
    matrix = np.array([[1, 0], [1, 0], [1, 0], [0, 1]], dtype=int)

    output = aln2hmm.alignment_matrix_to_state_vector(matrix)

    # Should handle insertions
    assert (
        len(output.states) == 4
    )  # 1 match + 2 inserts at col 0, 1 match at col 1
    # Check for insertion states
    insertion_states = [s for s in output.states if s.insertion_code == "i"]
    assert len(insertion_states) == 2


def test_alignment_matrix_to_state_vector_large_matrix():
    """Test with larger alignment matrix."""
    size = 10
    # Create diagonal alignment
    matrix = np.eye(size, dtype=int)

    output = aln2hmm.alignment_matrix_to_state_vector(matrix)

    # Should produce matches, one per position
    assert all(s.insertion_code == "m" for s in output.states)
    assert len(output.states) == size
    assert output.imgt_start == 0
    assert output.imgt_end == size


def test_alignment_matrix_to_state_vector_offset_start():
    """Test alignment starting at non-zero column (like IMGT position 2)."""
    # Create alignment where first match is at column 1 (IMGT position 2)
    matrix = np.array(
        [
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=int,
    )

    output = aln2hmm.alignment_matrix_to_state_vector(matrix)

    # Should start at column 1
    assert output.imgt_start == 1

    # First state should be at IMGT position 2
    assert output.states[0].residue_number == 2
    assert output.states[0].insertion_code == "m"
    # mapped_residue should be offset by start (1)
    assert output.states[0].mapped_residue == 1  # seq 0 + offset 1

    # All three matches
    assert len(output.states) == 3
    assert all(s.insertion_code == "m" for s in output.states)

    # Verify positions are 2, 3, 4
    assert [s.residue_number for s in output.states] == [2, 3, 4]
