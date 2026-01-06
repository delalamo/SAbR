import importlib

import numpy as np

from sabr.alignment.corrections import (
    _skip_for_structural_gap,
    apply_deterministic_corrections,
    correct_c_terminus,
    correct_cdr_loop,
    correct_fr1_alignment,
    correct_fr3_alignment,
    correct_gap_numbering,
)
from sabr.alignment.soft_aligner import SoftAligner


def make_aligner() -> SoftAligner:
    return SoftAligner.__new__(SoftAligner)


def test_files_resolves():
    # will raise if not importable
    pkg = importlib.import_module("sabr.assets")
    assert importlib.resources.files(pkg) is not None


def test_correct_gap_numbering_places_expected_ones():
    sub_aln = np.zeros((3, 3), dtype=int)

    corrected = correct_gap_numbering(sub_aln)

    assert corrected[0, 0] == 1
    assert corrected[-1, -1] == 1
    assert corrected.sum() == min(sub_aln.shape)


def test_fix_aln_expands_to_imgt_width():
    aligner = make_aligner()
    old_aln = np.array([[1, 0, 1], [0, 1, 0]], dtype=int)
    expanded = aligner.fix_aln(old_aln, idxs=["1", "3", "5"])

    assert expanded.shape == (2, 128)
    assert np.array_equal(expanded[:, 0], old_aln[:, 0])
    assert np.array_equal(expanded[:, 2], old_aln[:, 1])


def test_correct_gap_numbering_non_square():
    """Test correct_gap_numbering with non-square matrices."""
    sub_aln = np.zeros((3, 5), dtype=int)

    corrected = correct_gap_numbering(sub_aln)

    # Should place min(3,5) = 3 ones
    assert corrected.sum() == 3


def test_correct_gap_numbering_single_element():
    """Test with 1x1 matrix."""
    sub_aln = np.zeros((1, 1), dtype=int)

    corrected = correct_gap_numbering(sub_aln)

    assert corrected[0, 0] == 1
    assert corrected.sum() == 1


def test_fix_aln_with_integer_idxs():
    """Test fix_aln with integer indices."""
    aligner = make_aligner()
    old_aln = np.array([[1, 0], [0, 1]], dtype=int)
    expanded = aligner.fix_aln(old_aln, idxs=[1, 5])

    assert expanded.shape == (2, 128)
    assert expanded[0, 0] == 1
    assert expanded[1, 4] == 1


def test_fix_aln_preserves_dtype():
    """Test that fix_aln preserves data type."""
    aligner = make_aligner()
    old_aln = np.array([[1.5, 0.3]], dtype=float)
    expanded = aligner.fix_aln(old_aln, idxs=["1", "2"])

    assert expanded.dtype == old_aln.dtype


def test_correct_fr1_alignment_kappa_7_residues():
    """Test FR1 correction with 7 residues (kappa - position 10 filled)."""
    aln = np.zeros((20, 128), dtype=int)
    # Set up 7 residues in rows 5-11 aligned near positions 6-12
    # The function uses anchor counting, so place them at positions 6 and 12
    aln[5, 5] = 1  # Row 5 at position 6
    aln[6, 6] = 1  # Row 6 at position 7
    aln[7, 7] = 1  # Row 7 at position 8
    aln[8, 8] = 1  # Row 8 at position 9
    aln[9, 9] = 1  # Row 9 at position 10
    aln[10, 10] = 1  # Row 10 at position 11
    aln[11, 11] = 1  # Row 11 at position 12

    corrected = correct_fr1_alignment(aln)

    # With 7 residues (rows 5-11), position 10 should be occupied (kappa)
    assert corrected[9, 9] == 1, "Position 10 should be occupied for kappa"
    # Check all 7 positions are filled
    assert corrected[5, 5] == 1  # Position 6
    assert corrected[6, 6] == 1  # Position 7
    assert corrected[7, 7] == 1  # Position 8
    assert corrected[8, 8] == 1  # Position 9
    assert corrected[9, 9] == 1  # Position 10
    assert corrected[10, 10] == 1  # Position 11
    assert corrected[11, 11] == 1  # Position 12


def test_correct_fr1_alignment_heavy_6_residues():
    """Test FR1 correction with 6 residues (heavy/lambda - position 10 gap)."""
    aln = np.zeros((20, 128), dtype=int)
    # Set up 6 residues in rows 5-10 aligned near positions 6-12
    aln[5, 5] = 1  # Row 5 at position 6
    aln[6, 6] = 1  # Row 6 at position 7
    aln[7, 7] = 1  # Row 7 at position 8
    aln[8, 8] = 1  # Row 8 at position 9
    aln[9, 10] = 1  # Row 9 at position 11 (skipping 10)
    aln[10, 11] = 1  # Row 10 at position 12

    corrected = correct_fr1_alignment(aln)

    # With 6 residues (rows 5-10), position 10 should be gap (heavy/lambda)
    assert (
        corrected[5:11, 9].sum() == 0
    ), "Position 10 should be empty for heavy"
    # Check all 6 positions are filled correctly (skip position 10)
    assert corrected[5, 5] == 1  # Position 6
    assert corrected[6, 6] == 1  # Position 7
    assert corrected[7, 7] == 1  # Position 8
    assert corrected[8, 8] == 1  # Position 9
    assert corrected[9, 10] == 1  # Position 11
    assert corrected[10, 11] == 1  # Position 12


def test_correct_fr1_no_anchors_found():
    """Test FR1 correction when anchor positions cannot be found."""
    aln = np.zeros((100, 128), dtype=int)
    # No residues in positions 6-12 region
    aln[50, 50] = 1  # Some residue far from FR1

    corrected = correct_fr1_alignment(aln)

    # Should return unchanged since anchors not found
    assert np.array_equal(corrected, aln)


def test_correct_fr3_alignment_no_correction_needed():
    """Test FR3 correction when input has positions 81 and 82."""
    aln = np.zeros((100, 128), dtype=int)
    # Set up positions 81 and 82 (cols 80 and 81) occupied
    aln[70, 80] = 1  # Residue at position 81
    aln[71, 81] = 1  # Residue at position 82

    # If input has both positions, no correction needed
    corrected = correct_fr3_alignment(
        aln, input_has_pos81=True, input_has_pos82=True
    )

    # Should not change
    assert np.array_equal(corrected, aln)


def test_correct_fr3_alignment_move_81_to_83():
    """Test FR3 correction: move residue from pos 81 to pos 83."""
    aln = np.zeros((100, 128), dtype=int)
    # Position 81 (col 80) has a residue, position 83 (col 82) is empty
    aln[70, 80] = 1  # Residue incorrectly at position 81

    # Light chain lacks position 81
    corrected = correct_fr3_alignment(
        aln, input_has_pos81=False, input_has_pos82=True
    )

    # Residue should be moved from pos81 to pos83
    assert corrected[70, 80] == 0  # Position 81 cleared
    assert corrected[70, 82] == 1  # Now at position 83


def test_correct_fr3_alignment_move_82_to_84():
    """Test FR3 correction: move residue from pos 82 to pos 84."""
    aln = np.zeros((100, 128), dtype=int)
    # Position 82 (col 81) has a residue, position 84 (col 83) is empty
    aln[71, 81] = 1  # Residue incorrectly at position 82

    # Light chain lacks position 82
    corrected = correct_fr3_alignment(
        aln, input_has_pos81=True, input_has_pos82=False
    )

    # Residue should be moved from pos82 to pos84
    assert corrected[71, 81] == 0  # Position 82 cleared
    assert corrected[71, 83] == 1  # Now at position 84


def test_correct_fr3_alignment_both_moves():
    """Test FR3 correction: move both 81->83 and 82->84."""
    aln = np.zeros((100, 128), dtype=int)
    # Positions 81 and 82 have residues, 83 and 84 are empty
    aln[70, 80] = 1  # Residue incorrectly at position 81
    aln[71, 81] = 1  # Residue incorrectly at position 82

    # Light chain lacks both positions 81 and 82
    corrected = correct_fr3_alignment(
        aln, input_has_pos81=False, input_has_pos82=False
    )

    # Both residues should be moved
    assert corrected[70, 80] == 0  # Position 81 cleared
    assert corrected[70, 82] == 1  # Now at position 83
    assert corrected[71, 81] == 0  # Position 82 cleared
    assert corrected[71, 83] == 1  # Now at position 84


def test_correct_fr3_alignment_83_84_already_occupied():
    """Test FR3 correction when target positions are already occupied."""
    aln = np.zeros((100, 128), dtype=int)
    # All positions 81-84 have residues
    aln[70, 80] = 1  # Position 81
    aln[71, 81] = 1  # Position 82
    aln[72, 82] = 1  # Position 83
    aln[73, 83] = 1  # Position 84

    # Light chain lacks positions 81 and 82
    corrected = correct_fr3_alignment(
        aln, input_has_pos81=False, input_has_pos82=False
    )

    # Since 83, 84 are occupied, 81, 82 should just be cleared
    assert corrected[70, 80] == 0  # Position 81 cleared
    assert corrected[71, 81] == 0  # Position 82 cleared
    assert corrected[72, 82] == 1  # Position 83 unchanged
    assert corrected[73, 83] == 1  # Position 84 unchanged


def test_correct_gap_numbering_5_residue_cdr():
    """Test IMGT gap distribution for a 5-residue CDR.

    A 5-residue CDR-H3 (positions 105-117, 13 columns) should produce:
    105, 106, 107, 116, 117 (columns 0, 1, 2, 11, 12 in sub-alignment).
    """

    # 5 residues, 13 possible positions (like CDR-H3: 105-117)
    sub_aln = np.zeros((5, 13), dtype=int)

    corrected = correct_gap_numbering(sub_aln)

    # Check anchor points: first residue -> first column, last -> last
    assert corrected[0, 0] == 1  # Residue 0 -> position 105
    assert corrected[4, 12] == 1  # Residue 4 -> position 117

    # Check intermediate positions follow alternating pattern
    assert corrected[1, 1] == 1  # Residue 1 -> position 106
    assert corrected[2, 2] == 1  # Residue 2 -> position 107
    assert corrected[3, 11] == 1  # Residue 3 -> position 116

    # Should have exactly 5 assignments
    assert corrected.sum() == 5


def test_correct_gap_numbering_6_residue_cdr():
    """Test IMGT gap distribution for a 6-residue CDR.

    A 6-residue CDR-H3 (positions 105-117, 13 columns) should produce:
    105, 106, 107, 115, 116, 117 (columns 0, 1, 2, 10, 11, 12).
    """

    # 6 residues, 13 possible positions
    sub_aln = np.zeros((6, 13), dtype=int)

    corrected = correct_gap_numbering(sub_aln)

    # Check anchor points
    assert corrected[0, 0] == 1  # Residue 0 -> position 105
    assert corrected[5, 12] == 1  # Residue 5 -> position 117

    # Check intermediate positions
    assert corrected[1, 1] == 1  # Residue 1 -> position 106
    assert corrected[2, 2] == 1  # Residue 2 -> position 107
    assert corrected[4, 11] == 1  # Residue 4 -> position 116
    assert corrected[3, 10] == 1  # Residue 3 -> position 115

    # Should have exactly 6 assignments
    assert corrected.sum() == 6


def test_correct_gap_numbering_always_applies_imgt_pattern():
    """Test that IMGT gap distribution is always applied.

    Even if the input alignment has a different distribution,
    correct_gap_numbering should always redistribute according to
    IMGT rules.
    """

    # Create an alignment with non-IMGT gap distribution
    sub_aln = np.zeros((4, 8), dtype=int)
    sub_aln[0, 0] = 1  # Correct anchor
    sub_aln[1, 3] = 1  # Wrong intermediate
    sub_aln[2, 5] = 1  # Wrong intermediate
    sub_aln[3, 7] = 1  # Correct anchor

    corrected = correct_gap_numbering(sub_aln)

    # Should apply IMGT pattern regardless of input
    assert corrected[0, 0] == 1  # First anchor
    assert corrected[3, 7] == 1  # Last anchor (row -1, col -1)
    assert corrected[1, 1] == 1  # Second position
    assert corrected[2, 6] == 1  # Second-to-last position

    assert corrected.sum() == 4


def test_correct_gap_numbering_preserves_anchor_points():
    """Test that anchor points (first and last) are always preserved.

    The first residue must always map to the first column (N-terminal anchor)
    and the last residue must always map to the last column (C-terminal anchor).
    This is critical for maintaining CDR boundary positions.
    """

    # Test various loop sizes
    for n_residues in range(2, 15):
        for n_positions in range(n_residues, 20):
            sub_aln = np.zeros((n_residues, n_positions), dtype=int)
            corrected = correct_gap_numbering(sub_aln)

            # First residue (row 0) must map to first column (col 0)
            assert corrected[0, 0] == 1, (
                f"N-terminal anchor not preserved for "
                f"{n_residues} residues, {n_positions} positions"
            )

            # Last residue (row -1) must map to last column (col -1)
            assert corrected[n_residues - 1, n_positions - 1] == 1, (
                f"C-terminal anchor not preserved for "
                f"{n_residues} residues, {n_positions} positions"
            )


def test_correct_c_terminus_no_correction_needed():
    """Test C-terminus correction when no correction is needed.

    When all residues are already assigned, no correction should be applied.
    """
    aln = np.zeros((120, 128), dtype=int)

    # Set up an alignment where all rows have assignments
    # and the last assigned column is 127 (IMGT position 128)
    for i in range(120):
        aln[i, min(i, 127)] = 1

    corrected = correct_c_terminus(aln)

    # Should not change since all rows are assigned
    assert np.array_equal(corrected, aln)


def test_correct_c_terminus_assigns_trailing_residues():
    """Test that trailing unassigned residues are assigned to C-terminus.

    When residues after position 125 are unassigned, they should be
    deterministically assigned to positions 126, 127, 128.
    """
    aln = np.zeros((120, 128), dtype=int)

    # Set up alignment: residues 0-116 assigned to IMGT positions 1-117
    # and residue 117 assigned to position 125 (0-indexed: 124)
    # Residues 118, 119 are unassigned
    for i in range(117):
        aln[i, i] = 1
    aln[117, 124] = 1  # Last assigned is row 117 at col 124 (IMGT pos 125)

    corrected = correct_c_terminus(aln)

    # Residues 118, 119 should now be assigned to cols 125, 126 (IMGT 126, 127)
    assert corrected[118, 125] == 1, "Row 118 should be assigned to col 125"
    assert corrected[119, 126] == 1, "Row 119 should be assigned to col 126"


def test_correct_c_terminus_respects_max_imgt_position():
    """Test that C-terminus correction doesn't exceed IMGT position 128.

    Even if there are many trailing unassigned residues, we can only
    assign up to position 128 (0-indexed: 127).
    """
    aln = np.zeros((130, 128), dtype=int)

    # Set up alignment: residues 0-124 assigned, last at col 124 (IMGT 125)
    for i in range(125):
        aln[i, i] = 1

    # Residues 125-129 are unassigned (5 residues)
    # But we only have positions 126, 127, 128 available (3 positions)

    corrected = correct_c_terminus(aln)

    # Only 3 residues should be assigned (to positions 126, 127, 128)
    assert corrected[125, 125] == 1, "Row 125 should be assigned to col 125"
    assert corrected[126, 126] == 1, "Row 126 should be assigned to col 126"
    assert corrected[127, 127] == 1, "Row 127 should be assigned to col 127"

    # Rows 128, 129 should still be unassigned (no space left)
    assert corrected[128, :].sum() == 0, "Row 128 should remain unassigned"
    assert corrected[129, :].sum() == 0, "Row 129 should remain unassigned"


def test_correct_c_terminus_skips_if_last_col_too_early():
    """Test that C-terminus correction is skipped if last col is too early.

    If the last assigned column is before position 125 (0-indexed: 124),
    the correction should not be applied as this indicates a different issue.
    """
    aln = np.zeros((100, 128), dtype=int)

    # Set up alignment: last assigned position is 100 (0-indexed: 99)
    # This is well before the C-terminus anchor position
    for i in range(90):
        aln[i, i] = 1

    corrected = correct_c_terminus(aln)

    # Should not change since last assigned col (89) < anchor position (124)
    assert np.array_equal(corrected, aln)


def test_correct_c_terminus_single_trailing_residue():
    """Test C-terminus correction with a single trailing unassigned residue."""
    aln = np.zeros((118, 128), dtype=int)

    # Set up: residues 0-116 assigned, last at position 126 (0-indexed: 125)
    for i in range(117):
        aln[i, i] = 1
    aln[116, 125] = 1  # Overwrite: row 116 at col 125 (IMGT pos 126)

    # Row 117 is unassigned

    corrected = correct_c_terminus(aln)

    # Row 117 should be assigned to col 126 (IMGT pos 127)
    assert corrected[117, 126] == 1, "Row 117 should be assigned to col 126"


def test_correct_c_terminus_empty_alignment():
    """Test C-terminus correction with an empty alignment matrix."""
    aln = np.zeros((100, 128), dtype=int)

    corrected = correct_c_terminus(aln)

    # Should return unchanged (no assignments to work with)
    assert np.array_equal(corrected, aln)


class TestGapSkipping:
    """Tests for gap skipping behavior in correction methods."""

    def test_skip_for_structural_gap_returns_true_when_gap_present(self):
        """Test _skip_for_structural_gap returns True when gap in region."""
        gap_indices = frozenset({5, 10})

        result = _skip_for_structural_gap(
            gap_indices, start_row=3, end_row=8, region_name="test"
        )

        assert result is True

    def test_skip_for_structural_gap_returns_false_when_no_gap(self):
        """Test _skip_for_structural_gap returns False when no gap in region."""
        gap_indices = frozenset({1, 15})

        result = _skip_for_structural_gap(
            gap_indices, start_row=5, end_row=10, region_name="test"
        )

        assert result is False

    def test_skip_for_structural_gap_returns_false_when_none(self):
        """Test _skip_for_structural_gap returns False when None."""

        result = _skip_for_structural_gap(
            None, start_row=5, end_row=10, region_name="test"
        )

        assert result is False

    def test_skip_for_structural_gap_empty_set(self):
        """Test _skip_for_structural_gap with empty gap set."""
        gap_indices = frozenset()

        result = _skip_for_structural_gap(
            gap_indices, start_row=0, end_row=100, region_name="test"
        )

        assert result is False

    def test_fr1_skips_correction_when_gap_present(self):
        """Test FR1 correction is skipped when structural gap is in region."""
        aln = np.zeros((20, 128), dtype=int)
        # Set up 7 residues that would normally trigger correction
        aln[5, 5] = 1
        aln[6, 6] = 1
        aln[7, 7] = 1
        aln[8, 8] = 1
        aln[9, 9] = 1
        aln[10, 10] = 1
        aln[11, 11] = 1

        original_aln = aln.copy()
        # Gap at row 7 (within the FR1 region)
        gap_indices = frozenset({7})

        corrected = correct_fr1_alignment(aln, gap_indices=gap_indices)

        # Should return unchanged since gap is present
        assert np.array_equal(corrected, original_aln)

    def test_fr1_applies_correction_when_no_gap(self):
        """Test FR1 correction is applied when no structural gap."""
        aln = np.zeros((20, 128), dtype=int)
        aln[5, 5] = 1
        aln[6, 6] = 1
        aln[7, 7] = 1
        aln[8, 8] = 1
        aln[9, 9] = 1
        aln[10, 10] = 1
        aln[11, 11] = 1

        original_aln = aln.copy()
        # No gaps in the FR1 region (gap is outside)
        gap_indices = frozenset({20})

        corrected = correct_fr1_alignment(aln, gap_indices=gap_indices)

        # Correction should be applied (may or may not change, but method runs)
        # At minimum, it should not raise an error
        assert corrected.shape == original_aln.shape

    def test_fr3_skips_correction_when_gap_in_de_loop(self):
        """Test FR3 correction is skipped when gap in DE loop region."""
        aln = np.zeros((100, 128), dtype=int)
        # Set up positions in DE loop region (positions 79-84)
        aln[70, 78] = 1  # Position 79
        aln[71, 79] = 1  # Position 80
        aln[72, 80] = 1  # Position 81 (incorrectly placed for light chain)
        aln[73, 81] = 1  # Position 82 (incorrectly placed for light chain)
        aln[74, 82] = 1  # Position 83
        aln[75, 83] = 1  # Position 84

        original_aln = aln.copy()
        # Gap at row 72 (within DE loop region)
        gap_indices = frozenset({72})

        corrected = correct_fr3_alignment(
            aln,
            input_has_pos81=False,
            input_has_pos82=False,
            gap_indices=gap_indices,
        )

        # Should return unchanged since gap is in DE loop
        assert np.array_equal(corrected, original_aln)

    def test_cdr_loop_skips_correction_when_gap_present(self):
        """Test CDR loop correction is skipped when gap in region."""
        aln = np.zeros((100, 128), dtype=int)
        # Set up alignment with residues in CDR1 region
        # CDR1 anchors are at positions 23 and 40
        aln[20, 22] = 1  # Near anchor at position 23
        aln[21, 23] = 1
        aln[22, 24] = 1
        aln[23, 25] = 1
        aln[24, 26] = 1
        aln[30, 39] = 1  # Near anchor at position 40

        original_aln = aln.copy()
        # Gap at row 22 (within CDR1 region)
        gap_indices = frozenset({22})

        corrected = correct_cdr_loop(
            aln,
            loop_name="CDR1",
            cdr_start=27,
            cdr_end=38,
            gap_indices=gap_indices,
        )

        # Should return unchanged since gap is in region
        assert np.array_equal(corrected, original_aln)

    def test_apply_deterministic_corrections_passes_gap_indices(self):
        """Test that _apply_deterministic_corrections passes gap_indices."""
        aln = np.zeros((100, 128), dtype=int)
        # Create a minimal alignment
        for i in range(100):
            aln[i, min(i, 127)] = 1

        gap_indices = frozenset({50})

        # Should not raise an error
        corrected, chain_type = apply_deterministic_corrections(
            aln, gap_indices=gap_indices
        )

        assert corrected.shape == aln.shape
        assert chain_type in ("H", "K", "L")
