import importlib

import numpy as np

from sabr import constants, mpnn_embeddings, softaligner


def make_aligner() -> softaligner.SoftAligner:
    return softaligner.SoftAligner.__new__(softaligner.SoftAligner)


def test_files_resolves():
    # will raise if not importable
    pkg = importlib.import_module("sabr.assets")
    assert importlib.resources.files(pkg) is not None


def test_normalize_orders_indices():
    embed = np.vstack(
        [np.full((1, constants.EMBED_DIM), i, dtype=float) for i in range(3)]
    )
    mp = mpnn_embeddings.MPNNEmbeddings(
        name="demo",
        embeddings=embed,
        stdev=embed,
        idxs=["3", "1", "2"],
    )
    aligner = make_aligner()

    normalized = aligner.normalize(mp)

    assert normalized.idxs == [1, 2, 3]
    expected = np.vstack([embed[1], embed[2], embed[0]])
    assert np.array_equal(normalized.embeddings, expected)


def test_correct_gap_numbering_places_expected_ones():
    aligner = make_aligner()
    sub_aln = np.zeros((3, 3), dtype=int)

    corrected = aligner.correct_gap_numbering(sub_aln)

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
    aligner = make_aligner()
    sub_aln = np.zeros((3, 5), dtype=int)

    corrected = aligner.correct_gap_numbering(sub_aln)

    # Should place min(3,5) = 3 ones
    assert corrected.sum() == 3


def test_correct_gap_numbering_single_element():
    """Test with 1x1 matrix."""
    aligner = make_aligner()
    sub_aln = np.zeros((1, 1), dtype=int)

    corrected = aligner.correct_gap_numbering(sub_aln)

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


def test_filter_embeddings_by_chain_type_none():
    """Test that None chain_type returns all embeddings."""
    aligner = make_aligner()
    embed = np.ones((5, constants.EMBED_DIM), dtype=float)
    aligner.all_embeddings = [
        mpnn_embeddings.MPNNEmbeddings(
            name="humanH", embeddings=embed, idxs=["1", "2", "3", "4", "5"]
        ),
        mpnn_embeddings.MPNNEmbeddings(
            name="humanK", embeddings=embed, idxs=["1", "2", "3", "4", "5"]
        ),
        mpnn_embeddings.MPNNEmbeddings(
            name="mouseL", embeddings=embed, idxs=["1", "2", "3", "4", "5"]
        ),
    ]

    filtered = aligner.filter_embeddings_by_chain_type(None)

    assert len(filtered) == 3


def test_filter_embeddings_by_chain_type_heavy():
    """Test that 'heavy' chain_type returns only H embeddings."""
    aligner = make_aligner()
    embed = np.ones((5, constants.EMBED_DIM), dtype=float)
    aligner.all_embeddings = [
        mpnn_embeddings.MPNNEmbeddings(
            name="humanH", embeddings=embed, idxs=["1", "2", "3", "4", "5"]
        ),
        mpnn_embeddings.MPNNEmbeddings(
            name="humanK", embeddings=embed, idxs=["1", "2", "3", "4", "5"]
        ),
        mpnn_embeddings.MPNNEmbeddings(
            name="mouseL", embeddings=embed, idxs=["1", "2", "3", "4", "5"]
        ),
    ]

    filtered = aligner.filter_embeddings_by_chain_type(
        constants.ChainType.HEAVY
    )

    assert len(filtered) == 1
    assert filtered[0].name == "humanH"


def test_filter_embeddings_by_chain_type_light():
    """Test that 'light' chain_type returns only K and L embeddings."""
    aligner = make_aligner()
    embed = np.ones((5, constants.EMBED_DIM), dtype=float)
    aligner.all_embeddings = [
        mpnn_embeddings.MPNNEmbeddings(
            name="humanH", embeddings=embed, idxs=["1", "2", "3", "4", "5"]
        ),
        mpnn_embeddings.MPNNEmbeddings(
            name="humanK", embeddings=embed, idxs=["1", "2", "3", "4", "5"]
        ),
        mpnn_embeddings.MPNNEmbeddings(
            name="mouseL", embeddings=embed, idxs=["1", "2", "3", "4", "5"]
        ),
    ]

    filtered = aligner.filter_embeddings_by_chain_type(
        constants.ChainType.LIGHT
    )

    assert len(filtered) == 2
    assert all(emb.name[-1] in ("K", "L") for emb in filtered)


def test_correct_fr1_alignment_no_correction_needed():
    """Test FR1 correction when no correction is needed."""
    aligner = make_aligner()
    aln = np.zeros((15, 128), dtype=int)
    # Set up normal alignment where row matches column
    aln[9, 9] = 1  # Position 10 (0-indexed: 9) is filled

    # With input_has_pos10=True (kappa), position 10 should remain
    corrected = aligner.correct_fr1_alignment(
        aln, chain_type=None, input_has_pos10=True
    )

    # Should not change
    assert np.array_equal(corrected, aln)


def test_correct_fr1_alignment_with_shift():
    """Test FR1 correction when shift is needed."""
    aligner = make_aligner()
    aln = np.zeros((15, 128), dtype=int)
    # Position 10 (0-indexed: 9) is empty
    aln[:, 9] = 0
    # Row 7 at column 6 means residue 8 is at position 7 (shifted)
    aln[7, 6] = 1

    corrected = aligner.correct_fr1_alignment(
        aln, chain_type=None, input_has_pos10=True
    )

    # Should have shifted the alignment
    assert corrected[7, 6] == 0  # Original position should be cleared


def test_correct_fr1_heavy_chain_move_to_pos9():
    """Test heavy chain FR1 correction: move residue from pos10 to pos9."""
    aligner = make_aligner()
    aln = np.zeros((15, 128), dtype=int)
    # Position 9 (col 8) is empty, position 10 (col 9) has a residue
    aln[8, 9] = 1  # Residue 9 incorrectly at position 10

    # Heavy chains don't have position 10 (input_has_pos10=False)
    corrected = aligner.correct_fr1_alignment(
        aln, chain_type=constants.ChainType.HEAVY, input_has_pos10=False
    )

    # Residue should be moved from pos10 to pos9
    assert corrected[8, 8] == 1  # Now at position 9
    assert corrected[8, 9] == 0  # Position 10 cleared


def test_correct_fr1_heavy_chain_move_to_pos11():
    """Test heavy chain FR1 correction: move residue from pos10 to pos11."""
    aligner = make_aligner()
    aln = np.zeros((15, 128), dtype=int)
    # Position 9 (col 8) is filled, position 10 (col 9) has a residue,
    # position 11 (col 10) is empty
    aln[7, 8] = 1  # Residue 8 at position 9
    aln[8, 9] = 1  # Residue 9 incorrectly at position 10 (should be at 11)

    # Heavy chains don't have position 10 (input_has_pos10=False)
    corrected = aligner.correct_fr1_alignment(
        aln, chain_type=constants.ChainType.HEAVY, input_has_pos10=False
    )

    # Residue should be moved from pos10 to pos11
    assert corrected[7, 8] == 1  # Position 9 unchanged
    assert corrected[8, 9] == 0  # Position 10 cleared
    assert corrected[8, 10] == 1  # Now at position 11


def test_correct_fr1_lambda_chain_clears_pos10():
    """Test lambda chain FR1 correction: position 10 cleared like heavy."""
    aligner = make_aligner()
    aln = np.zeros((15, 128), dtype=int)
    # Position 9 (col 8) is empty, position 10 (col 9) has a residue
    aln[8, 9] = 1  # Residue incorrectly at position 10

    # Lambda chains don't have position 10 (input_has_pos10=False)
    corrected = aligner.correct_fr1_alignment(
        aln, chain_type=constants.ChainType.LIGHT, input_has_pos10=False
    )

    # Residue should be moved from pos10 to pos9
    assert corrected[8, 8] == 1  # Now at position 9
    assert corrected[8, 9] == 0  # Position 10 cleared


def test_correct_fr3_alignment_no_correction_needed():
    """Test FR3 correction when input has positions 81 and 82."""
    aligner = make_aligner()
    aln = np.zeros((100, 128), dtype=int)
    # Set up positions 81 and 82 (cols 80 and 81) occupied
    aln[70, 80] = 1  # Residue at position 81
    aln[71, 81] = 1  # Residue at position 82

    # If input has both positions, no correction needed
    corrected = aligner.correct_fr3_alignment(
        aln, input_has_pos81=True, input_has_pos82=True
    )

    # Should not change
    assert np.array_equal(corrected, aln)


def test_correct_fr3_alignment_move_81_to_83():
    """Test FR3 correction: move residue from pos 81 to pos 83."""
    aligner = make_aligner()
    aln = np.zeros((100, 128), dtype=int)
    # Position 81 (col 80) has a residue, position 83 (col 82) is empty
    aln[70, 80] = 1  # Residue incorrectly at position 81

    # Light chain lacks position 81
    corrected = aligner.correct_fr3_alignment(
        aln, input_has_pos81=False, input_has_pos82=True
    )

    # Residue should be moved from pos81 to pos83
    assert corrected[70, 80] == 0  # Position 81 cleared
    assert corrected[70, 82] == 1  # Now at position 83


def test_correct_fr3_alignment_move_82_to_84():
    """Test FR3 correction: move residue from pos 82 to pos 84."""
    aligner = make_aligner()
    aln = np.zeros((100, 128), dtype=int)
    # Position 82 (col 81) has a residue, position 84 (col 83) is empty
    aln[71, 81] = 1  # Residue incorrectly at position 82

    # Light chain lacks position 82
    corrected = aligner.correct_fr3_alignment(
        aln, input_has_pos81=True, input_has_pos82=False
    )

    # Residue should be moved from pos82 to pos84
    assert corrected[71, 81] == 0  # Position 82 cleared
    assert corrected[71, 83] == 1  # Now at position 84


def test_correct_fr3_alignment_both_moves():
    """Test FR3 correction: move both 81->83 and 82->84."""
    aligner = make_aligner()
    aln = np.zeros((100, 128), dtype=int)
    # Positions 81 and 82 have residues, 83 and 84 are empty
    aln[70, 80] = 1  # Residue incorrectly at position 81
    aln[71, 81] = 1  # Residue incorrectly at position 82

    # Light chain lacks both positions 81 and 82
    corrected = aligner.correct_fr3_alignment(
        aln, input_has_pos81=False, input_has_pos82=False
    )

    # Both residues should be moved
    assert corrected[70, 80] == 0  # Position 81 cleared
    assert corrected[70, 82] == 1  # Now at position 83
    assert corrected[71, 81] == 0  # Position 82 cleared
    assert corrected[71, 83] == 1  # Now at position 84


def test_correct_fr3_alignment_83_84_already_occupied():
    """Test FR3 correction when target positions are already occupied."""
    aligner = make_aligner()
    aln = np.zeros((100, 128), dtype=int)
    # All positions 81-84 have residues
    aln[70, 80] = 1  # Position 81
    aln[71, 81] = 1  # Position 82
    aln[72, 82] = 1  # Position 83
    aln[73, 83] = 1  # Position 84

    # Light chain lacks positions 81 and 82
    corrected = aligner.correct_fr3_alignment(
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
    aligner = make_aligner()

    # 5 residues, 13 possible positions (like CDR-H3: 105-117)
    sub_aln = np.zeros((5, 13), dtype=int)

    corrected = aligner.correct_gap_numbering(sub_aln)

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
    aligner = make_aligner()

    # 6 residues, 13 possible positions
    sub_aln = np.zeros((6, 13), dtype=int)

    corrected = aligner.correct_gap_numbering(sub_aln)

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
    aligner = make_aligner()

    # Create an alignment with non-IMGT gap distribution
    sub_aln = np.zeros((4, 8), dtype=int)
    sub_aln[0, 0] = 1  # Correct anchor
    sub_aln[1, 3] = 1  # Wrong intermediate
    sub_aln[2, 5] = 1  # Wrong intermediate
    sub_aln[3, 7] = 1  # Correct anchor

    corrected = aligner.correct_gap_numbering(sub_aln)

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
    aligner = make_aligner()

    # Test various loop sizes
    for n_residues in range(2, 15):
        for n_positions in range(n_residues, 20):
            sub_aln = np.zeros((n_residues, n_positions), dtype=int)
            corrected = aligner.correct_gap_numbering(sub_aln)

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
