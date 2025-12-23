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


def test_correct_de_loop_no_correction_needed():
    """Test DE loop correction when no correction is needed."""
    aligner = make_aligner()
    aln = np.zeros((10, 128), dtype=int)
    # Set up a pattern that doesn't match correction criteria
    aln[:, 80] = 0
    aln[:, 81] = 1
    aln[:, 82] = 1

    corrected = aligner.correct_de_loop(aln)

    # Should not change
    assert np.array_equal(corrected, aln)


def test_correct_de_loop_case1():
    """Test DE loop correction case 1: position 80 filled, 81-82 empty."""
    aligner = make_aligner()
    aln = np.zeros((10, 128), dtype=int)
    aln[5, 80] = 1  # One residue at position 80
    aln[:, 81:83] = 0  # Positions 81-82 empty

    corrected = aligner.correct_de_loop(aln)

    # Should move position 80 to 82
    assert corrected[5, 80] == 0
    assert corrected[5, 82] == 1


def test_correct_de_loop_case2():
    """Test DE loop correction case 2: 80 and 82 filled, 81 empty."""
    aligner = make_aligner()
    aln = np.zeros((10, 128), dtype=int)
    aln[5, 80] = 1  # One residue at position 80
    aln[:, 81] = 0  # Position 81 empty
    aln[7, 82] = 1  # One residue at position 82

    corrected = aligner.correct_de_loop(aln)

    # Should move position 80 to 81
    assert corrected[5, 80] == 0
    assert corrected[5, 81] == 1
    assert corrected[7, 82] == 1


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


def test_correct_light_chain_fr1_no_correction_needed():
    """Test light chain FR1 correction when no correction is needed."""
    aligner = make_aligner()
    aln = np.zeros((15, 128), dtype=int)
    # Set up normal alignment where position 10 is filled
    aln[9, 9] = 1  # Position 10 (0-indexed: 9) is filled

    corrected = aligner.correct_light_chain_fr1(aln)

    # Should not change
    assert np.array_equal(corrected, aln)


def test_correct_light_chain_fr1_with_shift():
    """Test light chain FR1 correction when shift is needed."""
    aligner = make_aligner()
    aln = np.zeros((15, 128), dtype=int)
    # Position 10 (0-indexed: 9) is empty
    aln[:, 9] = 0
    # Row 7 at column 6 means residue 8 is at position 7 (shifted)
    aln[7, 6] = 1

    corrected = aligner.correct_light_chain_fr1(aln)

    # Should have shifted the alignment
    assert corrected[7, 6] == 0  # Original position should be cleared


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
