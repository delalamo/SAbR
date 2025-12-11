import importlib

import numpy as np

from sabr import constants, mpnn_embeddings, softaligner


def make_aligner():
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

    filtered = aligner.filter_embeddings_by_chain_type("heavy")

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

    filtered = aligner.filter_embeddings_by_chain_type("light")

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
