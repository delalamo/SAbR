import importlib

import numpy as np
import pytest

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


@pytest.mark.parametrize(
    "n_residues,n_positions,expected_sum",
    [
        (3, 3, 3),  # square matrix
        (3, 5, 3),  # non-square
        (1, 1, 1),  # single element
    ],
)
def test_correct_gap_numbering_basic(n_residues, n_positions, expected_sum):
    """Test correct_gap_numbering with various matrix sizes."""
    aligner = make_aligner()
    sub_aln = np.zeros((n_residues, n_positions), dtype=int)

    corrected = aligner.correct_gap_numbering(sub_aln)

    assert corrected[0, 0] == 1  # First anchor
    assert corrected[-1, -1] == 1  # Last anchor
    assert corrected.sum() == expected_sum


@pytest.mark.parametrize(
    "n_residues,n_positions,expected_positions",
    [
        # 5-residue CDR: 105, 106, 107, 116, 117 -> cols 0, 1, 2, 11, 12
        (5, 13, [(0, 0), (1, 1), (2, 2), (3, 11), (4, 12)]),
        # 6-residue CDR: cols 0, 1, 2, 10, 11, 12
        (6, 13, [(0, 0), (1, 1), (2, 2), (3, 10), (4, 11), (5, 12)]),
        # 4-residue with 8 positions
        (4, 8, [(0, 0), (1, 1), (2, 6), (3, 7)]),
    ],
)
def test_correct_gap_numbering_imgt_pattern(
    n_residues, n_positions, expected_positions
):
    """Test IMGT gap distribution for CDRs of various sizes."""
    aligner = make_aligner()
    sub_aln = np.zeros((n_residues, n_positions), dtype=int)

    corrected = aligner.correct_gap_numbering(sub_aln)

    for row, col in expected_positions:
        assert corrected[row, col] == 1, (
            f"Expected position ({row}, {col}) to be 1 for "
            f"{n_residues}x{n_positions} matrix"
        )
    assert corrected.sum() == n_residues


def test_correct_gap_numbering_preserves_anchor_points():
    """Test that anchor points (first and last) are always preserved."""
    aligner = make_aligner()

    for n_residues in range(2, 15):
        for n_positions in range(n_residues, 20):
            sub_aln = np.zeros((n_residues, n_positions), dtype=int)
            corrected = aligner.correct_gap_numbering(sub_aln)

            assert corrected[0, 0] == 1, (
                f"N-terminal anchor not preserved for "
                f"{n_residues} residues, {n_positions} positions"
            )
            assert corrected[n_residues - 1, n_positions - 1] == 1, (
                f"C-terminal anchor not preserved for "
                f"{n_residues} residues, {n_positions} positions"
            )


def test_fix_aln_expands_to_imgt_width():
    aligner = make_aligner()
    old_aln = np.array([[1, 0, 1], [0, 1, 0]], dtype=int)
    expanded = aligner.fix_aln(old_aln, idxs=["1", "3", "5"])

    assert expanded.shape == (2, 128)
    assert np.array_equal(expanded[:, 0], old_aln[:, 0])
    assert np.array_equal(expanded[:, 2], old_aln[:, 1])


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


@pytest.mark.parametrize(
    "chain_type,expected_count,expected_names",
    [
        (None, 3, ["humanH", "humanK", "mouseL"]),
        (constants.ChainType.HEAVY, 1, ["humanH"]),
        (constants.ChainType.LIGHT, 2, ["humanK", "mouseL"]),
    ],
)
def test_filter_embeddings_by_chain_type(
    chain_type, expected_count, expected_names
):
    """Test filter_embeddings_by_chain_type with different chain types."""
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

    filtered = aligner.filter_embeddings_by_chain_type(chain_type)

    assert len(filtered) == expected_count
    assert [emb.name for emb in filtered] == expected_names


@pytest.mark.parametrize(
    "description,setup,chain_type,input_has_pos10,expected_changes",
    [
        # No correction needed
        (
            "no_correction_needed",
            {(9, 9): 1},
            None,
            True,
            {},  # No changes
        ),
        # Heavy chain: move from pos10 to pos9
        (
            "heavy_move_to_pos9",
            {(8, 9): 1},
            constants.ChainType.HEAVY,
            False,
            {(8, 8): 1, (8, 9): 0},
        ),
        # Heavy chain: move from pos10 to pos11 (pos9 occupied)
        (
            "heavy_move_to_pos11",
            {(7, 8): 1, (8, 9): 1},
            constants.ChainType.HEAVY,
            False,
            {(7, 8): 1, (8, 9): 0, (8, 10): 1},
        ),
        # Lambda chain: pos10 cleared like heavy
        (
            "lambda_clears_pos10",
            {(8, 9): 1},
            constants.ChainType.LIGHT,
            False,
            {(8, 8): 1, (8, 9): 0},
        ),
    ],
)
def test_correct_fr1_alignment(
    description, setup, chain_type, input_has_pos10, expected_changes
):
    """Test FR1 alignment correction with various scenarios."""
    aligner = make_aligner()
    aln = np.zeros((15, 128), dtype=int)

    # Apply setup
    for (row, col), val in setup.items():
        aln[row, col] = val

    original = aln.copy()
    corrected = aligner.correct_fr1_alignment(
        aln, chain_type=chain_type, input_has_pos10=input_has_pos10
    )

    if not expected_changes:
        assert np.array_equal(
            corrected, original
        ), f"{description}: expected no change"
    else:
        for (row, col), expected_val in expected_changes.items():
            assert (
                corrected[row, col] == expected_val
            ), f"{description}: expected [{row},{col}] = {expected_val}"


def test_correct_fr1_alignment_with_shift():
    """Test FR1 correction when shift is needed."""
    aligner = make_aligner()
    aln = np.zeros((15, 128), dtype=int)
    aln[:, 9] = 0
    aln[7, 6] = 1

    corrected = aligner.correct_fr1_alignment(
        aln, chain_type=None, input_has_pos10=True
    )

    assert corrected[7, 6] == 0  # Original position should be cleared


@pytest.mark.parametrize(
    "description,setup,input_has_pos81,input_has_pos82,expected_changes",
    [
        # No correction needed
        (
            "no_correction_needed",
            {(70, 80): 1, (71, 81): 1},
            True,
            True,
            {},
        ),
        # Move 81 to 83
        (
            "move_81_to_83",
            {(70, 80): 1},
            False,
            True,
            {(70, 80): 0, (70, 82): 1},
        ),
        # Move 82 to 84
        (
            "move_82_to_84",
            {(71, 81): 1},
            True,
            False,
            {(71, 81): 0, (71, 83): 1},
        ),
        # Both moves
        (
            "both_moves",
            {(70, 80): 1, (71, 81): 1},
            False,
            False,
            {(70, 80): 0, (70, 82): 1, (71, 81): 0, (71, 83): 1},
        ),
        # Target positions already occupied
        (
            "targets_occupied",
            {(70, 80): 1, (71, 81): 1, (72, 82): 1, (73, 83): 1},
            False,
            False,
            {(70, 80): 0, (71, 81): 0, (72, 82): 1, (73, 83): 1},
        ),
    ],
)
def test_correct_fr3_alignment(
    description, setup, input_has_pos81, input_has_pos82, expected_changes
):
    """Test FR3 alignment correction with various scenarios."""
    aligner = make_aligner()
    aln = np.zeros((100, 128), dtype=int)

    # Apply setup
    for (row, col), val in setup.items():
        aln[row, col] = val

    original = aln.copy()
    corrected = aligner.correct_fr3_alignment(
        aln, input_has_pos81=input_has_pos81, input_has_pos82=input_has_pos82
    )

    if not expected_changes:
        assert np.array_equal(
            corrected, original
        ), f"{description}: expected no change"
    else:
        for (row, col), expected_val in expected_changes.items():
            assert (
                corrected[row, col] == expected_val
            ), f"{description}: expected [{row},{col}] = {expected_val}"


@pytest.mark.parametrize(
    "description,n_rows,setup,expected_assignments",
    [
        # No correction needed - all rows assigned
        (
            "no_correction_needed",
            120,
            lambda aln: [
                setattr(aln, "__setitem__", None)
                or aln.__setitem__((i, min(i, 127)), 1)
                for i in range(120)
            ],
            None,  # Check no change
        ),
        # Empty alignment
        (
            "empty_alignment",
            100,
            lambda aln: None,
            None,  # Check no change
        ),
    ],
)
def test_correct_c_terminus_parametrized(
    description, n_rows, setup, expected_assignments
):
    """Test C-terminus correction edge cases."""
    aligner = make_aligner()
    aln = np.zeros((n_rows, 128), dtype=int)

    # Special handling for the "no_correction_needed" case
    if description == "no_correction_needed":
        for i in range(n_rows):
            aln[i, min(i, 127)] = 1

    original = aln.copy()
    corrected = aligner.correct_c_terminus(aln)

    if expected_assignments is None:
        assert np.array_equal(
            corrected, original
        ), f"{description}: expected no change"


def test_correct_c_terminus_assigns_trailing_residues():
    """Test that trailing unassigned residues are assigned to C-terminus."""
    aligner = make_aligner()
    aln = np.zeros((120, 128), dtype=int)

    for i in range(117):
        aln[i, i] = 1
    aln[117, 124] = 1  # Last assigned is row 117 at col 124 (IMGT pos 125)

    corrected = aligner.correct_c_terminus(aln)

    assert corrected[118, 125] == 1, "Row 118 should be assigned to col 125"
    assert corrected[119, 126] == 1, "Row 119 should be assigned to col 126"


def test_correct_c_terminus_respects_max_imgt_position():
    """Test that C-terminus correction doesn't exceed IMGT position 128."""
    aligner = make_aligner()
    aln = np.zeros((130, 128), dtype=int)

    for i in range(125):
        aln[i, i] = 1

    corrected = aligner.correct_c_terminus(aln)

    assert corrected[125, 125] == 1, "Row 125 should be assigned to col 125"
    assert corrected[126, 126] == 1, "Row 126 should be assigned to col 126"
    assert corrected[127, 127] == 1, "Row 127 should be assigned to col 127"
    assert corrected[128, :].sum() == 0, "Row 128 should remain unassigned"
    assert corrected[129, :].sum() == 0, "Row 129 should remain unassigned"


def test_correct_c_terminus_skips_if_last_col_too_early():
    """Test that C-terminus correction is skipped if last col is too early."""
    aligner = make_aligner()
    aln = np.zeros((100, 128), dtype=int)

    for i in range(90):
        aln[i, i] = 1

    original = aln.copy()
    corrected = aligner.correct_c_terminus(aln)

    assert np.array_equal(corrected, original)


def test_correct_c_terminus_single_trailing_residue():
    """Test C-terminus correction with a single trailing unassigned residue."""
    aligner = make_aligner()
    aln = np.zeros((118, 128), dtype=int)

    for i in range(117):
        aln[i, i] = 1
    aln[116, 125] = 1  # Overwrite: row 116 at col 125 (IMGT pos 126)

    corrected = aligner.correct_c_terminus(aln)

    assert corrected[117, 126] == 1, "Row 117 should be assigned to col 126"
