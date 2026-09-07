import warnings

import numpy as np
import pytest

from sabr.corrections import (
    apply_corrections,
    cdr_columns,
    correct_cdr_loop,
    correct_de_loop,
    de_loop_positions,
)
from sabr.numbering import alignment_to_states, number_alignment


@pytest.mark.parametrize(
    ("residues", "positions"),
    [
        (5, [0, 1, 2, 11, 12]),
        (6, [0, 1, 2, 10, 11, 12]),
    ],
)
def test_cdr_gaps_follow_the_imgt_alternating_pattern(residues, positions):
    assert cdr_columns(residues, 13) == positions


def test_cdr_overflow_leaves_central_residues_as_insertions():
    assert cdr_columns(14, 13) == [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        None,
        7,
        8,
        9,
        10,
        11,
        12,
    ]


@pytest.mark.parametrize(
    ("name", "start", "end", "anchors"),
    [
        ("CDR1", 27, 38, (23, 40)),
        ("CDR2", 56, 65, (54, 67)),
        ("CDR3", 105, 117, (104, 118)),
    ],
)
def test_structural_gap_skips_each_affected_cdr(name, start, end, anchors):
    alignment = np.zeros((20, 128), dtype=int)
    alignment[2, anchors[0] - 1] = 1
    alignment[12, anchors[1] - 1] = 1
    original = alignment.copy()
    with pytest.warns(UserWarning, match=f"Skipping {name}.*structural gap"):
        corrected = correct_cdr_loop(
            alignment,
            name,
            start,
            end,
            gap_indices=frozenset({7}),
        )
    np.testing.assert_array_equal(corrected, original)


def test_multiple_structural_gaps_warn_for_only_the_affected_regions():
    alignment = np.eye(128, dtype=int)
    with pytest.warns(UserWarning) as captured:
        corrected = apply_corrections(
            alignment.copy(), gap_indices=frozenset({30, 58})
        )
    assert [str(item.message) for item in captured] == [
        "Skipping CDR1 deterministic correction: structural gap detected "
        "between rows 22 and 39; using the raw embedding alignment for "
        "this region.",
        "Skipping CDR2 deterministic correction: structural gap detected "
        "between rows 53 and 66; using the raw embedding alignment for "
        "this region.",
    ]
    np.testing.assert_array_equal(corrected, alignment)


def test_gap_outside_corrected_regions_emits_no_warning():
    alignment = np.eye(128, dtype=int)
    with warnings.catch_warnings(record=True) as captured:
        apply_corrections(alignment.copy(), gap_indices=frozenset({10}))
    assert captured == []


@pytest.mark.parametrize(
    ("residue_count", "expected"),
    [
        (0, []),
        (1, [80]),
        (2, [80, 84]),
        (3, [80, 83, 84]),
        (4, [80, 82, 83, 84]),
        (5, [80, 81, 82, 83, 84]),
        (6, [80, 81, 82, 82, 83, 84]),
    ],
)
def test_de_loop_positions_follow_imgt_79_85_convention(
    residue_count, expected
):
    assert de_loop_positions(residue_count) == expected


@pytest.mark.parametrize(
    ("expected_columns", "expected_82_states"),
    [
        ([], [(82, "d")]),
        ([79], [(82, "d")]),
        ([79, 83], [(82, "d")]),
        ([79, 82, 83], [(82, "d")]),
        ([79, 81, 82, 83], [(82, "m")]),
        ([79, 80, 81, 82, 83], [(82, "m")]),
        ([79, 80, 81, None, 82, 83], [(82, "m"), (82, "i")]),
    ],
)
def test_de_loop_is_rebuilt_between_79_and_85(
    expected_columns, expected_82_states
):
    residue_count = len(expected_columns)
    alignment = np.zeros((residue_count + 2, 128), dtype=int)
    alignment[0, 78] = 1
    alignment[-1, 84] = 1
    for row in range(1, residue_count + 1):
        alignment[row, 78] = 1

    corrected = correct_de_loop(alignment)
    expected = np.zeros_like(alignment)
    expected[0, 78] = 1
    expected[-1, 84] = 1
    for row, column in enumerate(expected_columns, start=1):
        if column is not None:
            expected[row, column] = 1
    np.testing.assert_array_equal(corrected, expected)

    states, _, _ = alignment_to_states(corrected)
    assert [state for state, _ in states if state[0] == 82] == (
        expected_82_states
    )


def test_six_residue_de_loop_numbers_the_insertion_as_82a():
    alignment = np.zeros((8, 128), dtype=int)
    alignment[0, 78] = 1
    alignment[-1, 84] = 1
    alignment[1:7, 78] = 1

    records = number_alignment(correct_de_loop(alignment), "A" * 8, "imgt", "H")

    assert [(number, code.strip()) for _, number, code, _ in records] == [
        (79, ""),
        (80, ""),
        (81, ""),
        (82, ""),
        (82, "A"),
        (83, ""),
        (84, ""),
        (85, ""),
    ]


def test_structural_gap_skips_de_loop_correction():
    alignment = np.zeros((7, 128), dtype=int)
    for row, position in enumerate(range(79, 86)):
        alignment[row, position - 1] = 1
    original = alignment.copy()

    with pytest.warns(UserWarning, match=r"Skipping DE loop \(79-85\)"):
        corrected = correct_de_loop(alignment, gap_indices=frozenset({1}))
    np.testing.assert_array_equal(corrected, original)


def test_standard_de_loop_keeps_the_learned_alignment():
    alignment = np.zeros((130, 128), dtype=int)
    for row in range(125):
        alignment[row, row] = 1
    original = alignment.copy()
    corrected = apply_corrections(alignment)
    np.testing.assert_array_equal(corrected, original)


def test_gap_in_cdr1_does_not_suppress_cdr2_repair():
    alignment = np.eye(128, dtype=int)
    alignment[30] = 0
    alignment[58] = 0
    with pytest.warns(UserWarning, match="Skipping CDR1") as captured:
        corrected = apply_corrections(alignment, frozenset({30}))
    assert len(captured) == 1
    assert not corrected[30].any()
    assert corrected[58, 58] == 1


@pytest.mark.parametrize("anchor_column", [22, 39])
def test_missing_cdr_anchor_preserves_alignment(anchor_column, caplog):
    alignment = np.eye(128, dtype=int)
    alignment[:, anchor_column - 2 : anchor_column + 3] = 0
    original = alignment.copy()
    corrected = correct_cdr_loop(alignment, "CDR1", 27, 38)
    np.testing.assert_array_equal(corrected, original)
    assert "missing anchor" in caplog.text
