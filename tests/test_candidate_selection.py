"""Candidate selection must consider the corrected alignment's validity."""

import numpy as np
import pytest

from sabr import alignment


def _mock_candidates(monkeypatch, candidates):
    references = {}
    traced = []
    for index, (name, (_matrix, _, positions)) in enumerate(candidates.items()):
        references[name] = (np.full((len(positions), 64), index), positions)

    def align_reference(query, reference, mode, **kwargs):
        name = tuple(candidates)[int(reference[0, 0])]
        matrix, score, _ = candidates[name]
        if kwargs.get("score_only"):
            matrix = None
        else:
            traced.append(name)
        return matrix, np.zeros((len(query), len(reference) + 2)), score

    monkeypatch.setattr(
        alignment, "load_references", lambda *a, **k: references
    )
    monkeypatch.setattr(alignment, "_align_reference", align_reference)
    monkeypatch.setattr(alignment, "apply_corrections", lambda a, **k: a)
    return traced


@pytest.mark.parametrize("chain_type", ["H,K", "auto"])
def test_invalid_winner_falls_back_in_score_order(monkeypatch, chain_type):
    traced = _mock_candidates(
        monkeypatch,
        {
            "H": (np.ones((2, 2)), 10, (1, 2)),
            "K": (np.eye(2), 9, (1, 2)),
            "L": (np.eye(2), 9, (1, 2)),
        },
    )
    matrix, name, score = alignment.align(
        np.zeros((2, 64)), frozenset(), chain_type, 0
    )
    assert name == "K"
    assert score == 9
    np.testing.assert_array_equal(matrix[:, :2], np.eye(2))
    assert traced == ["H", "K"]


@pytest.mark.parametrize("invalid", ["empty", "missing_domain", "overlap"])
def test_invalid_composite_falls_back_to_valid_composite(monkeypatch, invalid):
    matrix = np.zeros((4, 4))
    if invalid == "missing_domain":
        matrix[:2, :2] = np.eye(2)
    elif invalid == "overlap":
        matrix[:] = np.eye(4)
        matrix[0, 2] = 1
    _mock_candidates(
        monkeypatch,
        {
            "HK": (matrix, 1000, (1, 2, 129, 130)),
            "HL": (np.eye(4), 10, (1, 2, 129, 130)),
        },
    )
    result, name, score = alignment.align(
        np.zeros((4, 64)), frozenset(), "HK,HL", 0
    )
    assert name == "HL"
    assert score == 10
    np.testing.assert_array_equal(result[:, [0, 1, 128, 129]], np.eye(4))


def test_all_invalid_reports_each_requested_candidate(monkeypatch):
    _mock_candidates(
        monkeypatch,
        {
            "H": (np.zeros((2, 2)), 10, (1, 2)),
            "K": (np.ones((2, 2)), 9, (1, 2)),
            "L": (np.eye(2), 8, (1, 2)),
        },
    )
    with pytest.raises(ValueError) as error:
        alignment.align(np.zeros((2, 64)), frozenset(), "H,K", 0)
    assert "No valid alignment" in str(error.value)
    assert "H:" in str(error.value)
    assert "K:" in str(error.value)
    assert "no assigned residues" in str(error.value)
    assert "L:" not in str(error.value)


def test_selection_validates_after_corrections(monkeypatch):
    _mock_candidates(
        monkeypatch,
        {
            "H": (np.eye(2), 10, (1, 2)),
            "K": (np.eye(2), 9, (3, 4)),
        },
    )
    gaps = frozenset({0})

    def correct(matrix, *, gap_indices):
        assert gap_indices == gaps
        if matrix[0, 0]:
            matrix[0, 1] = 1
        return matrix

    monkeypatch.setattr(alignment, "apply_corrections", correct)
    _, name, score = alignment.align(np.zeros((2, 64)), gaps, "H,K", 0)
    assert (name, score) == ("K", 9)
