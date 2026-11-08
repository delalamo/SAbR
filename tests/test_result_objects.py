"""Named candidate diagnostics and domain boundaries remain inspectable."""

import numpy as np
import pytest

from sabr import alignment, numbering, types


@pytest.mark.parametrize("valid", [False, True])
def test_candidate_evaluation_keeps_scores_and_diagnostics_separate(
    monkeypatch, valid
):
    raw = np.eye(2) if valid else np.ones((2, 2))
    candidate = types.AlignmentCandidate(
        reference_type="H",
        reference_positions=(1, 2),
        raw_alignment=raw.copy(),
        raw_score=12.5,
        selection_score=10.0,
    )
    monkeypatch.setattr(alignment, "correct_regions", lambda a, **k: ())
    evaluated = alignment._evaluate_candidate(
        candidate, np.zeros((2, 64)), np.zeros((2, 64)), frozenset(), "sabr"
    )
    assert evaluated.reference_type == "H"
    assert evaluated.reference_positions == (1, 2)
    assert evaluated.raw_score == 12.5
    assert evaluated.selection_score == 10.0
    np.testing.assert_array_equal(evaluated.raw_alignment, raw)
    assert candidate.corrected_alignment is None
    assert candidate.rejection_reason is None
    assert candidate.state is types.CandidateState.TRACED
    if valid:
        assert evaluated.state is types.CandidateState.ACCEPTED
        assert evaluated.rejection_reason is None
        np.testing.assert_array_equal(evaluated.corrected_alignment[:, :2], raw)
    else:
        assert evaluated.state is types.CandidateState.REJECTED
        assert "query row more than once" in evaluated.rejection_reason
        assert evaluated.corrected_alignment is None


def _candidate(**fields):
    return types.AlignmentCandidate(
        reference_type="H",
        reference_positions=(1, 2),
        raw_score=1.0,
        selection_score=1.0,
        **{"raw_alignment": None, **fields},
    )


def test_candidate_moves_follow_the_transition_table():
    states = types.CandidateState
    scored = _candidate()
    traced = scored.traced(np.eye(2))
    accepted = traced.accepted(np.eye(2, dtype=int), ((),))
    rejected = traced.rejected("invalid")
    candidates = (scored, traced, accepted, rejected)
    assert [candidate.state for candidate in candidates] == list(states)
    assert set(types.CANDIDATE_TRANSITIONS) == set(states)
    moves = {
        states.TRACED: lambda candidate: candidate.traced(np.eye(2)),
        states.ACCEPTED: lambda candidate: candidate.accepted(
            np.eye(2, dtype=int), ((),)
        ),
        states.REJECTED: lambda candidate: candidate.rejected("invalid"),
    }
    for candidate in candidates:
        for target, move in moves.items():
            if target in types.CANDIDATE_TRANSITIONS[candidate.state]:
                assert move(candidate).state is target
            else:
                with pytest.raises(ValueError, match="cannot move"):
                    move(candidate)
    with pytest.raises(ValueError, match="did not reach"):
        scored.traced(None)


@pytest.mark.parametrize(
    "fields",
    [
        {"corrected_alignment": np.eye(2, dtype=int)},
        {
            "raw_alignment": np.eye(2),
            "corrected_alignment": np.eye(2, dtype=int),
            "rejection_reason": "invalid",
        },
        {"raw_alignment": np.eye(2), "corrections": ((),)},
    ],
)
def test_candidate_rejects_fields_that_match_no_state(fields):
    with pytest.raises(ValueError):
        _candidate(**fields)


def test_domain_metadata_distinguishes_columns_rows_and_output_offsets():
    matrix = np.zeros((12, 384), dtype=int)
    matrix[1:4, :3] = np.eye(3, dtype=int)
    matrix[5:8, 128:131] = np.eye(3, dtype=int)
    matrix[9:11, 256:258] = np.eye(2, dtype=int)
    domains = numbering._split_domains(matrix, "HKL")
    assert [domain.chain_type for domain in domains] == list("HKL")
    assert [domain.reference_span for domain in domains] == [
        (0, 128),
        (128, 256),
        (256, 384),
    ]
    assert [domain.query_span for domain in domains] == [
        (1, 4),
        (5, 8),
        (9, 11),
    ]
    assert [domain.number_offset for domain in domains] == [0, 1000, 2000]
    for domain in domains:
        start, end = domain.reference_span
        np.testing.assert_array_equal(domain.alignment, matrix[:, start:end])


def test_reduced_single_domain_metadata_keeps_original_columns():
    matrix = np.zeros((6, 4), dtype=int)
    matrix[1:5] = np.eye(4, dtype=int)
    (domain,) = numbering._split_domains(matrix, "H")
    assert domain.reference_span == (0, 4)
    assert domain.query_span == (1, 5)
    assert domain.number_offset == 0
    np.testing.assert_array_equal(domain.alignment, matrix)
