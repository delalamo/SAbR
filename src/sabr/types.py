"""Internal result objects and existing tuple-compatible boundary contracts."""

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating[Any]]
IntArray = NDArray[np.integer[Any]]


class CorrectionStatus(StrEnum):
    """How one deterministic region correction ended.

    Every status except APPLIED keeps the learned alignment for the region.
    STRUCTURAL_GAP is provisional in ``renumber_structure``: unresolved-residue
    metadata can still complete the region, so it is reported only when
    ``sabr.unresolved.metadata_regions`` does not cover it.
    """

    APPLIED = "applied"
    MISSING_ANCHOR = "missing_anchor"
    ANCHORS_OUT_OF_ORDER = "anchors_out_of_order"
    TOO_FEW_RESIDUES = "too_few_residues"
    STRUCTURAL_GAP = "structural_gap"


@dataclass(frozen=True, slots=True)
class RegionCorrection:
    """The outcome of correcting one region (CDR1-3 or DE) of one domain.

    ``message`` is the report for this outcome, or None when there is nothing
    to report. ``sabr.corrections.report_correction`` chooses its channel.
    """

    region: str
    status: CorrectionStatus
    message: str | None


class CandidateState(StrEnum):
    """Lifecycle of one reference candidate during alignment selection.

    Score-only ranking creates SCORED candidates; all others start TRACED.
    ACCEPTED and REJECTED are terminal. CANDIDATE_TRANSITIONS lists every
    legal move, and AlignmentCandidate's transition methods enforce it.
    """

    SCORED = "scored"
    TRACED = "traced"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


CANDIDATE_TRANSITIONS = {
    CandidateState.SCORED: frozenset({CandidateState.TRACED}),
    CandidateState.TRACED: frozenset(
        {CandidateState.ACCEPTED, CandidateState.REJECTED}
    ),
    CandidateState.ACCEPTED: frozenset(),
    CandidateState.REJECTED: frozenset(),
}


@dataclass(frozen=True, slots=True, eq=False)
class AlignmentCandidate:
    """One reference evaluation, with separate ranking and raw model scores.

    Raw assignments use reduced reference columns; corrected assignments use
    full IMGT blocks. ``state`` is derived from which results are present:
    a SCORED candidate has no raw alignment yet, an ACCEPTED one has a
    corrected alignment plus one tuple of region outcomes per domain, and a
    REJECTED one has a rejection reason. Construction rejects combinations
    that match no state.
    """

    reference_type: str
    reference_positions: tuple[int, ...]
    raw_score: float
    selection_score: float
    raw_alignment: FloatArray | None
    corrected_alignment: IntArray | None = None
    rejection_reason: str | None = None
    corrections: tuple[tuple[RegionCorrection, ...], ...] = ()

    def __post_init__(self) -> None:
        if self.corrected_alignment is not None and (
            self.raw_alignment is None or self.rejection_reason is not None
        ):
            raise ValueError(
                "A corrected alignment requires a raw alignment and no "
                "rejection reason."
            )
        if self.corrections and self.corrected_alignment is None:
            raise ValueError("Only accepted candidates carry corrections.")

    @property
    def state(self) -> CandidateState:
        if self.rejection_reason is not None:
            return CandidateState.REJECTED
        if self.corrected_alignment is not None:
            return CandidateState.ACCEPTED
        if self.raw_alignment is None:
            return CandidateState.SCORED
        return CandidateState.TRACED

    def traced(self, raw_alignment: FloatArray) -> "AlignmentCandidate":
        """Attach the soft alignment of a score-only candidate."""
        return self._advance(CandidateState.TRACED, raw_alignment=raw_alignment)

    def accepted(
        self,
        corrected_alignment: IntArray,
        corrections: tuple[tuple[RegionCorrection, ...], ...],
    ) -> "AlignmentCandidate":
        """Record a validated alignment and its per-domain region outcomes."""
        return self._advance(
            CandidateState.ACCEPTED,
            corrected_alignment=corrected_alignment,
            corrections=corrections,
        )

    def rejected(self, reason: str) -> "AlignmentCandidate":
        """Record why this candidate cannot be used."""
        return self._advance(CandidateState.REJECTED, rejection_reason=reason)

    def _advance(
        self, target: CandidateState, **changes: Any
    ) -> "AlignmentCandidate":
        if target not in CANDIDATE_TRANSITIONS[self.state]:
            raise ValueError(
                f"{self.reference_type} candidate cannot move from "
                f"{self.state} to {target}."
            )
        candidate = replace(self, **changes)
        if candidate.state is not target:
            raise ValueError(
                f"{self.reference_type} candidate did not reach {target}."
            )
        return candidate


@dataclass(frozen=True, slots=True, eq=False)
class DomainAssignment:
    """A domain view with separate input columns, query rows and output offset.

    Spans are zero-based and half-open; query_span is None for an empty block.
    alignment retains all query rows and shares storage with its source.
    number_offset is in the output scheme, independent of reference width.
    """

    chain_type: str
    reference_span: tuple[int, int]
    query_span: tuple[int, int] | None
    number_offset: int
    alignment: IntArray


# (binary [query residues, 128 * domains], selected reference, raw soft score).
AlignmentResult = tuple[IntArray, str, float]

# (soft [query residues, reference residues] or None for score-only calls,
#  anchored similarities [query residues, reference residues + 2], raw score).
ReferenceAlignmentResult = tuple[FloatArray | None, FloatArray, float]

# (read-only embeddings [reference residues, 64], absolute IMGT positions).
Reference = tuple[FloatArray, tuple[int, ...]]

# ANARCI's index addresses the padded sequence; deletions consume no residue.
NumberingState = (
    tuple[tuple[int, Literal["m", "i"]], int]
    | tuple[tuple[int, Literal["d"]], None]
)
NumberingStates = tuple[list[NumberingState], int, int]

# (zero-based query row, assigned residue number, insertion code, amino acid).
# Insertion codes may be blank/space and are not restricted to one character.
NumberingRecord = tuple[int, int, str, str]
