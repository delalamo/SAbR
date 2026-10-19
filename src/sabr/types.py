"""Internal result objects and existing tuple-compatible boundary contracts."""

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating[Any]]
IntArray = NDArray[np.integer[Any]]


@dataclass(frozen=True, slots=True, eq=False)
class AlignmentCandidate:
    """One reference evaluation, with separate ranking and raw model scores.

    Raw assignments use reduced reference columns; corrected assignments use
    full IMGT blocks. A score-only candidate has no raw alignment yet. After
    evaluation, either corrected_alignment or rejection_reason is populated.
    """

    reference_type: str
    reference_positions: tuple[int, ...]
    raw_score: float
    selection_score: float
    raw_alignment: FloatArray | None
    corrected_alignment: IntArray | None = None
    rejection_reason: str | None = None


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
