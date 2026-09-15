"""Shared tuple contracts; aliases preserve the existing runtime containers."""

from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating[Any]]
IntArray = NDArray[np.integer[Any]]

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
