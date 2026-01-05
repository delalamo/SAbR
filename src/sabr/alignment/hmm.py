#!/usr/bin/env python3
"""Alignment matrix to HMM state vector conversion module.

This module converts binary alignment matrices to HMMER-style state vectors
for use with ANARCI numbering schemes. The conversion process:

1. Extracts the alignment path from the binary matrix
2. Identifies matched, inserted, and deleted positions
3. Handles orphan residues (CDR insertions not aligned to any IMGT position)
4. Returns a state vector compatible with ANARCI's numbering functions

Key components:
    State: Dataclass representing an HMM state (match/insert/delete)
    alignment_matrix_to_state_vector: Main conversion function
    report_output: Logging utility for state vectors

The alignment matrix format:
    - Rows: Sequence positions (0-indexed)
    - Columns: IMGT positions (0-indexed, so column 0 = IMGT position 1)
    - Value 1: Indicates alignment between sequence and IMGT position
"""

import logging
from dataclasses import dataclass
from typing import Iterator, List, Optional

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class State:
    """Represents an HMM state with residue number and insertion code.

    This dataclass encapsulates a single state in the HMMER-style state vector.
    It can be unpacked like a tuple for backward compatibility with ANARCI.

    Attributes:
        residue_number: The IMGT position number (1-128).
        insertion_code: State type indicator:
            - "m": Match state (residue aligned to this IMGT position)
            - "i": Insertion state (extra residue after this position)
            - "d": Deletion state (no residue at this IMGT position)
        mapped_residue: Index into the padded sequence for match/insert states,
            or None for deletion states. Used by ANARCI to extract residues.
    """

    residue_number: int
    insertion_code: str
    mapped_residue: Optional[int] = None

    def __iter__(self) -> Iterator:
        """Allow unpacking like a tuple for backward compatibility."""
        yield (self.residue_number, self.insertion_code)
        yield self.mapped_residue

    def __getitem__(self, index: int):
        """Allow indexing like a tuple for backward compatibility."""
        if index == 0:
            return (self.residue_number, self.insertion_code)
        elif index == 1:
            return self.mapped_residue
        else:
            raise IndexError(f"State index out of range: {index}")


@dataclass(frozen=True)
class Aln2HmmOutput:
    """Output from alignment_matrix_to_state_vector conversion.

    Attributes:
        states: List of State objects representing the HMM state vector.
        imgt_start: First IMGT column index (0-indexed), for leading dashes.
        imgt_end: Value such that n_aligned = imgt_end - imgt_start gives the
            number of sequence positions to include after the leading dashes.
        first_aligned_row: First sequence row (0-indexed) that is aligned,
            used for alignment_start in thread_alignment.
    """

    states: List[State]
    imgt_start: int
    imgt_end: int
    first_aligned_row: int


def alignment_matrix_to_state_vector(
    matrix: np.ndarray,
) -> Aln2HmmOutput:
    """Return an HMMER-style state vector from a binary alignment matrix.

    The alignment matrix has shape (n_residues, n_imgt_positions) where:
    - Rows are sequence positions (0-indexed)
    - Columns are IMGT positions (0-indexed, so col 0 = IMGT position 1)
    - matrix[seq_idx, imgt_col] = 1 means sequence position seq_idx
      aligns to IMGT column imgt_col

    Handles orphan residues (e.g., CDR3 insertions) that don't map to any
    IMGT column by treating them as insertions after the previous matched
    position.

    Returns:
        Aln2HmmOutput containing:
            - states: List of State objects representing the HMM state vector
            - imgt_start: First IMGT column index (0-indexed)
            - imgt_end: Value such that n_aligned = imgt_end - imgt_start
            - first_aligned_row: First sequence row (0-indexed) that is aligned
    """
    if matrix.ndim != 2:
        raise ValueError("matrix must be 2D")
    LOGGER.info(f"Converting alignment matrix with shape {matrix.shape}")

    path = sorted(np.argwhere(np.transpose(matrix) == 1).tolist())
    if len(path) == 0:
        raise ValueError(
            "Alignment matrix contains no path (no non-zero elements found)"
        )

    col_to_rows = {}
    for col, row in path:
        if col not in col_to_rows:
            col_to_rows[col] = []
        col_to_rows[col].append(row)

    row_to_col = {}
    for col, row in path:
        row_to_col[row] = col

    first_aligned_row = path[0][1]
    last_aligned_row = path[-1][1]
    orphan_rows = {
        row
        for row in range(first_aligned_row, last_aligned_row + 1)
        if row not in row_to_col
    }

    if orphan_rows:
        LOGGER.info(
            f"Found {len(orphan_rows)} orphan residues (CDR insertions)"
        )

    offset = path[0][0]
    states = []

    for col in range(path[0][0], path[-1][0] + 1):
        imgt_pos = col + 1

        if col in col_to_rows:
            rows = col_to_rows[col]
            states.append(State(imgt_pos, "m", rows[0] + offset))

            for row in rows[1:]:
                states.append(State(imgt_pos, "i", row + offset))

            next_matched_row = None
            for next_col in range(col + 1, path[-1][0] + 1):
                if next_col in col_to_rows:
                    next_matched_row = col_to_rows[next_col][0]
                    break

            if next_matched_row is not None:
                for orphan_row in range(rows[-1] + 1, next_matched_row):
                    if orphan_row in orphan_rows:
                        states.append(State(imgt_pos, "i", orphan_row + offset))
        else:
            states.append(State(imgt_pos, "d", None))

    report_output(states)

    max_row = (
        max(last_aligned_row, max(orphan_rows))
        if orphan_rows
        else last_aligned_row
    )
    imgt_start = path[0][0]
    imgt_end = max_row + 1 + imgt_start

    return Aln2HmmOutput(
        states=states,
        imgt_start=imgt_start,
        imgt_end=imgt_end,
        first_aligned_row=first_aligned_row,
    )


def report_output(states: List[State]) -> None:
    """Log each HMM state at INFO level for debugging purposes.

    Args:
        states: List of State objects to log.
    """
    LOGGER.info(f"Reporting {len(states)} HMM states")
    for idx, state in enumerate(states):
        mapped = state.mapped_residue
        LOGGER.info(
            f"{idx} (({state.residue_number}, '{state.insertion_code}'), "
            f"{mapped})"
        )
