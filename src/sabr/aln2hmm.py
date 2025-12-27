#!/usr/bin/env python3

import logging
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class State:
    """Represents an HMM state with residue number and insertion code.

    This dataclass can be used like a tuple for backward compatibility.
    """

    residue_number: int
    insertion_code: str
    mapped_residue: Optional[int] = None

    def to_tuple(self) -> Tuple[Tuple[int, str], Optional[int]]:
        """Convert to ANARCI-compatible tuple format."""
        return ((self.residue_number, self.insertion_code), self.mapped_residue)

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


def alignment_matrix_to_state_vector(
    matrix: np.ndarray,
) -> Tuple[List[State], int, int, int]:
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
        states: List of State objects representing the HMM state vector
        imgt_start: First IMGT column index (0-indexed), used for leading dashes
        imgt_end: Value such that subsequence = "-" * imgt_start + sequence[:imgt_end-imgt_start]
             has sufficient length for all mapped_residue values
        first_aligned_row: First sequence row (0-indexed) that is aligned,
             used for alignment_start in thread_alignment
    """
    if matrix.ndim != 2:
        raise ValueError("matrix must be 2D")
    LOGGER.info(f"Converting alignment matrix with shape {matrix.shape}")

    path = sorted(np.argwhere(np.transpose(matrix) == 1).tolist())
    if len(path) == 0:
        raise ValueError(
            "Alignment matrix contains no path (no non-zero elements found)"
        )

    col_to_rows: dict = {}
    for col, row in path:
        if col not in col_to_rows:
            col_to_rows[col] = []
        col_to_rows[col].append(row)

    row_to_col: dict = {}
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
    states: List[State] = []

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

    max_row = max(last_aligned_row, max(orphan_rows)) if orphan_rows else last_aligned_row
    imgt_start = path[0][0]
    imgt_end = max_row + 1 + imgt_start

    return states, imgt_start, imgt_end, first_aligned_row


def report_output(states: List[State]) -> None:
    """Log each HMM state at INFO level."""
    LOGGER.info(f"Reporting {len(states)} HMM states")
    for idx, state in enumerate(states):
        mapped = state.mapped_residue
        LOGGER.info(
            f"{idx} (({state.residue_number}, '{state.insertion_code}'), "
            f"{mapped})"
        )
