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
        out: List of State objects representing the HMM state vector
        start: First IMGT column index (0-indexed), used for leading dashes
        end: Value such that subsequence = "-" * start + sequence[:end-start]
             has sufficient length for all mapped_residue values
        first_aligned_row: First sequence row (0-indexed) that is aligned,
             used for alignment_start in thread_alignment
    """
    if matrix.ndim != 2:
        raise ValueError("matrix must be 2D")
    LOGGER.info(f"Converting alignment matrix with shape {matrix.shape}")

    # Extract path: list of [imgt_col, seq_row] pairs where alignment is 1
    # After transpose, indexing is (imgt_col, seq_row)
    path = sorted(np.argwhere(np.transpose(matrix) == 1).tolist())
    if len(path) == 0:
        raise ValueError(
            "Alignment matrix contains no path (no non-zero elements found)"
        )

    # Build column-to-rows mapping for handling matches and insertions
    col_to_rows: dict = {}
    for col, row in path:
        if col not in col_to_rows:
            col_to_rows[col] = []
        col_to_rows[col].append(row)

    # Build row-to-column mapping to identify orphan residues
    row_to_col: dict = {}
    for col, row in path:
        row_to_col[row] = col

    # Identify orphan residues (not in any path entry)
    # These are residues between the first and last aligned positions
    # that don't map to any IMGT column (e.g., CDR3 insertions)
    first_aligned_row = path[0][1]
    last_aligned_row = path[-1][1]
    orphan_rows = set()
    for row in range(first_aligned_row, last_aligned_row + 1):
        if row not in row_to_col:
            orphan_rows.add(row)

    if orphan_rows:
        LOGGER.info(
            f"Found {len(orphan_rows)} orphan residues (CDR insertions)"
        )

    # Offset for mapped_residue: the subsequence has leading dashes
    # equal to the first IMGT column index
    offset = path[0][0]

    out = []

    # Generate states for all columns from first to last in the alignment
    for col in range(path[0][0], path[-1][0] + 1):
        # IMGT positions are 1-indexed
        imgt_pos = col + 1

        if col in col_to_rows:
            rows = col_to_rows[col]
            # First row at this column is a match state
            first_row = rows[0]
            out.append(State(imgt_pos, "m", first_row + offset))

            # Additional rows at same column are insertion states
            for row in rows[1:]:
                out.append(State(imgt_pos, "i", row + offset))

            # Check for orphan residues after this match but before next match
            # These are CDR insertions that don't map to any IMGT column
            next_matched_row = None
            for next_col in range(col + 1, path[-1][0] + 1):
                if next_col in col_to_rows:
                    next_matched_row = col_to_rows[next_col][0]
                    break

            if next_matched_row is not None:
                # Add orphan residues between last row here and next match
                last_row_here = rows[-1]
                for orphan_row in range(last_row_here + 1, next_matched_row):
                    if orphan_row in orphan_rows:
                        out.append(State(imgt_pos, "i", orphan_row + offset))
        else:
            # No sequence position maps to this column - delete state
            out.append(State(imgt_pos, "d", None))

    report_output(out)

    max_row = last_aligned_row
    if orphan_rows:
        max_row = max(max_row, max(orphan_rows))
    start = path[0][0]
    end = max_row + 1 + start

    return out, start, end, first_aligned_row


def report_output(out: List[State]) -> None:
    """Log each HMM state in ``out`` at INFO level."""
    LOGGER.info(f"Reporting {len(out)} HMM states")
    for idx, st in enumerate(out):
        if st.mapped_residue is None:
            LOGGER.info(
                f"{idx} (({st.residue_number}, '{st.insertion_code}'), None)"
            )
        else:
            LOGGER.info(
                f"{idx} (({st.residue_number}, '{st.insertion_code}'), "
                f"{st.mapped_residue})"
            )
