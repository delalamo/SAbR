"""Deterministic CDR and DE-loop corrections.

Alignment matrices are binary [query residues, 128] arrays for one IMGT
domain. Columns are zero-based (IMGT position minus one); gap indices mark
the query row before a peptide break. Corrections mutate and return the
same matrix, leaving central insertion rows unassigned where appropriate.
"""

import logging
import warnings

import numpy as np

from sabr import constants

LOGGER = logging.getLogger(__name__)


def _aligned_row_near(aln: np.ndarray, target_col: int) -> int | None:
    """Return the row aligned at or within two columns of a target."""
    for offset in (0, -1, 1, -2, 2):
        column = target_col + offset
        if 0 <= column < aln.shape[1]:
            rows = np.where(aln[:, column] == 1)[0]
            if len(rows):
                return int(rows[0])

    return None


def cdr_columns(residue_count: int, position_count: int) -> list[int | None]:
    """Return IMGT's outside-in CDR columns, preserving central insertions."""
    columns = [None] * residue_count
    for index in range(min(residue_count, position_count)):
        if index % 2 == 0:
            row = column = index // 2
        else:
            offset = (index + 1) // 2
            row = residue_count - offset
            column = position_count - offset
        columns[row] = column
    return columns


def _skip_for_structural_gap(
    gap_indices: frozenset[int] | None,
    start_row: int,
    end_row: int,
    region_name: str,
) -> bool:
    """Warn and return true when a regional correction crosses a gap."""
    if gap_indices and any(
        index in gap_indices for index in range(start_row, end_row)
    ):
        message = (
            f"Skipping {region_name} deterministic correction: structural "
            f"gap detected between rows {start_row} and {end_row}; using "
            "the raw embedding alignment for this region."
        )
        warnings.warn(message, UserWarning, stacklevel=3)
        return True
    return False


def correct_cdr_loop(
    aln: np.ndarray,
    loop_name: str,
    cdr_start: int,
    cdr_end: int,
    gap_indices: frozenset[int] | None = None,
) -> np.ndarray:
    """Place framework and CDR residues between one pair of anchors."""
    anchor_start, anchor_end = constants.CDR_ANCHORS[loop_name]
    anchor_start_col = anchor_start - 1
    anchor_end_col = anchor_end - 1

    anchor_start_row = _aligned_row_near(aln, anchor_start_col)
    anchor_end_row = _aligned_row_near(aln, anchor_end_col)
    if anchor_start_row is None or anchor_end_row is None:
        LOGGER.warning(
            "Skipping %s; missing anchor near IMGT %d or %d.",
            loop_name,
            anchor_start,
            anchor_end,
        )
        return aln

    if anchor_start_row >= anchor_end_row:
        LOGGER.warning(
            f"Skipping {loop_name}; anchor start row "
            f"({anchor_start_row}) >= end row ({anchor_end_row})"
        )
        return aln

    if _skip_for_structural_gap(
        gap_indices, anchor_start_row, anchor_end_row, loop_name
    ):
        return aln

    fw_before_cdr = list(range(anchor_start + 1, cdr_start))
    fw_after_cdr = list(range(cdr_end + 1, anchor_end))
    n_fw_before = len(fw_before_cdr)
    n_fw_after = len(fw_after_cdr)

    intermediate_rows = list(range(anchor_start_row + 1, anchor_end_row))
    n_residues = len(intermediate_rows)

    if n_residues < n_fw_before + n_fw_after:
        LOGGER.warning(
            f"Skipping {loop_name}; not enough residues "
            f"({n_residues}) between anchors for FW positions "
            f"({n_fw_before} + {n_fw_after})"
        )
        return aln

    n_cdr_residues = n_residues - n_fw_before - n_fw_after

    LOGGER.info(
        f"{loop_name}: anchors at {anchor_start} (row "
        f"{anchor_start_row}) and {anchor_end} (row "
        f"{anchor_end_row}). {n_residues} residues: "
        f"{n_fw_before} FW, {n_cdr_residues} CDR, {n_fw_after} FW"
    )

    aln[intermediate_rows, :] = 0

    for row, position in zip(intermediate_rows, fw_before_cdr):
        aln[row, position - 1] = 1

    fw_after_rows = intermediate_rows[-n_fw_after:] if n_fw_after > 0 else []
    for row, position in zip(fw_after_rows, fw_after_cdr):
        aln[row, position - 1] = 1

    cdr_rows = intermediate_rows[n_fw_before:]
    if n_fw_after > 0:
        cdr_rows = cdr_rows[:-n_fw_after]

    position_count = cdr_end - cdr_start + 1
    for row, column in zip(
        cdr_rows, cdr_columns(len(cdr_rows), position_count)
    ):
        if column is not None:
            aln[row, cdr_start - 1 + column] = 1

    return aln


def de_loop_positions(n_residues: int) -> list[int]:
    """Return IMGT positions for residues between anchors 79 and 85.

    Position 80 is filled first, followed by positions 84 through 81 from
    right to left. Once positions 80 through 84 are occupied, additional
    residues become insertions on position 82.
    """
    if n_residues <= 0:
        return []
    if n_residues <= 5:
        return [80, *range(86 - n_residues, 85)]
    return [80, 81, 82, *([82] * (n_residues - 5)), 83, 84]


def correct_de_loop(
    aln: np.ndarray,
    gap_indices: frozenset[int] | None = None,
) -> np.ndarray:
    """Assign the residues between IMGT 79 and 85 by loop length."""
    anchor_79_row = _aligned_row_near(aln, 78)
    anchor_85_row = _aligned_row_near(aln, 84)

    if anchor_79_row is None or anchor_85_row is None:
        LOGGER.warning(
            "Skipping DE loop correction; missing anchor near IMGT 79 or 85."
        )
        return aln
    if anchor_79_row >= anchor_85_row:
        LOGGER.warning(
            "Skipping DE loop correction; anchor 79 row (%d) is not before "
            "anchor 85 row (%d).",
            anchor_79_row,
            anchor_85_row,
        )
        return aln
    if _skip_for_structural_gap(
        gap_indices, anchor_79_row, anchor_85_row, "DE loop (79-85)"
    ):
        return aln

    intermediate_rows = list(range(anchor_79_row + 1, anchor_85_row))
    positions = de_loop_positions(len(intermediate_rows))

    # Clear the learned assignments for this region before rebuilding it.
    # Additional occurrences of 82 remain unassigned here; alignment_to_states
    # converts those orphan rows to 82A, 82B, and later insertion states.
    aln[intermediate_rows, :] = 0
    assigned_positions = set()
    for row, position in zip(intermediate_rows, positions):
        if position in assigned_positions:
            continue
        aln[row, position - 1] = 1
        assigned_positions.add(position)

    if intermediate_rows:
        LOGGER.info(
            "DE loop correction: assigned %d residue(s) to %s.",
            len(intermediate_rows),
            positions,
        )
    return aln


def apply_corrections(
    aln: np.ndarray,
    gap_indices: frozenset[int] | None = None,
) -> np.ndarray:
    """Apply deterministic CDR and DE-loop corrections."""
    for loop_name, (cdr_start, cdr_end) in constants.IMGT_LOOPS.items():
        aln = correct_cdr_loop(
            aln, loop_name, cdr_start, cdr_end, gap_indices=gap_indices
        )

    correct_de_loop(aln, gap_indices=gap_indices)
    return aln
