#!/usr/bin/env python3
"""Deterministic alignment correction functions.

This module provides functions for correcting soft alignments using
deterministic rules based on IMGT numbering conventions. These corrections
ensure proper numbering of CDR loops, framework regions, and termini.

Key corrections:
- CDR loop corrections (CDR1, CDR2, CDR3)
- FR1 corrections for positions 6-12
- FR3/DE loop corrections for positions 81-84
- C-terminus corrections for positions 126-128
"""

import logging
from typing import FrozenSet, Optional, Tuple

import numpy as np

from sabr.core import constants
from sabr.core.util import detect_chain_type, has_gap_in_region

LOGGER = logging.getLogger(__name__)


def find_nearest_occupied_column(
    aln: np.ndarray,
    target_col: int,
    search_range: int = 2,
    direction: str = "both",
) -> Tuple[Optional[int], Optional[int]]:
    """Find the nearest column with an alignment match within a search window.

    Args:
        aln: The alignment matrix (rows=sequence, cols=IMGT positions).
        target_col: The 0-indexed column to search near.
        search_range: How many columns to search in each direction.
        direction: "both" searches both directions, "forward" only searches
            higher column indices, "backward" only searches lower indices.

    Returns:
        Tuple of (row_index, col_index) where a match was found, or
        (None, None) if no match found in the search window.
    """
    n_cols = aln.shape[1]

    offsets = [0]
    for i in range(1, search_range + 1):
        if direction == "both":
            offsets.extend([-i, i])
        elif direction == "backward":
            offsets.append(-i)
        elif direction == "forward":
            offsets.append(i)

    for offset in offsets:
        col = target_col + offset
        if 0 <= col < n_cols:
            rows = np.where(aln[:, col] == 1)[0]
            if len(rows) >= 1:
                return int(rows[0]), col

    return None, None


def correct_gap_numbering(sub_aln: np.ndarray) -> np.ndarray:
    """Redistribute loop gaps to an alternating IMGT-style pattern."""
    new_aln = np.zeros_like(sub_aln)
    for i in range(min(sub_aln.shape)):
        pos = ((i + 1) // 2) * ((-1) ** i)
        new_aln[pos, pos] = 1
    return new_aln


def _skip_for_structural_gap(
    gap_indices: Optional[FrozenSet[int]],
    start_row: int,
    end_row: int,
    region_name: str,
) -> bool:
    """Check if deterministic correction should be skipped due to a gap.

    Args:
        gap_indices: FrozenSet of row indices where structural gaps occur.
        start_row: First row index of the region to check.
        end_row: Last row index of the region to check (inclusive).
        region_name: Name of the region for logging.

    Returns:
        True if a gap is found and correction should be skipped.
    """
    if gap_indices and has_gap_in_region(gap_indices, start_row, end_row):
        LOGGER.warning(
            f"Skipping {region_name} deterministic correction; "
            f"structural gap detected between rows {start_row} and "
            f"{end_row}. Using embedding similarity instead."
        )
        return True
    return False


def correct_fr1_alignment(
    aln: np.ndarray,
    chain_type: Optional[str] = None,
    gap_indices: Optional[FrozenSet[int]] = None,
) -> np.ndarray:
    """Fix FR1 alignment issues in positions 6-12 deterministically.

    Uses anchor positions 6 and 12 to count residues and determine if
    position 10 should be occupied:
    - 7 residues between positions 6-12 → position 10 occupied (kappa)
    - 6 residues between positions 6-12 → position 10 gap (heavy/lambda)

    The residues are then redistributed deterministically to fill
    positions 6-12 with or without position 10.

    Args:
        aln: The alignment matrix.
        chain_type: The chain type (used for logging).
        gap_indices: FrozenSet of row indices where structural gaps occur.
            If a gap is found in the region, deterministic correction
            is skipped and embedding similarity is used instead.

    Returns:
        Corrected alignment matrix.
    """
    # Anchor columns (0-indexed)
    pos6_col = constants.FR1_ANCHOR_START_COL
    pos12_col = constants.FR1_ANCHOR_END_COL

    # Find rows aligned to positions near anchors 6 and 12
    start_row, _ = find_nearest_occupied_column(
        aln, pos6_col, search_range=2, direction="forward"
    )
    end_row, _ = find_nearest_occupied_column(
        aln, pos12_col, search_range=2, direction="forward"
    )

    if start_row is None or end_row is None or start_row >= end_row:
        LOGGER.debug(
            f"FR1 correction: could not find anchor positions "
            f"(start_row={start_row}, end_row={end_row})"
        )
        return aln

    # Check for structural gaps in the region
    if _skip_for_structural_gap(gap_indices, start_row, end_row, "FR1"):
        return aln

    # Count residues between anchors (inclusive)
    n_residues = end_row - start_row + 1

    # Kappa chains have 7 residues (6,7,8,9,10,11,12)
    # Heavy/Lambda chains have 6 residues (6,7,8,9,11,12 - skip 10)
    should_have_pos10 = n_residues >= constants.FR1_KAPPA_RESIDUE_COUNT

    LOGGER.info(
        f"FR1 correction: {n_residues} residues between rows "
        f"{start_row}-{end_row}, position 10 "
        f"{'occupied' if should_have_pos10 else 'gap'}"
    )

    # Clear the FR1 region (positions 6-12, cols 5-11)
    aln[start_row : end_row + 1, pos6_col : pos12_col + 1] = 0

    # Redistribute residues deterministically
    if should_have_pos10:
        # Kappa: fill positions 6,7,8,9,10,11,12
        target_cols = [5, 6, 7, 8, 9, 10, 11]  # 0-indexed
    else:
        # Heavy/Lambda: fill positions 6,7,8,9,11,12 (skip 10)
        target_cols = [5, 6, 7, 8, 10, 11]  # 0-indexed, skip col 9

    for i, row in enumerate(range(start_row, end_row + 1)):
        if i < len(target_cols):
            aln[row, target_cols[i]] = 1

    return aln


def correct_fr3_alignment(
    aln: np.ndarray,
    input_has_pos81: bool = False,
    input_has_pos82: bool = False,
    gap_indices: Optional[FrozenSet[int]] = None,
) -> np.ndarray:
    """Fix FR3 alignment issues in positions 81-84 for light chains.

    Light chains (kappa and lambda) typically skip positions 81-82 in IMGT
    numbering, having residues at 79, 80, 83, 84, ... instead of the full
    79, 80, 81, 82, 83, 84, ... pattern seen in heavy chains.

    When using unified embeddings (which include 81-82 from heavy chains),
    the aligner may incorrectly place light chain residues at positions
    81-82 instead of 83-84. This function corrects that misalignment.

    Args:
        aln: The alignment matrix.
        input_has_pos81: Whether the input sequence has position 81.
        input_has_pos82: Whether the input sequence has position 82.
        gap_indices: FrozenSet of row indices where structural gaps occur.
            If a gap is found in the DE loop region, deterministic
            correction is skipped and embedding similarity is used instead.

    Returns:
        Corrected alignment matrix.
    """
    pos81_col = constants.FR3_POS81_COL
    pos82_col = constants.FR3_POS82_COL
    pos83_col = constants.FR3_POS83_COL
    pos84_col = constants.FR3_POS84_COL

    # Check for structural gaps in DE loop region (positions 79-84)
    if gap_indices:
        de_start_col = 78  # 0-indexed for position 79
        de_end_col = 83  # 0-indexed for position 84
        de_start_row, _ = find_nearest_occupied_column(
            aln, de_start_col, search_range=2, direction="both"
        )
        de_end_row, _ = find_nearest_occupied_column(
            aln, de_end_col, search_range=2, direction="both"
        )
        if (
            de_start_row is not None
            and de_end_row is not None
            and _skip_for_structural_gap(
                gap_indices, de_start_row, de_end_row, "FR3/DE loop"
            )
        ):
            return aln

    pos81_occupied = aln[:, pos81_col].sum() == 1
    pos82_occupied = aln[:, pos82_col].sum() == 1
    pos83_occupied = aln[:, pos83_col].sum() == 1
    pos84_occupied = aln[:, pos84_col].sum() == 1

    if not input_has_pos81 and pos81_occupied:
        if not pos83_occupied:
            LOGGER.info(
                "Moving residue from position 81 to position 83 "
                "(chain lacks position 81)"
            )
            aln[:, pos83_col] = aln[:, pos81_col]
            aln[:, pos81_col] = 0
            pos83_occupied = True
        else:
            LOGGER.info(
                "Clearing position 81 (chain lacks position 81, "
                "but position 83 already occupied)"
            )
            aln[:, pos81_col] = 0

    if not input_has_pos82 and pos82_occupied:
        if not pos84_occupied:
            LOGGER.info(
                "Moving residue from position 82 to position 84 "
                "(chain lacks position 82)"
            )
            aln[:, pos84_col] = aln[:, pos82_col]
            aln[:, pos82_col] = 0
        else:
            LOGGER.info(
                "Clearing position 82 (chain lacks position 82, "
                "but position 84 already occupied)"
            )
            aln[:, pos82_col] = 0

    return aln


def correct_c_terminus(aln: np.ndarray) -> np.ndarray:
    """Fix C-terminus alignment for the last residues (positions 126-128).

    When residues at the end of the sequence are unassigned after the
    last aligned IMGT position (around 125/126), this function
    deterministically assigns them to positions 127, 128.

    The logic:
    1. Find the last row (sequence position) with any assignment
    2. Find the last column (IMGT position) with any assignment
    3. If there are unassigned rows after the last assigned row,
       and the last assigned column is around position 125 or 126,
       assign those trailing residues to subsequent positions (127, 128)

    Args:
        aln: The alignment matrix (rows=sequence, cols=IMGT positions).

    Returns:
        Corrected alignment matrix with C-terminus residues assigned.
    """
    n_rows, n_cols = aln.shape

    # Find the last row that has any assignment
    row_sums = aln.sum(axis=1)
    assigned_rows = np.where(row_sums > 0)[0]
    if len(assigned_rows) == 0:
        return aln

    last_assigned_row = assigned_rows[-1]

    # Find the last column that has any assignment
    col_sums = aln.sum(axis=0)
    assigned_cols = np.where(col_sums > 0)[0]
    if len(assigned_cols) == 0:
        return aln

    last_assigned_col = assigned_cols[-1]

    # Check if there are unassigned rows after the last assigned row
    # These are residues that weren't aligned to any IMGT position
    n_unassigned_trailing = n_rows - last_assigned_row - 1

    if n_unassigned_trailing <= 0:
        # No unassigned trailing residues
        return aln

    # Only apply the fix if the last assigned column is around
    # position 125 or 126 (0-indexed: 124 or 125)
    # This indicates the C-terminus wasn't fully aligned
    if last_assigned_col < constants.C_TERMINUS_ANCHOR_POSITION:
        LOGGER.debug(
            f"C-terminus: last assigned col {last_assigned_col} is "
            f"before anchor position "
            f"{constants.C_TERMINUS_ANCHOR_POSITION}, skipping correction"
        )
        return aln

    # Assign trailing residues to subsequent IMGT positions
    # Starting from last_assigned_col + 1, up to position 127 (0-indexed)
    LOGGER.info(
        f"Correcting C-terminus: {n_unassigned_trailing} unassigned "
        f"residues after row {last_assigned_row}, "
        f"last assigned col was {last_assigned_col}"
    )

    next_col = last_assigned_col + 1
    for i in range(n_unassigned_trailing):
        row_to_assign = last_assigned_row + 1 + i
        if next_col >= n_cols:
            LOGGER.debug(
                f"C-terminus: cannot assign row {row_to_assign}, "
                f"no more IMGT positions available (max col {n_cols - 1})"
            )
            break

        # Clear any existing assignment in this row (shouldn't be any)
        aln[row_to_assign, :] = 0
        # Assign to the next available IMGT position
        aln[row_to_assign, next_col] = 1
        LOGGER.info(
            f"C-terminus: assigned row {row_to_assign} to "
            f"IMGT position {next_col + 1}"
        )
        next_col += 1

    return aln


def correct_cdr_loop(
    aln: np.ndarray,
    loop_name: str,
    cdr_start: int,
    cdr_end: int,
    gap_indices: Optional[FrozenSet[int]] = None,
) -> np.ndarray:
    """Apply deterministic correction to a single CDR loop region.

    Finds anchor positions flanking the loop, counts residues between them,
    assigns framework positions linearly, and CDR positions in an
    alternating IMGT pattern.

    Args:
        aln: The alignment matrix to correct.
        loop_name: Name of the loop (e.g., "CDR1", "CDR2", "CDR3").
        cdr_start: First IMGT position of the CDR (1-indexed).
        cdr_end: Last IMGT position of the CDR (1-indexed).
        gap_indices: FrozenSet of row indices where structural gaps occur.
            If a gap is found in the region, deterministic correction
            is skipped and embedding similarity is used instead.

    Returns:
        Corrected alignment matrix.
    """
    anchor_start, anchor_end = constants.CDR_ANCHORS[loop_name]
    anchor_start_col = anchor_start - 1
    anchor_end_col = anchor_end - 1

    anchor_start_row, _ = find_nearest_occupied_column(
        aln, anchor_start_col, search_range=2, direction="both"
    )
    anchor_end_row, _ = find_nearest_occupied_column(
        aln, anchor_end_col, search_range=2, direction="both"
    )

    if anchor_start_row is None or anchor_end_row is None:
        # Build detailed info about what's nearby, searching away from CDRs
        details = []
        if anchor_start_row is None:
            # Search backward (away from CDR) with larger range
            closest_row, closest_col = find_nearest_occupied_column(
                aln, anchor_start_col, search_range=10, direction="backward"
            )
            if closest_row is not None:
                details.append(
                    f"closest residue to start anchor {anchor_start}: "
                    f"row {closest_row} at IMGT position {closest_col + 1}"
                )
            else:
                details.append(
                    f"no residues found near start anchor {anchor_start}"
                )
        if anchor_end_row is None:
            # Search forward (away from CDR) with larger range
            closest_row, closest_col = find_nearest_occupied_column(
                aln, anchor_end_col, search_range=10, direction="forward"
            )
            if closest_row is not None:
                details.append(
                    f"closest residue to end anchor {anchor_end}: "
                    f"row {closest_row} at IMGT position {closest_col + 1}"
                )
            else:
                details.append(
                    f"no residues found near end anchor {anchor_end}"
                )
        detail_str = "; ".join(details) if details else ""
        LOGGER.warning(
            f"Skipping {loop_name}; missing anchor at position "
            f"{anchor_start} (col {anchor_start_col}±2) or "
            f"{anchor_end} (col {anchor_end_col}±2). {detail_str}"
        )
        return aln

    if anchor_start_row >= anchor_end_row:
        LOGGER.warning(
            f"Skipping {loop_name}; anchor start row "
            f"({anchor_start_row}) >= end row ({anchor_end_row})"
        )
        return aln

    # Check for structural gaps in the region
    if _skip_for_structural_gap(
        gap_indices, anchor_start_row, anchor_end_row, loop_name
    ):
        return aln

    # Calculate FW positions between anchors (outside CDR range)
    fw_before_cdr = list(range(anchor_start + 1, cdr_start))
    fw_after_cdr = list(range(cdr_end + 1, anchor_end))
    n_fw_before = len(fw_before_cdr)
    n_fw_after = len(fw_after_cdr)

    # Rows between anchors (exclusive of anchor rows)
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

    # Clear alignments for intermediate rows in the region
    region_start_col = anchor_start
    region_end_col = anchor_end - 1
    for row in intermediate_rows:
        aln[row, region_start_col:region_end_col] = 0

    # Assign FW positions before CDR (linear assignment)
    for i, pos in enumerate(fw_before_cdr):
        row = intermediate_rows[i]
        aln[row, pos - 1] = 1

    # Assign FW positions after CDR (linear assignment)
    for i, pos in enumerate(fw_after_cdr):
        row = intermediate_rows[-(n_fw_after - i)]
        aln[row, pos - 1] = 1

    # Assign CDR positions using alternating pattern
    if n_cdr_residues > 0:
        cdr_rows = intermediate_rows[n_fw_before:]
        if n_fw_after > 0:
            cdr_rows = cdr_rows[:-n_fw_after]

        cdr_start_col = cdr_start - 1
        n_cdr_positions = cdr_end - cdr_start + 1

        sub_aln = np.zeros((n_cdr_residues, n_cdr_positions), dtype=aln.dtype)
        sub_aln = correct_gap_numbering(sub_aln)
        for i, row in enumerate(cdr_rows):
            aln[row, cdr_start_col : cdr_start_col + n_cdr_positions] = sub_aln[
                i, :
            ]

    return aln


def apply_deterministic_corrections(
    aln: np.ndarray,
    gap_indices: Optional[FrozenSet[int]] = None,
) -> Tuple[np.ndarray, str]:
    """Apply all deterministic alignment corrections.

    Applies corrections in order: CDR loops, FR1, FR3 (light chains),
    and C-terminus. Regions with structural gaps (as indicated by
    gap_indices) will skip deterministic correction and rely on
    embedding similarity instead.

    Args:
        aln: The raw alignment matrix.
        gap_indices: FrozenSet of row indices where structural gaps occur.
            Gaps are detected from backbone C-N distances exceeding
            the threshold. If None, no gap checking is performed.

    Returns:
        Tuple of (corrected alignment, detected chain type).
    """
    # Correct all CDR loops
    for loop_name, (cdr_start, cdr_end) in constants.IMGT_LOOPS.items():
        aln = correct_cdr_loop(
            aln, loop_name, cdr_start, cdr_end, gap_indices=gap_indices
        )

    # Detect chain type from DE loop (positions 81-82)
    detected_chain_type = detect_chain_type(aln)
    is_light_chain = detected_chain_type in ("K", "L")

    # Apply FR1 correction
    aln = correct_fr1_alignment(
        aln, chain_type=detected_chain_type, gap_indices=gap_indices
    )

    # FR3 positions 81-82: Heavy chains have them, light chains don't
    if is_light_chain:
        aln = correct_fr3_alignment(
            aln,
            input_has_pos81=False,
            input_has_pos82=False,
            gap_indices=gap_indices,
        )

    # Apply C-terminus correction
    aln = correct_c_terminus(aln)

    return aln, detected_chain_type
