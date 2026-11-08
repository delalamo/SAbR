"""Deterministic CDR and DE-loop corrections.

Alignment matrices are binary [query residues, 128] arrays for one IMGT
domain. Columns are zero-based (IMGT position minus one); gap indices mark
the query row before a peptide break. Corrections mutate and return the
same matrix, leaving central insertion rows unassigned where appropriate.

Each region correction ends in one ``CorrectionStatus``. ``correct_regions``
returns those outcomes without reporting them, so the pipeline can report a
region once its outcome is final. The other public functions report each
outcome immediately through ``report_correction``.
"""

import logging
import warnings

import numpy as np

from sabr import constants
from sabr.types import CorrectionStatus, RegionCorrection

LOGGER = logging.getLogger(__name__)


def _aligned_row(aln: np.ndarray, target_col: int) -> int | None:
    """Return the exact anchor row; nearby matches have different boundaries."""
    rows = np.flatnonzero(aln[:, target_col] == 1)
    return int(rows[0]) if len(rows) else None


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


def _structural_gap(
    gap_indices: frozenset[int] | None,
    start_row: int,
    end_row: int,
    region: str,
    region_name: str,
) -> RegionCorrection | None:
    """Return the skip outcome when a regional correction crosses a gap."""
    if gap_indices and any(
        index in gap_indices for index in range(start_row, end_row)
    ):
        return RegionCorrection(
            region,
            CorrectionStatus.STRUCTURAL_GAP,
            f"Skipping {region_name} deterministic correction: structural "
            f"gap detected between rows {start_row} and {end_row}; using "
            "the raw embedding alignment for this region.",
        )
    return None


def _correct_cdr(
    aln: np.ndarray,
    loop_name: str,
    cdr_start: int,
    cdr_end: int,
    gap_indices: frozenset[int] | None,
) -> RegionCorrection:
    """Correct one CDR in place and return its unreported outcome."""
    anchor_start, anchor_end = constants.CDR_ANCHORS[loop_name]
    anchor_start_row = _aligned_row(aln, anchor_start - 1)
    anchor_end_row = _aligned_row(aln, anchor_end - 1)
    if anchor_start_row is None or anchor_end_row is None:
        return RegionCorrection(
            loop_name,
            CorrectionStatus.MISSING_ANCHOR,
            f"Skipping {loop_name}; missing anchor at IMGT {anchor_start} "
            f"or {anchor_end}; preserving the learned alignment.",
        )

    if anchor_start_row >= anchor_end_row:
        return RegionCorrection(
            loop_name,
            CorrectionStatus.ANCHORS_OUT_OF_ORDER,
            f"Skipping {loop_name}; anchor start row "
            f"({anchor_start_row}) >= end row ({anchor_end_row})",
        )

    gap = _structural_gap(
        gap_indices, anchor_start_row, anchor_end_row, loop_name, loop_name
    )
    if gap is not None:
        return gap

    fw_before_cdr = list(range(anchor_start + 1, cdr_start))
    fw_after_cdr = list(range(cdr_end + 1, anchor_end))
    n_fw_before = len(fw_before_cdr)
    n_fw_after = len(fw_after_cdr)

    intermediate_rows = list(range(anchor_start_row + 1, anchor_end_row))
    n_residues = len(intermediate_rows)

    if n_residues < n_fw_before + n_fw_after:
        return RegionCorrection(
            loop_name,
            CorrectionStatus.TOO_FEW_RESIDUES,
            f"Skipping {loop_name}; not enough residues "
            f"({n_residues}) between anchors for FW positions "
            f"({n_fw_before} + {n_fw_after})",
        )

    n_cdr_residues = n_residues - n_fw_before - n_fw_after

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

    return RegionCorrection(
        loop_name,
        CorrectionStatus.APPLIED,
        f"{loop_name}: anchors at {anchor_start} (row "
        f"{anchor_start_row}) and {anchor_end} (row "
        f"{anchor_end_row}). {n_residues} residues: "
        f"{n_fw_before} FW, {n_cdr_residues} CDR, {n_fw_after} FW",
    )


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


def _correct_de(
    aln: np.ndarray, gap_indices: frozenset[int] | None
) -> RegionCorrection:
    """Correct the DE loop in place and return its unreported outcome."""
    anchor_79_row = _aligned_row(aln, 78)
    anchor_85_row = _aligned_row(aln, 84)

    if anchor_79_row is None or anchor_85_row is None:
        return RegionCorrection(
            "DE",
            CorrectionStatus.MISSING_ANCHOR,
            "Skipping DE loop correction; missing anchor at IMGT 79 or 85; "
            "preserving the learned alignment.",
        )
    if anchor_79_row >= anchor_85_row:
        return RegionCorrection(
            "DE",
            CorrectionStatus.ANCHORS_OUT_OF_ORDER,
            f"Skipping DE loop correction; anchor 79 row ({anchor_79_row}) "
            f"is not before anchor 85 row ({anchor_85_row}).",
        )
    gap = _structural_gap(
        gap_indices, anchor_79_row, anchor_85_row, "DE", "DE loop (79-85)"
    )
    if gap is not None:
        return gap

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

    return RegionCorrection(
        "DE",
        CorrectionStatus.APPLIED,
        (
            f"DE loop correction: assigned {len(intermediate_rows)} "
            f"residue(s) to {positions}."
            if intermediate_rows
            else None
        ),
    )


def correct_regions(
    aln: np.ndarray,
    gap_indices: frozenset[int] | None = None,
) -> tuple[RegionCorrection, ...]:
    """Correct CDR1-3, then the DE loop, in place; return each outcome.

    Nothing is reported, so a caller can decide which outcomes are final.
    """
    cdrs = tuple(
        _correct_cdr(aln, loop_name, cdr_start, cdr_end, gap_indices)
        for loop_name, (cdr_start, cdr_end) in constants.IMGT_LOOPS.items()
    )
    return (*cdrs, _correct_de(aln, gap_indices))


def report_correction(
    correction: RegionCorrection, *, stacklevel: int = 1
) -> None:
    """Log an outcome, warning with UserWarning for a structural-gap skip.

    Applied corrections log at INFO and other skips at WARNING. As in
    ``warnings.warn``, ``stacklevel`` is counted from the caller.
    """
    if correction.message is None:
        return
    if correction.status is CorrectionStatus.APPLIED:
        LOGGER.info("%s", correction.message)
    elif correction.status is CorrectionStatus.STRUCTURAL_GAP:
        warnings.warn(
            correction.message, UserWarning, stacklevel=stacklevel + 1
        )
    else:
        LOGGER.warning("%s", correction.message)


def correct_cdr_loop(
    aln: np.ndarray,
    loop_name: str,
    cdr_start: int,
    cdr_end: int,
    gap_indices: frozenset[int] | None = None,
) -> np.ndarray:
    """Place framework and CDR residues between two exact IMGT anchors.

    Missing anchors leave the learned alignment intact: treating a nearby
    match as the requested position can overwrite or duplicate assignments.
    """
    report_correction(
        _correct_cdr(aln, loop_name, cdr_start, cdr_end, gap_indices),
        stacklevel=2,
    )
    return aln


def correct_de_loop(
    aln: np.ndarray,
    gap_indices: frozenset[int] | None = None,
) -> np.ndarray:
    """Assign the residues between IMGT 79 and 85 by loop length."""
    report_correction(_correct_de(aln, gap_indices), stacklevel=2)
    return aln


def apply_corrections(
    aln: np.ndarray,
    gap_indices: frozenset[int] | None = None,
) -> np.ndarray:
    """Apply deterministic CDR and DE-loop corrections, reporting each."""
    for correction in correct_regions(aln, gap_indices):
        report_correction(correction, stacklevel=2)
    return aln
