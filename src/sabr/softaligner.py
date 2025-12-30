#!/usr/bin/env python3
"""SoftAlign-based antibody sequence alignment module.

This module provides the SoftAligner class which aligns query antibody
embeddings against unified reference embeddings to generate IMGT-compatible
alignments.

Key components:
- SoftAligner: Main class for running alignments

The alignment process includes:
1. Embedding comparison against unified reference
2. Deterministic corrections for CDR loops, DE loop, FR1, and C-terminus
3. Expansion to full 128-position IMGT alignment matrix
4. Chain type detection from DE loop occupancy
"""

import logging
from importlib.resources import as_file, files
from typing import List, Optional, Tuple

import numpy as np

from sabr import constants, jax_backend, mpnn_embeddings, softalign_output, util

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


class SoftAligner:
    """Align a query embedding against unified reference embeddings."""

    def __init__(
        self,
        embeddings_name: str = "embeddings.npz",
        embeddings_path: str = "sabr.assets",
        temperature: float = constants.DEFAULT_TEMPERATURE,
        random_seed: int = 0,
    ) -> None:
        """
        Initialize the SoftAligner by loading reference embeddings and backend.

        Args:
            embeddings_name: Name of the reference embeddings file.
            embeddings_path: Package path containing the embeddings file.
            temperature: Alignment temperature parameter.
            random_seed: Random seed for reproducibility.
        """
        self.unified_embedding = self.read_embeddings(
            embeddings_name=embeddings_name,
            embeddings_path=embeddings_path,
        )
        self.temperature = temperature
        self._backend = jax_backend.AlignmentBackend(random_seed=random_seed)

    def read_embeddings(
        self,
        embeddings_name: str = "embeddings.npz",
        embeddings_path: str = "sabr.assets",
    ) -> mpnn_embeddings.MPNNEmbeddings:
        """Load packaged reference embeddings as ``MPNNEmbeddings``."""
        path = files(embeddings_path) / embeddings_name
        with as_file(path) as p:
            data = np.load(p, allow_pickle=True)
            embedding = mpnn_embeddings.MPNNEmbeddings(
                name=str(data["name"]),
                embeddings=data["array"],
                stdev=data["stdev"],
                idxs=list(data["idxs"]),
            )
        LOGGER.info(f"Loaded embeddings from {path}")
        return embedding

    def correct_gap_numbering(self, sub_aln: np.ndarray) -> np.ndarray:
        """Redistribute loop gaps to an alternating IMGT-style pattern."""
        new_aln = np.zeros_like(sub_aln)
        for i in range(min(sub_aln.shape)):
            pos = ((i + 1) // 2) * ((-1) ** i)
            new_aln[pos, pos] = 1
        return new_aln

    def fix_aln(self, old_aln: np.ndarray, idxs: List[int]) -> np.ndarray:
        """Expand an alignment onto IMGT positions using saved indices."""
        aln = np.zeros(
            (old_aln.shape[0], constants.IMGT_MAX_POSITION), dtype=old_aln.dtype
        )
        aln[:, np.asarray(idxs, dtype=int) - 1] = old_aln

        return aln

    def correct_fr1_alignment(
        self,
        aln: np.ndarray,
        chain_type: Optional[str] = None,
    ) -> np.ndarray:
        """
        Fix FR1 alignment issues in positions 6-12 deterministically.

        Uses anchor positions 6 and 12 to count residues and determine if
        position 10 should be occupied:
        - 7 residues between positions 6-12 → position 10 occupied (kappa)
        - 6 residues between positions 6-12 → position 10 gap (heavy/lambda)

        The residues are then redistributed deterministically to fill
        positions 6-12 with or without position 10.

        Args:
            aln: The alignment matrix
            chain_type: The chain type (used for logging)

        Returns:
            Corrected alignment matrix
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
        self,
        aln: np.ndarray,
        input_has_pos81: bool = False,
        input_has_pos82: bool = False,
    ) -> np.ndarray:
        """
        Fix FR3 alignment issues in positions 81-84 for light chains.

        Light chains (kappa and lambda) typically skip positions 81-82 in IMGT
        numbering, having residues at 79, 80, 83, 84, ... instead of the full
        79, 80, 81, 82, 83, 84, ... pattern seen in heavy chains.

        When using unified embeddings (which include 81-82 from heavy chains),
        the aligner may incorrectly place light chain residues at positions
        81-82 instead of 83-84. This function corrects that misalignment.

        Args:
            aln: The alignment matrix
            input_has_pos81: Whether the input sequence has position 81
            input_has_pos82: Whether the input sequence has position 82

        Returns:
            Corrected alignment matrix
        """
        pos81_col = constants.FR3_POS81_COL
        pos82_col = constants.FR3_POS82_COL
        pos83_col = constants.FR3_POS83_COL
        pos84_col = constants.FR3_POS84_COL

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

    def correct_c_terminus(self, aln: np.ndarray) -> np.ndarray:
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
                LOGGER.warning(
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

    def _correct_cdr_loop(
        self,
        aln: np.ndarray,
        loop_name: str,
        cdr_start: int,
        cdr_end: int,
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
            LOGGER.warning(
                f"Skipping {loop_name}; missing anchor at position "
                f"{anchor_start} (col {anchor_start_col}±2) or "
                f"{anchor_end} (col {anchor_end_col}±2)"
            )
            return aln

        if anchor_start_row >= anchor_end_row:
            LOGGER.warning(
                f"Skipping {loop_name}; anchor start row "
                f"({anchor_start_row}) >= end row ({anchor_end_row})"
            )
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

            sub_aln = np.zeros(
                (n_cdr_residues, n_cdr_positions), dtype=aln.dtype
            )
            sub_aln = self.correct_gap_numbering(sub_aln)
            for i, row in enumerate(cdr_rows):
                aln[row, cdr_start_col : cdr_start_col + n_cdr_positions] = (
                    sub_aln[i, :]
                )

        return aln

    def _apply_deterministic_corrections(
        self, aln: np.ndarray
    ) -> Tuple[np.ndarray, str]:
        """Apply all deterministic alignment corrections.

        Applies corrections in order: CDR loops, FR1, FR3 (light chains),
        and C-terminus.

        Args:
            aln: The raw alignment matrix.

        Returns:
            Tuple of (corrected alignment, detected chain type).
        """
        # Correct all CDR loops
        for loop_name, (cdr_start, cdr_end) in constants.IMGT_LOOPS.items():
            aln = self._correct_cdr_loop(aln, loop_name, cdr_start, cdr_end)

        # Detect chain type from DE loop (positions 81-82)
        detected_chain_type = util.detect_chain_type(aln)
        is_light_chain = detected_chain_type in ("K", "L")

        # Apply FR1 correction
        aln = self.correct_fr1_alignment(aln, chain_type=detected_chain_type)

        # FR3 positions 81-82: Heavy chains have them, light chains don't
        if is_light_chain:
            aln = self.correct_fr3_alignment(
                aln, input_has_pos81=False, input_has_pos82=False
            )

        # Apply C-terminus correction
        aln = self.correct_c_terminus(aln)

        return aln, detected_chain_type

    def __call__(
        self,
        input_data: mpnn_embeddings.MPNNEmbeddings,
        deterministic_loop_renumbering: bool = True,
    ) -> softalign_output.SoftAlignOutput:
        """Align input embeddings against the unified reference embedding.

        Args:
            input_data: Pre-computed MPNN embeddings for the query chain.
            deterministic_loop_renumbering: Whether to apply deterministic
                renumbering corrections for CDR loops, FR1, FR3, and
                C-terminus. Default is True.

        Returns:
            SoftAlignOutput with the best alignment.
        """
        LOGGER.info(
            f"Aligning embeddings with length={input_data.embeddings.shape[0]}"
        )

        alignment, sim_matrix, score = self._backend.align(
            input_embeddings=input_data.embeddings,
            target_embeddings=self.unified_embedding.embeddings,
            target_stdev=self.unified_embedding.stdev,
            temperature=self.temperature,
        )

        aln = self.fix_aln(alignment, self.unified_embedding.idxs)
        aln = np.array(aln, dtype=int)

        if deterministic_loop_renumbering:
            aln, detected_chain_type = self._apply_deterministic_corrections(
                aln
            )
        else:
            detected_chain_type = util.detect_chain_type(aln)
            LOGGER.info(f"Detected chain type: {detected_chain_type}")

        return softalign_output.SoftAlignOutput(
            chain_type=detected_chain_type,
            alignment=aln,
            score=score,
            sim_matrix=sim_matrix,
            idxs1=input_data.idxs,
            idxs2=[str(x) for x in range(1, constants.IMGT_MAX_POSITION + 1)],
        )
