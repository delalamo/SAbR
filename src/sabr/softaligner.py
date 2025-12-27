#!/usr/bin/env python3
"""SoftAlign-based antibody sequence alignment module.

This module provides the SoftAligner class which aligns query antibody
embeddings against a library of species reference embeddings to identify
the best-matching species and generate IMGT-compatible alignments.

Key components:
- SoftAligner: Main class for running alignments
- _align_fn: Internal alignment function using the SoftAlign neural model

The alignment process includes:
1. Embedding comparison against all species references
2. Selection of best-matching species by similarity score
3. Deterministic corrections for CDR loops, DE loop, FR1, and C-terminus
4. Expansion to full 128-position IMGT alignment matrix
"""

import logging
from importlib.resources import as_file, files
from typing import List, Optional, Tuple

import haiku as hk
import jax
import numpy as np
from jax import numpy as jnp

from sabr import constants, model, mpnn_embeddings, softalign_output, util

LOGGER = logging.getLogger(__name__)


def _align_fn(
    input: mpnn_embeddings.MPNNEmbeddings,
    target: mpnn_embeddings.MPNNEmbeddings,
    temperature: float = constants.DEFAULT_TEMPERATURE,
) -> softalign_output.SoftAlignOutput:
    """Align two embedding sets with the SoftAlign model and return result."""
    input_array = input.embeddings
    target_array = target.embeddings
    target_stdev = jnp.array(target.stdev)
    target_array = target_array / target_stdev

    LOGGER.info(
        f"Running align_fn with input shape {input_array.shape}, "
        f"target shape {target_array.shape}, temperature={temperature}"
    )
    e2e_model = model.create_e2e_model()
    if input_array.ndim != 2 or target_array.ndim != 2:
        raise ValueError(
            "align_fn expects 2D arrays; got shapes "
            f"{input_array.shape} and {target_array.shape}"
        )
    for array_shape in (input_array.shape, target_array.shape):
        if array_shape[1] != constants.EMBED_DIM:
            raise ValueError(
                f"last dim must be {constants.EMBED_DIM}; got "
                f"{input_array.shape} and {target_array.shape}"
            )
    lens = jnp.array([input_array.shape[0], target_array.shape[0]])[None, :]
    batched_input = jnp.array(input_array[None, :])
    batched_target = jnp.array(target_array[None, :])
    alignment, sim_matrix, score = e2e_model.align(
        batched_input, batched_target, lens, temperature
    )
    LOGGER.debug(
        "Alignment complete: alignment shape "
        f"{alignment.shape}, sim_matrix shape {sim_matrix.shape}, "
        f"score={float(score[0])}"
    )
    return softalign_output.SoftAlignOutput(
        alignment=np.asarray(alignment[0]),
        sim_matrix=np.asarray(sim_matrix[0]),
        score=float(score[0]),
        species=None,
        idxs1=input.idxs,
        idxs2=target.idxs,
    )


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
    """Align a query embedding against packaged species embeddings."""

    def __init__(
        self,
        params_name: str = "CONT_SW_05_T_3_1",
        params_path: str = "softalign.models",
        embeddings_name: str = "embeddings.npz",
        embeddings_path: str = "sabr.assets",
        temperature: float = constants.DEFAULT_TEMPERATURE,
        random_seed: int = 0,
    ) -> None:
        """
        Initialize the SoftAligner by loading model parameters and embeddings.
        """
        self.all_embeddings = self.read_embeddings(
            embeddings_name=embeddings_name,
            embeddings_path=embeddings_path,
        )
        self.model_params = util.read_softalign_params(
            params_name=params_name, params_path=params_path
        )
        self.temperature = temperature
        self.key = jax.random.PRNGKey(random_seed)
        self.transformed_align_fn = hk.transform(_align_fn)

    def normalize(
        self, mp: mpnn_embeddings.MPNNEmbeddings
    ) -> mpnn_embeddings.MPNNEmbeddings:
        """Return embeddings reordered by sorted integer indices."""
        idxs_int = [int(x) for x in mp.idxs]
        order = np.argsort(np.asarray(idxs_int, dtype=np.int64))
        if not np.array_equal(order, np.arange(len(order))):
            norm_msg = (
                f"Normalizing embedding order for {mp.name} "
                f"(size={len(order)})"
            )
            LOGGER.debug(norm_msg)
        return mpnn_embeddings.MPNNEmbeddings(
            name=mp.name,
            embeddings=mp.embeddings[order, ...],
            idxs=[idxs_int[i] for i in order],
            stdev=mp.stdev[order, ...],
        )

    def read_embeddings(
        self,
        embeddings_name: str = "embeddings.npz",
        embeddings_path: str = "sabr.assets",
    ) -> List[mpnn_embeddings.MPNNEmbeddings]:
        """Load packaged species embeddings as ``MPNNEmbeddings``."""
        out_embeddings = []
        path = files(embeddings_path) / embeddings_name
        with as_file(path) as p:
            data = np.load(p, allow_pickle=True)["arr_0"].item()
            for species, embeddings_dict in data.items():
                out_embeddings.append(
                    mpnn_embeddings.MPNNEmbeddings(
                        name=species,
                        embeddings=embeddings_dict.get("array"),
                        stdev=embeddings_dict.get("stdev"),
                        idxs=embeddings_dict.get("idxs"),
                    )
                )
        if len(out_embeddings) == 0:
            raise RuntimeError(f"Couldn't load from {path}")
        LOGGER.info(f"Loaded {len(out_embeddings)} embeddings from {path}")
        return out_embeddings

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
        input_has_pos10: bool = False,
    ) -> np.ndarray:
        """
        Fix FR1 alignment issues in positions 6-11 for all chain types.

        Antibody chains can have shifted alignments in FR1 where the row
        (sequence position) doesn't match the expected column (IMGT position).
        For example, row 7 at column 6 means residue 8 is at position 7.

        Position 10 handling:
        - Kappa light chains: position 10 is occupied (input_has_pos10=True)
        - Lambda light chains and Heavy chains: position 10 is skipped

        With unified embeddings, position 10 exists in the reference but may
        not apply to all input chains. This function detects and corrects
        alignment shifts in FR1 regardless of position 10 occupancy.

        Args:
            aln: The alignment matrix
            input_has_pos10: Whether the input sequence has position 10
                (True for kappa, False for heavy/lambda)

        Returns:
            Corrected alignment matrix
        """
        fr1_start = constants.LIGHT_CHAIN_FR1_START
        fr1_end = constants.LIGHT_CHAIN_FR1_END

        for col_idx in range(fr1_start, fr1_end + 1):
            if aln[:, col_idx].sum() == 1:
                row = np.where(aln[:, col_idx] == 1)[0][0]
                if row > col_idx:
                    shift_amount = row - col_idx
                    LOGGER.info(
                        f"Correcting FR1 alignment: detected shift of "
                        f"{shift_amount} at position {col_idx + 1}"
                    )
                    for c in range(fr1_end, col_idx - 1, -1):
                        if aln[:, c].sum() == 1:
                            new_col = c + shift_amount
                            if new_col < aln.shape[1]:
                                aln[:, new_col] = aln[:, c]
                                aln[:, c] = 0
                    break

        if not input_has_pos10:
            pos9_col, pos10_col, pos11_col = 8, 9, 10

            pos9_occupied = aln[:, pos9_col].sum() == 1
            pos10_occupied = aln[:, pos10_col].sum() == 1
            pos11_occupied = aln[:, pos11_col].sum() == 1

            if pos10_occupied:
                if not pos9_occupied:
                    LOGGER.info(
                        "Moving residue from position 10 to position 9 "
                        "(chain lacks position 10)"
                    )
                    aln[:, pos9_col] = aln[:, pos10_col]
                    aln[:, pos10_col] = 0
                elif not pos11_occupied:
                    LOGGER.info(
                        "Moving residue from position 10 to position 11 "
                        "(chain lacks position 10)"
                    )
                    aln[:, pos11_col] = aln[:, pos10_col]
                    aln[:, pos10_col] = 0
                else:
                    LOGGER.info(
                        "Clearing position 10 (chain lacks position 10)"
                    )
                    aln[:, pos10_col] = 0

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
        pos81_col, pos82_col, pos83_col, pos84_col = 80, 81, 82, 83

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

    def __call__(
        self,
        input_data: mpnn_embeddings.MPNNEmbeddings,
        deterministic_loop_renumbering: bool = True,
    ) -> softalign_output.SoftAlignOutput:
        """
        Align input embeddings against the unified reference embedding.

        Args:
            input_data: Pre-computed MPNN embeddings for the query chain.
            deterministic_loop_renumbering: Whether to apply deterministic
                renumbering corrections for:
                - Light chain FR1 positions 7-10
                - DE loop positions 80-85 (all chains)
                - CDR loops (CDR1, CDR2, CDR3) for all chains
                - C-terminus positions 126-128 (all chains)
                When False, raw alignment output used without corrections
                Default is True.

        Returns:
            SoftAlignOutput with the best alignment.
        """
        LOGGER.info(
            f"Aligning embeddings with length={input_data.embeddings.shape[0]}"
        )

        # Use the single unified embedding
        unified_embedding = self.all_embeddings[0]
        out = self.transformed_align_fn.apply(
            self.model_params,
            self.key,
            input_data,
            unified_embedding,
            self.temperature,
        )
        aln = self.fix_aln(out.alignment, unified_embedding.idxs)
        aln = np.array(aln, dtype=int)

        if deterministic_loop_renumbering:
            for loop_name, (startres, endres) in constants.IMGT_LOOPS.items():
                startres_idx = startres - 1
                endres_idx = endres - 1

                cdr_region = aln[:, startres_idx:endres]
                if cdr_region.sum() == 0:
                    LOGGER.info(
                        f"Skipping {loop_name}; no residues aligned within "
                        f"CDR range (cols {startres_idx}-{endres - 1})"
                    )
                    continue

                start_row, start_col = find_nearest_occupied_column(
                    aln, startres_idx, search_range=2, direction="both"
                )
                end_row, end_col = find_nearest_occupied_column(
                    aln, endres_idx, search_range=2, direction="both"
                )

                if start_row is None or end_row is None:
                    LOGGER.warning(
                        f"Skipping {loop_name}; missing start "
                        f"(searched {startres_idx}±2) or end "
                        f"(searched {endres_idx}±2)"
                    )
                    continue

                if start_row >= end_row:
                    LOGGER.warning(
                        f"Skipping {loop_name}; start row ({start_row}) >= "
                        f"end row ({end_row})"
                    )
                    continue

                if start_col > endres_idx or end_col < startres_idx:
                    LOGGER.warning(
                        f"Skipping {loop_name}; detected boundaries "
                        f"(cols {start_col}-{end_col}) don't overlap "
                        f"CDR range (cols {startres_idx}-{endres_idx})"
                    )
                    continue

                if start_col != startres_idx or end_col != endres_idx:
                    LOGGER.info(
                        f"{loop_name}: soft boundary detection used - "
                        f"start col {start_col} (expected {startres_idx}), "
                        f"end col {end_col} (expected {endres_idx})"
                    )

                aln[start_row : end_row + 1, startres_idx:endres] = 0

                if start_col < startres_idx:
                    aln[start_row, start_col] = 0
                if end_col > endres_idx:
                    aln[end_row, end_col] = 0

                n_residues = end_row - start_row + 1
                n_positions = endres - startres_idx

                sub_aln = np.zeros((n_residues, n_positions), dtype=aln.dtype)
                sub_aln = self.correct_gap_numbering(sub_aln)
                aln[start_row : end_row + 1, startres_idx:endres] = sub_aln

            # Detect chain type from DE loop (positions 81-82)
            detected_chain_type = util.detect_chain_type(aln)
            is_light_chain = detected_chain_type in ("K", "L")

            input_has_pos10 = "10" in input_data.idxs or 10 in input_data.idxs
            aln = self.correct_fr1_alignment(
                aln, input_has_pos10=input_has_pos10
            )

            input_has_pos81 = "81" in input_data.idxs or 81 in input_data.idxs
            input_has_pos82 = "82" in input_data.idxs or 82 in input_data.idxs
            if is_light_chain and (not input_has_pos81 or not input_has_pos82):
                aln = self.correct_fr3_alignment(
                    aln,
                    input_has_pos81=input_has_pos81,
                    input_has_pos82=input_has_pos82,
                )

            # Apply C-terminus correction for unassigned trailing residues
            aln = self.correct_c_terminus(aln)

        # Detect chain type from alignment for reporting
        reported_species = util.detect_chain_type(aln)
        LOGGER.info(f"Detected chain type: {reported_species}")

        return softalign_output.SoftAlignOutput(
            species=reported_species,
            alignment=aln,
            score=out.score,
            sim_matrix=None,
            idxs1=input_data.idxs,
            idxs2=[str(x) for x in range(1, constants.IMGT_MAX_POSITION + 1)],
        )
