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
3. Deterministic corrections for CDR loops, DE loop, and FR1 positions
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

    # Build search order based on direction preference
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
            if len(rows) == 1:
                return int(rows[0]), col
            elif len(rows) > 1:
                # Multiple matches - take the first one
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

    def correct_de_loop(self, aln: np.ndarray) -> np.ndarray:
        """Fix DE loop alignment at positions 81-83 (0-indexed: 80-82)."""
        pos0, pos1, pos2 = constants.DE_LOOP_POSITIONS
        # DE loop manual fix
        if aln[:, pos0].sum() == 1 and aln[:, pos1 : pos2 + 1].sum() == 0:
            LOGGER.info("Correcting DE loop")
            aln[:, pos2] = aln[:, pos0]
            aln[:, pos0] = 0
        elif (
            aln[:, pos0].sum() == 1
            and aln[:, pos1].sum() == 0
            and aln[:, pos2].sum() == 1
        ):
            LOGGER.info("Correcting DE loop")
            aln[:, pos1] = aln[:, pos0]
            aln[:, pos0] = 0
        return aln

    def correct_fr1_alignment(
        self,
        aln: np.ndarray,
        chain_type: Optional[constants.ChainType] = None,
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
            chain_type: The chain type (used for logging)
            input_has_pos10: Whether the input sequence has position 10
                (True for kappa, False for heavy/lambda)

        Returns:
            Corrected alignment matrix
        """
        fr1_start = constants.LIGHT_CHAIN_FR1_START  # 0-indexed col 5 = pos 6
        fr1_end = constants.LIGHT_CHAIN_FR1_END  # 0-indexed col 9 = pos 10

        # Check for alignment shifts in FR1 region (positions 6-10)
        # Detect if residues are systematically shifted
        for col_idx in range(fr1_start, fr1_end + 1):
            if aln[:, col_idx].sum() == 1:
                row = np.where(aln[:, col_idx] == 1)[0][0]
                # If row > col_idx, alignment is shifted backward
                # (residue at row is placed at earlier column than expected)
                if row > col_idx:
                    shift_amount = row - col_idx
                    LOGGER.info(
                        f"Correcting FR1 alignment: detected shift of "
                        f"{shift_amount} at position {col_idx + 1}"
                    )
                    # Shift all matches forward to correct positions
                    # Work backwards to avoid overwriting
                    for c in range(fr1_end, col_idx - 1, -1):
                        if aln[:, c].sum() == 1:
                            new_col = c + shift_amount
                            if new_col < aln.shape[1]:
                                aln[:, new_col] = aln[:, c]
                                aln[:, c] = 0
                    break

        # For chains without position 10 (heavy, lambda), clear position 10
        # The unified embeddings include position 10 (from kappa), so
        # heavy/lambda residues may incorrectly align there. We need to handle:
        # Case 1: pos10 occupied, pos9 empty -> move to pos9
        # Case 2: pos10 occupied, pos11 empty -> move to pos11
        # Case 3: pos10 occupied, both pos9 and pos11 filled -> just clear
        if not input_has_pos10:
            pos9_col = 8  # 0-indexed column for position 9
            pos10_col = 9  # 0-indexed column for position 10
            pos11_col = 10  # 0-indexed column for position 11

            pos9_occupied = aln[:, pos9_col].sum() == 1
            pos10_occupied = aln[:, pos10_col].sum() == 1
            pos11_occupied = aln[:, pos11_col].sum() == 1

            if pos10_occupied:
                if not pos9_occupied:
                    # Move residue from position 10 to position 9
                    LOGGER.info(
                        "Moving residue from position 10 to position 9 "
                        "(chain lacks position 10)"
                    )
                    aln[:, pos9_col] = aln[:, pos10_col]
                    aln[:, pos10_col] = 0
                elif not pos11_occupied:
                    # Move residue from position 10 to position 11
                    LOGGER.info(
                        "Moving residue from position 10 to position 11 "
                        "(chain lacks position 10)"
                    )
                    aln[:, pos11_col] = aln[:, pos10_col]
                    aln[:, pos10_col] = 0
                else:
                    # Both neighbors occupied - just clear position 10
                    LOGGER.info(
                        "Clearing position 10 (chain lacks position 10)"
                    )
                    aln[:, pos10_col] = 0

        return aln

    def filter_embeddings_by_chain_type(
        self, chain_type: Optional[constants.ChainType]
    ) -> List[mpnn_embeddings.MPNNEmbeddings]:
        """
        Filter embeddings based on chain type.

        Args:
            chain_type: ChainType.HEAVY for H embeddings only,
                       ChainType.LIGHT for K and L embeddings only,
                       None or ChainType.AUTO for all embeddings.

        Returns:
            Filtered list of embeddings.
        """
        # Check for unified embeddings - if present, use for all chain types
        unified_embeddings = [
            emb for emb in self.all_embeddings if emb.name == "unified"
        ]
        if unified_embeddings:
            LOGGER.info("Using unified embeddings for all chain types")
            return unified_embeddings

        if chain_type is None or chain_type == constants.ChainType.AUTO:
            return self.all_embeddings

        filtered = []
        for emb in self.all_embeddings:
            suffix = emb.name[-1].upper()
            if chain_type == constants.ChainType.HEAVY and suffix == "H":
                filtered.append(emb)
            elif chain_type == constants.ChainType.LIGHT and suffix in (
                "K",
                "L",
            ):
                filtered.append(emb)

        if not filtered:
            LOGGER.warning(
                f"No embeddings found for chain_type='{chain_type.value}', "
                f"using all embeddings"
            )
            return self.all_embeddings

        LOGGER.info(
            f"Filtered to {len(filtered)} embeddings for "
            f"chain_type='{chain_type.value}'"
        )
        return filtered

    def __call__(
        self,
        input_data: mpnn_embeddings.MPNNEmbeddings,
        chain_type: Optional[constants.ChainType] = None,
        deterministic_loop_renumbering: bool = True,
    ) -> softalign_output.SoftAlignOutput:
        """
        Align input embeddings to each species embedding and return best hit.

        Args:
            input_data: Pre-computed MPNN embeddings for the query chain.
            chain_type: Optional filter - ChainType.HEAVY for H only,
                ChainType.LIGHT for K/L only, None/AUTO for all.
            deterministic_loop_renumbering: Whether to apply deterministic
                renumbering corrections for:
                - Light chain FR1 positions 7-10
                - DE loop positions 80-85 (all chains)
                - CDR loops (CDR1, CDR2, CDR3) for all chains
                When False, raw alignment output used without corrections
                Default is True.

        Returns:
            SoftAlignOutput with the best alignment.
        """
        LOGGER.info(
            f"Aligning embeddings with length={input_data.embeddings.shape[0]}"
        )

        # Filter embeddings based on chain type
        embeddings_to_search = self.filter_embeddings_by_chain_type(chain_type)

        outputs = {}
        for species_embedding in embeddings_to_search:
            name = species_embedding.name
            out = self.transformed_align_fn.apply(
                self.model_params,
                self.key,
                input_data,
                species_embedding,
                self.temperature,
            )
            aln = self.fix_aln(out.alignment, species_embedding.idxs)

            outputs[name] = softalign_output.SoftAlignOutput(
                alignment=aln,
                score=out.score,
                species=name,
                sim_matrix=None,
                idxs1=input_data.idxs,
                idxs2=[
                    str(x) for x in range(1, constants.IMGT_MAX_POSITION + 1)
                ],
            )
        LOGGER.info(f"Evaluated alignments against {len(outputs)} species")

        best_match = max(outputs, key=lambda k: outputs[k].score)

        aln = np.array(outputs[best_match].alignment, dtype=int)

        if deterministic_loop_renumbering:
            for name, (startres, endres) in constants.IMGT_LOOPS.items():
                startres_idx = startres - 1
                endres_idx = endres - 1

                # Check if there are any aligned residues within the CDR range
                # If the CDR region is entirely empty, skip renumbering
                cdr_region = aln[:, startres_idx:endres]
                cdr_occupancy = cdr_region.sum()
                if cdr_occupancy == 0:
                    LOGGER.info(
                        f"Skipping {name}; no residues aligned within "
                        f"CDR range (cols {startres_idx}-{endres - 1})"
                    )
                    continue

                # Use soft boundary detection: search ±2 positions
                # For start, prefer exact or backward (lower column indices)
                # For end, prefer exact or forward (higher column indices)
                start_row, start_col = find_nearest_occupied_column(
                    aln, startres_idx, search_range=2, direction="both"
                )
                end_row, end_col = find_nearest_occupied_column(
                    aln, endres_idx, search_range=2, direction="both"
                )

                if start_row is None or end_row is None:
                    LOGGER.warning(
                        f"Skipping {name}; missing start "
                        f"(searched {startres_idx}±2) or end "
                        f"(searched {endres_idx}±2)"
                    )
                    continue

                # Validate that start comes before end in sequence
                if start_row >= end_row:
                    LOGGER.warning(
                        f"Skipping {name}; start row ({start_row}) >= "
                        f"end row ({end_row})"
                    )
                    continue

                # Validate detected boundaries overlap with CDR range.
                # If boundaries are outside the CDR range (e.g., spanning
                # a deletion), the soft detection picked up non-CDR residues
                if start_col > endres_idx or end_col < startres_idx:
                    LOGGER.warning(
                        f"Skipping {name}; detected boundaries "
                        f"(cols {start_col}-{end_col}) don't overlap "
                        f"CDR range (cols {startres_idx}-{endres_idx})"
                    )
                    continue

                # Log if we used soft boundaries
                if start_col != startres_idx or end_col != endres_idx:
                    LOGGER.info(
                        f"{name}: soft boundary detection used - "
                        f"start col {start_col} (expected {startres_idx}), "
                        f"end col {end_col} (expected {endres_idx})"
                    )

                # Clear the entire CDR region in the alignment first
                # This prevents conflicts when re-assigning positions
                aln[start_row : end_row + 1, startres_idx:endres] = 0

                # Also clear any soft-detected positions outside the CDR range
                # to prevent duplicate alignments when pulling rows into CDR
                if start_col < startres_idx:
                    aln[start_row, start_col] = 0
                if end_col > endres_idx:
                    aln[end_row, end_col] = 0

                # Extract the sub-alignment using the found row range
                # but the canonical IMGT column range
                n_residues = end_row - start_row + 1
                # Column slice is startres_idx:endres
                n_positions = endres - startres_idx

                # Create a new sub-alignment with correct dimensions
                sub_aln = np.zeros((n_residues, n_positions), dtype=aln.dtype)

                # Apply deterministic gap numbering pattern
                sub_aln = self.correct_gap_numbering(sub_aln)

                # Place the corrected sub-alignment back
                aln[start_row : end_row + 1, startres_idx:endres] = sub_aln

            aln = self.correct_de_loop(aln)

            # Apply FR1 alignment correction
            # Check if input has position 10 (kappa) or not (heavy/lambda)
            input_has_pos10 = "10" in input_data.idxs or 10 in input_data.idxs
            is_light_chain = (
                chain_type == constants.ChainType.LIGHT
                or best_match[-1].upper() in ("K", "L")
            )
            if is_light_chain or chain_type == constants.ChainType.HEAVY:
                aln = self.correct_fr1_alignment(
                    aln, chain_type=chain_type, input_has_pos10=input_has_pos10
                )

        # Determine species to report
        # For unified embeddings, derive from chain_type parameter
        reported_species = best_match
        if best_match == "unified":
            if chain_type == constants.ChainType.HEAVY:
                reported_species = "H"
            elif chain_type == constants.ChainType.LIGHT:
                # Default to K for light chains (most common)
                reported_species = "K"
            else:
                # AUTO or None - default to H
                reported_species = "H"
            LOGGER.info(
                f"Unified embeddings: reporting species as "
                f"'{reported_species}' based on chain_type={chain_type}"
            )

        return softalign_output.SoftAlignOutput(
            species=reported_species,
            alignment=aln,
            score=0,
            sim_matrix=None,
            idxs1=outputs[best_match].idxs1,
            idxs2=outputs[best_match].idxs2,
        )
