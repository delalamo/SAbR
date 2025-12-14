#!/usr/bin/env python3

import logging
from importlib.resources import as_file, files
from typing import List, Optional, Tuple

import haiku as hk
import jax
import numpy as np
from jax import numpy as jnp
from softalign import END_TO_END_MODELS

from sabr import constants, mpnn_embeddings, softalign_output, util

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
    e2e_model = END_TO_END_MODELS.END_TO_END(
        constants.EMBED_DIM,
        constants.EMBED_DIM,
        constants.EMBED_DIM,
        constants.N_MPNN_LAYERS,
        constants.EMBED_DIM,
        affine=True,
        soft_max=False,
        dropout=0.0,
        augment_eps=0.0,
    )
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
        alignment=alignment[0],
        sim_matrix=sim_matrix[0],
        score=float(score[0]),
        species=None,
        idxs1=input.idxs,
        idxs2=target.idxs,
    )


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

    def fix_aln(self, old_aln, idxs):
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

    def correct_light_chain_fr1(self, aln: np.ndarray) -> np.ndarray:
        """
        Fix light chain FR1 alignment issues in positions 6-11.

        Light chains often have shifted alignments in FR1 where the row
        (sequence position) doesn't match the expected column (IMGT position).
        For example, row 7 at column 6 means residue 8 is at position 7.

        This fix shifts matches forward when position 10 is empty but
        earlier positions (7-9) have matches from later rows.
        """
        fr1_start = constants.LIGHT_CHAIN_FR1_START
        fr1_end = constants.LIGHT_CHAIN_FR1_END
        # Check if position 10 (0-indexed: fr1_end) is empty
        if aln[:, fr1_end].sum() == 0:
            # Find matches in positions 6-9 (0-indexed: fr1_start to fr1_end-1)
            for col_idx in range(fr1_start, fr1_end):
                if aln[:, col_idx].sum() == 1:
                    row = np.where(aln[:, col_idx] == 1)[0][0]
                    # If row > col_idx, the alignment is shifted
                    # (residue row+1 at position col_idx+1,
                    # expected at row+1)
                    if row > col_idx:
                        shift_amount = row - col_idx
                        LOGGER.info(
                            f"Correcting light chain FR1: detected shift of "
                            f"{shift_amount} at position {col_idx + 1}"
                        )
                        # Shift all matches from col_idx to fr1_end-1 forward
                        for c in range(fr1_end - 1, col_idx - 1, -1):
                            if aln[:, c].sum() == 1:
                                aln[:, c + shift_amount] = aln[:, c]
                                aln[:, c] = 0
                        break

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
    ) -> Tuple[str, softalign_output.SoftAlignOutput]:
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
                loop_start = np.where(aln[:, startres_idx] == 1)[0]
                loop_end = np.where(aln[:, endres - 1] == 1)[0]
                if len(loop_start) == 0 or len(loop_end) == 0:
                    LOGGER.warning(
                        (
                            f"Skipping {name}; missing start ({loop_start}) "
                            f"or end ({loop_end})"
                        )
                    )
                    continue
                if len(loop_start) > 1 or len(loop_end) > 1:
                    raise RuntimeError(f"Multiple start/end for loop {name}")
                loop_start, loop_end = loop_start[0], loop_end[0]
                sub_aln = aln[loop_start:loop_end, startres_idx:endres]
                aln[loop_start:loop_end, startres_idx:endres] = (
                    self.correct_gap_numbering(sub_aln)
                )

            aln = self.correct_de_loop(aln)

            # Apply light chain FR1 correction only for light chains
            if best_match[-1].upper() in ("K", "L"):
                aln = self.correct_light_chain_fr1(aln)

        return softalign_output.SoftAlignOutput(
            species=best_match,
            alignment=aln,
            score=0,
            sim_matrix=None,
            idxs1=outputs[best_match].idxs1,
            idxs2=outputs[best_match].idxs2,
        )
