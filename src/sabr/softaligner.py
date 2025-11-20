#!/usr/bin/env python3

import logging
import pickle
from importlib.resources import as_file, files
from typing import Any, Dict, List, Tuple

import haiku as hk
import jax
import numpy as np

from sabr import constants, ops, types

LOGGER = logging.getLogger(__name__)


class SoftAligner:
    """Embed a query chain and align it against packaged species embeddings."""

    def __init__(
        self,
        params_name: str = "CONT_SW_05_T_3_1",
        params_path: str = "softalign.models",
        embeddings_name: str = "embeddings.npz",
        embeddings_path: str = "sabr.assets",
        temperature: float = 10**-4,
        random_seed: int = 0,
    ) -> None:
        """
        Initialize the SoftAligner by loading model parameters and embeddings.
        """
        self.all_embeddings = self.read_embeddings(
            embeddings_name=embeddings_name,
            embeddings_path=embeddings_path,
        )
        self.model_params = self.read_softalign_params(
            params_name=params_name, params_path=params_path
        )
        self.temperature = temperature
        self.key = jax.random.PRNGKey(random_seed)
        self.transformed_align_fn = hk.transform(ops.align_fn)
        self.transformed_embed_fn = hk.transform(ops.embed_fn)

    def read_softalign_params(
        self,
        params_name: str = "CONT_SW_05_T_3_1",
        params_path: str = "softalign.models",
    ) -> Dict[str, Any]:
        """Load SoftAlign parameters from package resources."""
        path = files(params_path) / params_name
        params = pickle.load(open(path, "rb"))
        LOGGER.info(f"Loaded model parameters from {path}")
        return params

    def normalize(self, mp: types.MPNNEmbeddings) -> types.MPNNEmbeddings:
        """Return embeddings reordered by sorted integer indices."""
        idxs_int = [int(x) for x in mp.idxs]
        order = np.argsort(np.asarray(idxs_int, dtype=np.int64))
        if not np.array_equal(order, np.arange(len(order))):
            norm_msg = (
                f"Normalizing embedding order for {mp.name} "
                f"(size={len(order)})"
            )
            LOGGER.debug(norm_msg)
        return types.MPNNEmbeddings(
            name=mp.name,
            embeddings=mp.embeddings[order, ...],
            idxs=[idxs_int[i] for i in order],
        )

    def read_embeddings(
        self,
        embeddings_name: str = "embeddings.npz",
        embeddings_path: str = "sabr.assets",
    ) -> List[types.MPNNEmbeddings]:
        """Load packaged species embeddings as ``MPNNEmbeddings``."""
        out_embeddings = []
        path = files(embeddings_path) / embeddings_name
        with as_file(path) as p:
            data = np.load(p, allow_pickle=True)["arr_0"].item()
            for species, embeddings_dict in data.items():
                out_embeddings.append(
                    types.MPNNEmbeddings(
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
        # aln = np.zeros((old_aln.shape[0], 128))
        # for i, idx in enumerate(idxs):
        #     aln[:, int(idx) - 1] = old_aln[:, i]
        aln = np.zeros((old_aln.shape[0], 128), dtype=old_aln.dtype)
        aln[:, np.asarray(idxs, dtype=int) - 1] = old_aln

        return aln

    def correct_de_loop(self, aln: np.ndarray) -> np.ndarray:
        # DE loop manual fix
        if aln[:, 80].sum() == 1 and aln[:, 81:83].sum() == 0:
            LOGGER.info("Correcting DE loop")
            aln[:, 82] = aln[:, 80]
            aln[:, 80] = 0
        elif (
            aln[:, 80].sum() == 1
            and aln[:, 81].sum() == 0
            and aln[:, 82].sum() == 1
        ):
            LOGGER.info("Correcting DE loop")
            aln[:, 81] = aln[:, 80]
            aln[:, 80] = 0
        return aln

    def __call__(
        self, input_pdb: str, input_chain: str, correct_loops: bool = True
    ) -> Tuple[str, types.SoftAlignOutput]:
        """Align input chain to each species embedding and return best hit."""
        input_data = self.transformed_embed_fn.apply(
            self.model_params, self.key, input_pdb, input_chain
        )
        LOGGER.info(
            f"Computed embeddings for {input_pdb} chain {input_chain} "
            f"(length={input_data.embeddings.shape[0]})"
        )
        outputs = {}
        for species_embedding in self.all_embeddings:
            name = species_embedding.name
            out = self.transformed_align_fn.apply(
                self.model_params,
                self.key,
                input_data,
                species_embedding,
                self.temperature,
            )
            aln = self.fix_aln(out.alignment, species_embedding.idxs)

            outputs[name] = types.SoftAlignOutput(
                alignment=aln,
                score=out.score,
                species=name,
                sim_matrix=None,
                idxs1=input_data.idxs,
                idxs2=[str(x) for x in range(1, 129)],
            )
        LOGGER.info(f"Evaluated alignments against {len(outputs)} species")

        best_match = max(outputs, key=lambda k: outputs[k].score)

        aln = np.array(outputs[best_match].alignment, dtype=int)

        if correct_loops:
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

        return types.SoftAlignOutput(
            species=best_match,
            alignment=aln,
            score=0,
            sim_matrix=None,
            idxs1=outputs[best_match].idxs1,
            idxs2=outputs[best_match].idxs2,
        )
