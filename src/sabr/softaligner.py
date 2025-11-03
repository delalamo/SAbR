#!/usr/bin/env python3

import logging
import pickle
from importlib.resources import as_file, files
from typing import Any, Dict, List, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from sabr import constants, ops, types

LOGGER = logging.getLogger(__name__)


class SoftAligner:
    """
    Run structure-conditioned alignment against precomputed embeddings.

    This class wraps two Haiku-transformed functions—an embedding function and
    an alignment function—and applies them to an input PDB chain against a set
    of species-specific embeddings. It also provides utilities for reading the
    required assets, normalizing embedding indices, and post-processing the
    alignment to conform with IMGT loop conventions.

    Parameters
    ----------
    params_name : str, optional
        Filename (within `params_path`) of the serialized model parameters
        (e.g., a pickle file). Default is ``"CONT_SW_05_T_3_1"``.
    params_path : str, optional
        Package path where the parameters file can be found. This is passed to
        :func:`importlib.resources.files`. Default is ``"softalign.models"``.
    embeddings_name : str, optional
        Filename (within `embeddings_path`) of the serialized embeddings
        archive (``.npz``). Default is ``"embeddings.npz"``.
    embeddings_path : str, optional
        Package path where the embeddings archive resides. Default is
        ``"sabr.assets"``.
    temperature : float, optional
        Temperature used by the alignment scoring function. Default is
        ``1e-4``.
    random_seed : int, optional
        Seed for JAX PRNG key creation. Default is ``0``.
    DEBUG : bool, optional
        If ``True``, defer asset loading (useful for testing). Default is
        ``False``.

    Attributes
    ----------
    all_embeddings : list[types.MPNNEmbeddings]
        The set of species embeddings loaded from disk (omitted if `DEBUG`).
    model_params : dict
        Haiku/SoftAlign parameters loaded from disk (omitted if `DEBUG`).
    temperature : float
        Temperature used by the alignment function.
    key : jax.random.KeyArray
        PRNG key used by Haiku apply calls.
    transformed_align_fn : hk.Transformed
        Haiku-transformed alignment function.
    transformed_embed_fn : hk.Transformed
        Haiku-transformed embedding function.
    """

    def __init__(
        self,
        params_name: str = "CONT_SW_05_T_3_1",
        params_path: str = "softalign.models",
        embeddings_name: str = "embeddings.npz",
        embeddings_path: str = "sabr.assets",
        temperature: float = 10**-4,
        random_seed: int = 0,
        DEBUG: bool = False,
    ) -> None:
        """
        Initialize the SoftAligner by loading model parameters and embeddings.
        """
        init_parts = [
            f"params={params_name}",
            f"embeddings={embeddings_name}",
            f"temperature={temperature}",
            f"seed={random_seed}",
        ]
        init_msg = "Initializing SoftAligner with " + ", ".join(init_parts)
        LOGGER.info(init_msg)
        if not DEBUG:
            self.all_embeddings = self.read_embeddings(
                embeddings_name=embeddings_name,
                embeddings_path=embeddings_path,
            )
            embed_count = len(self.all_embeddings)
            LOGGER.info(f"Loaded {embed_count} species embeddings")
            self.model_params = self.read_softalign_params(
                params_name=params_name, params_path=params_path
            )
            param_msg = (
                "SoftAligner model parameters loaded "
                f"({len(self.model_params)} top-level entries)"
            )
            LOGGER.debug(param_msg)
        self.temperature = temperature
        self.key = jax.random.PRNGKey(random_seed)
        self.transformed_align_fn = hk.transform(ops.align_fn)
        self.transformed_embed_fn = hk.transform(ops.embed_fn)
        if DEBUG:
            LOGGER.debug("DEBUG mode enabled; asset loading deferred")

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
                        idxs=embeddings_dict.get("idxs"),
                    )
                )
        if len(out_embeddings) == 0:
            raise RuntimeError(f"Couldn't load from {path}")
        LOGGER.info(f"Loaded {len(out_embeddings)} embeddings from {path}")
        LOGGER.debug(
            "Embeddings include species: "
            f"{', '.join(sorted(e.name for e in out_embeddings))}"
        )
        return out_embeddings

    def calc_matches(
        self,
        aln: jnp.ndarray,
        res1: List[str],
        res2: List[str],
    ) -> Dict[str, str]:
        """Map residues from binary alignment while skipping IMGT gaps."""
        if aln.ndim != 2:
            raise ValueError(f"Alignment must be 2D; got shape {aln.shape}")
        if aln.shape[0] != len(res1):
            raise ValueError(
                f"alignment.shape[0] ({aln.shape[0]}) must match "
                f"len(input_residues) ({len(res1)})"
            )
        if aln.shape[1] != len(res2):
            raise ValueError(
                f"alignment.shape[1] ({aln.shape[1]}) must match "
                f"len(target_residues) ({len(res2)})"
            )
        matches = {}
        aln_array = np.array(aln)
        indices = np.argwhere(aln_array == 1)
        for i, j in indices:
            if j + 1 not in constants.CDR_RESIDUES + constants.ADDITIONAL_GAPS:
                matches[str(res1[i])] = str(res2[j])
        LOGGER.debug(f"Calculated {len(matches)} matches")
        return matches

    def correct_gap_numbering(self, sub_aln: np.ndarray) -> np.ndarray:
        """Redistribute loop gaps to an alternating IMGT-style pattern."""
        new_aln = np.zeros_like(sub_aln)
        for i in range(min(sub_aln.shape)):
            pos = ((i + 1) // 2) * ((-1) ** i)
            new_aln[pos, pos] = 1
        gap_msg = (
            "Corrected gap numbering for sub-alignment "
            f"with shape {sub_aln.shape}"
        )
        LOGGER.debug(gap_msg)
        return new_aln

    def fix_aln(self, old_aln, idxs):
        """Expand an alignment onto IMGT positions using saved indices."""
        aln = np.zeros((old_aln.shape[0], 128))
        for i, idx in enumerate(idxs):
            aln[:, int(idx) - 1] = old_aln[:, i]
        expand_msg = (
            f"Expanded alignment from shape {old_aln.shape} to {aln.shape}"
        )
        LOGGER.debug(expand_msg)
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
                input_data.embeddings,
                species_embedding.embeddings,
                self.temperature,
            )
            aln = self.fix_aln(out.alignment, species_embedding.idxs)

            outputs[name] = types.SoftAlignOutput(
                alignment=aln, score=out.score, species=name, sim_matrix=None
            )
        LOGGER.info(f"Evaluated alignments against {len(outputs)} species")

        best_match = max(outputs, key=lambda k: outputs[k].score)
        LOGGER.info(
            f"Best match: {best_match}; score {outputs[best_match].score}"
        )
        aln = np.array(outputs[best_match].alignment, dtype=int)

        if correct_loops:
            for name, (startres, endres) in constants.IMGT_LOOPS.items():
                startres_idx = startres - 1
                loop_start = np.where(aln[:, startres - 1] == 1)[0]
                loop_end = np.where(aln[:, endres - 1] == 1)[0]
                if len(loop_start) == 0 or len(loop_end) == 0:
                    LOGGER.info(f"Loop {name} not found")
                    for arr, r in [(loop_start, startres), (loop_end, endres)]:
                        if len(arr) == 0:
                            LOGGER.info(f"Residue {r} not found")
                    LOGGER.info("Skipping...")
                    continue
                elif len(loop_start) > 1 or len(loop_end) > 1:
                    raise RuntimeError(f"Multiple start/end for loop {name}")
                loop_start = loop_start[0]
                loop_end = loop_end[0]
                sub_aln = aln[loop_start:loop_end, startres_idx:endres]
                LOGGER.info(f"Found {name} from {loop_start} to {loop_end}")
                LOGGER.info(f"IMGT positions from {startres} to {endres}")
                LOGGER.info(f"Sub-alignment shape: {sub_aln.shape}")
                aln[loop_start:loop_end, startres_idx:endres] = (
                    self.correct_gap_numbering(sub_aln)
                )

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

        return types.SoftAlignOutput(
            species=best_match, alignment=aln, score=0, sim_matrix=None
        )
