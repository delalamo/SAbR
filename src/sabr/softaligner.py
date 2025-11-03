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
        """
        Load SoftAlign model parameters from a package resource.

        Parameters
        ----------
        params_name : str, optional
            Filename of the serialized parameters within `params_path`.
        params_path : str, optional
            Package path resolvable by :func:`importlib.resources.files`.

        Returns
        -------
        dict
            The deserialized parameter dictionary.

        Raises
        ------
        FileNotFoundError
            If the target file cannot be found by `files(params_path) / name`.
        pickle.UnpicklingError
            If the file exists but cannot be unpickled.
        """
        path = files(params_path) / params_name
        params = pickle.load(open(path, "rb"))
        LOGGER.info(f"Loaded model parameters from {path}")
        return params

    def normalize(self, mp: types.MPNNEmbeddings) -> types.MPNNEmbeddings:
        """
        Return an embedding object with rows reordered by sorted int indices.

        This utility is useful when the embedding rows are stored in an
        arbitrary order and `mp.idxs` encodes the target order.

        Parameters
        ----------
        mp : types.MPNNEmbeddings
            Embedding container with fields ``name``, ``embeddings``, and
            ``idxs`` (string or int-like identifiers).

        Returns
        -------
        types.MPNNEmbeddings
            A new container with rows permuted to ascending ``idxs`` and
            ``idxs`` cast to ``int``.

        Notes
        -----
        This function does not modify `mp` in place; it returns a new object.
        """
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
        """
        Load species embeddings from a package ``.npz`` archive.

        The archive is expected to contain a pickled dict under key ``"arr_0"``
        mapping species name -> dict with keys:
        ``{"array": np.ndarray, "idxs": list[str|int]}``.

        Parameters
        ----------
        embeddings_name : str, optional
            Filename of the ``.npz`` archive within `embeddings_path`.
        embeddings_path : str, optional
            Package path resolvable by :func:`importlib.resources.files`.

        Returns
        -------
        list[types.MPNNEmbeddings]
            List of per-species embeddings.

        Raises
        ------
        RuntimeError
            If no embeddings are found in the archive.
        FileNotFoundError
            If the archive cannot be resolved.
        ValueError
            If the archive structure does not match the expected schema.
        """
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
        """
        Convert a binary alignment matrix into residue-to-residue matches.

        Parameters
        ----------
        aln : jnp.ndarray
            A binary matrix of shape ``(len(res1), len(res2))`` where ones
            indicate matched positions.
        res1 : list[str]
            Residue identifiers (e.g., PDB numbers, IMGT labels) for the rows.
        res2 : list[str]
            Residue identifiers for the columns.

        Returns
        -------
        dict[str, str]
            Mapping from ``res1`` identifiers to ``res2`` identifiers for the
            subset of columns that are **not** in
            ``constants.CDR_RESIDUES + constants.ADDITIONAL_GAPS``.

        Raises
        ------
        ValueError
            If `aln` is not 2-D or its shape mismatches `res1`/`res2`.

        Notes
        -----
        Each ``1`` entry in `aln` is treated as a match. Columns whose IMGT
        integer index (1-based) falls into the excluded positions are skipped.
        """
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
        """
        Re-map a loop sub-alignment to an IMGT-like alternating pattern.

        Given a binary sub-alignment array ``sub_aln`` with shape ``(N, M)``,
        construct a new alignment of the same shape that places ones along an
        alternating index pattern (0, +1, -1, +2, -2, ...). This is intended
        to regularize gap placement for loops with expected numbering schemes
        (e.g., CDR2).

        Parameters
        ----------
        sub_aln : np.ndarray
            Binary sub-alignment of shape ``(N, M)``.

        Returns
        -------
        np.ndarray
            A new binary array with ones placed along the alternating pattern
            and zeros elsewhere.

        Notes
        -----
        The mapping assumes ``min(N, M)`` effective aligned positions and
        does not validate biochemical plausibility.
        """
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
        """
        Fixes alignment to introduce gaps expected from chain-specific
        numbering idiosyncrasies. Saved indices are one-based indexed
        """
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
        """
        Align an input PDB chain to each species embedding and pick the best.

        Parameters
        ----------
        input_pdb : str
            Path to the PDB file to embed and align.
        input_chain : str
            Chain identifier within the PDB structure.
        correct_loops : bool, optional
            If ``True``, apply IMGT-inspired loop regularization (including
            special handling for the DE loop). Default is ``True``.

        Returns
        -------
        tuple[str, types.SoftAlignOutput]
            The best-matching species name and its corresponding alignment
            output. The output contains:
            ``alignment`` (np.ndarray), ``name`` (str),
            and ``score`` (float, set to zero).

        Raises
        ------
        RuntimeError
            If multiple starts/ends are detected for an IMGT loop region.
        ValueError
            If residue lookups or embeddings are inconsistent.

        Notes
        -----
        - The method computes an embedding for the input once, then aligns it
          against all available species embeddings.
        - Loop correction currently targets IMGT-defined loop windows from
          ``constants.IMGT_LOOPS`` and performs a DE-loop fix-up as a
          post-processing step.
        """
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
