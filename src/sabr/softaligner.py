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
        if not DEBUG:
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
        """
        Read the softalign model parameters from a .pkl file in the package.
        """
        root = files(params_path)
        path = root / params_name
        params = pickle.load(open(path, "rb"))
        LOGGER.info(f"Loaded model parameters from {path}")
        return params

    def normalize(self, mp: types.MPNNEmbeddings) -> types.MPNNEmbeddings:
        idxs_int = [int(x) for x in mp.idxs]
        order = np.argsort(np.asarray(idxs_int, dtype=np.int64))
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
        Read the embeddings from a .npz file in the package.
        """
        root = files(embeddings_path)
        path = root / embeddings_name
        out_embeddings = []
        with as_file(path) as path:
            data = np.load(path, allow_pickle=True)["arr_0"].item()
            for species, embeddings_dict in data.items():
                out_embeddings.append(
                    types.MPNNEmbeddings(
                        name=species,
                        embeddings=embeddings_dict.get("array"),
                        idxs=embeddings_dict.get("idxs"),
                    )
                )
        if len(out_embeddings) == 0:
            raise RuntimeError(f"Could not load embeddings from {path}")
        LOGGER.info(f"Loaded {len(out_embeddings)} embeddings from {path}")
        return out_embeddings

    def calc_matches(
        self,
        aln: jnp.ndarray,
        res1: List[str],
        res2: List[str],
    ) -> Dict[str, str]:
        """
        Calculate the residue matches from the alignment matrix.
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
        return matches

    def correct_gap_numbering(self, sub_aln: np.ndarray) -> np.ndarray:
        """
        Generate an N×M binary matrix following the IMGT CDR2-like pattern.

        Parameters
        ----------
        N : int
            Number of residues (loop length)
        M : int
            Reference region length

        Returns
        -------
        np.ndarray
            An (N, M) binary matrix where ones represent aligned residues.
        """
        new_aln = np.zeros_like(sub_aln)
        for i in range(min(sub_aln.shape)):
            pos = ((i + 1) // 2) * ((-1) ** i)
            new_aln[pos, pos] = 1

        return new_aln

    def __call__(
        self, input_pdb: str, input_chain: str, correct_loops: bool = True
    ) -> Tuple[str, types.SoftAlignOutput]:
        """
        Compute alignment of input array against all species embeddings
        """
        input_data = self.transformed_embed_fn.apply(
            self.model_params, self.key, input_pdb, input_chain
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
            old_aln = out.alignment
            aln = np.zeros((old_aln.shape[0], 128))
            for i, idx in enumerate(species_embedding.idxs):
                aln[:, int(idx) - 1] = old_aln[:, i]
            outputs[name] = types.SoftAlignOutput(
                alignment=aln, sim_matrix=out.sim_matrix, score=out.score
            )

        best_match = max(outputs, key=lambda k: outputs[k].score)
        LOGGER.info(
            f"Best match: {best_match}; score {outputs[best_match].score}"
        )
        aln = np.array(outputs[best_match].alignment, dtype=int)

        if correct_loops:
            # correct 72/73
            for name, (startres, endres) in constants.IMGT_LOOPS.items():
                # starting from flanking residues
                # startres = startres - 1
                # endres = endres + 1
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

            # if aln[:, 72].sum() > 0 and aln[:, 71].sum() == 0:
            #     LOGGER.info("Correcting 72/73 gap")
            #     aln[:, 71] = aln[:, 72]
            #     aln[:, 72] = 0

            # # Residue 10, heavy chains only
            # if aln[:, 9].sum() == 1 and aln[:, 10].sum() == 0:
            #     aln[:, 10] = aln[:, 9]
            #     aln[:, 9] = 0

        return best_match, aln
