import logging
import pickle
from dataclasses import dataclass
from importlib.resources import as_file, files
from typing import Any, Dict, List, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from softalign import END_TO_END_MODELS as ete

from sabr import constants

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReferenceEmbeddings:
    species: str
    embeddings: np.ndarray
    idxs: List[str]

    def __post_init__(self) -> None:
        if self.embeddings.shape[0] != len(self.idxs):
            raise ValueError(
                f"embeddings.shape[0] ({self.embeddings.shape[0]}) must match "
                f"len(idxs) ({len(self.idxs)}). "
                f"Error raised for {self.species}"
            )


@dataclass(frozen=True)
class SoftAlignOutput:
    alignment: jnp.ndarray
    sim_matrix: jnp.ndarray
    score: float


def align_fn(
    input_array: np.ndarray, target_array: np.ndarray, temperature: float
) -> SoftAlignOutput:
    """
    Compute the alignment for the given input array using the SoftAlign model.
    """
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
    e2e_model = ete.END_TO_END(
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
    alignment, sim_matrix, score = e2e_model.apply(
        batched_input, batched_target, lens, temperature
    )
    return SoftAlignOutput(
        alignment=alignment[0], sim_matrix=sim_matrix[0], score=float(score[0])
    )


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
            self.all_embeddings: List[ReferenceEmbeddings] = (
                self.read_embeddings(
                    embeddings_name=embeddings_name,
                    embeddings_path=embeddings_path,
                )
            )
            self.model_params: Dict[str, Any] = self.read_softalign_params(
                params_name=params_name, params_path=params_path
            )
        self.temperature: float = temperature
        self.key = jax.random.PRNGKey(random_seed)
        self.transformed_align_fn = hk.transform(align_fn)

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

    def read_embeddings(
        self,
        embeddings_name: str = "embeddings.npz",
        embeddings_path: str = "sabr.assets",
    ) -> List[ReferenceEmbeddings]:
        """
        Read the embeddings from a .npz file in the package.
        """
        root = files(embeddings_path)
        path = root / embeddings_name
        out_embeddings = []
        with as_file(path) as path:
            data = np.load(path, allow_pickle=True)
            for species, embeddings_dict in data.items():
                out_embeddings.append(
                    ReferenceEmbeddings(
                        species=species,
                        embeddings=embeddings_dict.get("array"),
                        idxs=embeddings_dict.get("idxs"),
                    )
                )
        if len(out_embeddings) == 0:
            raise RuntimeError(f"Could not load embeddings from {path}")
        LOGGER.info(f"Loaded {len(out_embeddings)} embeddings from {path}")
        return out_embeddings

    def __call__(self, input_array: np.ndarray) -> Tuple[str, SoftAlignOutput]:
        """
        Compute alignment of input array against all species embeddings
        """
        outputs = {}
        for species_embedding in self.all_embeddings:
            outputs[species_embedding.species] = (
                self.transformed_align_fn.apply(
                    self.model_params,
                    self.key,
                    input_array,
                    species_embedding.embeddings,
                    self.temperature,
                )
            )
        max_species = max(outputs, key=lambda k: outputs[k].score)
        LOGGER.info(
            f"Best match: {max_species}; score {outputs[max_species].score}"
        )
        return max_species, outputs[max_species]
