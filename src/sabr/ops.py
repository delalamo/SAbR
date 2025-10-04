import logging

import numpy as np
from jax import numpy as jnp
from softalign import END_TO_END_MODELS, Input_MPNN

from sabr import constants, types

LOGGER = logging.getLogger(__name__)


def align_fn(
    input_array: np.ndarray, target_array: np.ndarray, temperature: float
) -> types.SoftAlignOutput:
    """
    Compute the alignment for the given input array using the SoftAlign model.
    """
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
    return types.SoftAlignOutput(
        alignment=alignment[0], sim_matrix=sim_matrix[0], score=float(score[0])
    )


def embed_fn(pdbfile: str, chains: str) -> types.MPNNEmbeddings:
    """
    Embed a PDB file using the softaligner
    """
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
    if len(chains) > 1:
        raise NotImplementedError("Only single chain embedding is supported")
    X1, mask1, chain1, res1, ids = Input_MPNN.get_inputs_mpnn(
        pdbfile, chain=chains
    )
    embeddings = e2e_model.MPNN(X1, mask1, chain1, res1)[0]
    if len(ids) != embeddings.shape[0]:
        LOGGER.info(
            (
                f"IDs length ({len(ids)}) does not match embeddings rows"
                f" ({embeddings.shape[0]})"
            )
        )
        for i, id_ in enumerate(ids):
            LOGGER.info(f"{i}: {id_}")
    return types.MPNNEmbeddings(
        name="INPUT_PDB", embeddings=embeddings, idxs=ids
    )
