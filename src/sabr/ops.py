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
    Compute a pairwise alignment using the SoftAlign end-to-end model.

    This function wraps the SoftAlign model's ``align`` method. It validates
    inputs, adds a batch dimension expected by the model, and returns a
    :class:`sabr.types.SoftAlignOutput` object containing the single best
    alignment for the provided input/target embeddings.

    Parameters
    ----------
    input_array : np.ndarray
        2-D array with shape ``(L_in, D)`` containing per-residue embeddings
        of the input sequence/structure. ``D`` must equal
        ``constants.EMBED_DIM``.
    target_array : np.ndarray
        2-D array with shape ``(L_tgt, D)`` containing per-residue embeddings
        of the target sequence/structure. ``D`` must equal
        ``constants.EMBED_DIM``.
    temperature : float
        Temperature parameter passed to the model's alignment routine.

    Returns
    -------
    sabr.types.SoftAlignOutput
        A container with:
        - ``alignment``: binary matrix ``(L_in, K)`` (model-specific width),
        - ``sim_matrix``: similarity matrix for the pair,
        - ``score``: scalar alignment score (higher is better).

    Raises
    ------
    ValueError
        If ``input_array`` or ``target_array`` is not 2-D, or if the final
        dimension does not equal ``constants.EMBED_DIM``.

    Notes
    -----
    - A fresh model instance is constructed on every call. If you plan to call
      this function many times, consider instantiating and reusing the model in
      your own code to avoid repeated setup costs.
    - The model expects batched inputs; this wrapper adds a batch dimension of
      size 1 and removes it from the outputs.
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
        alignment=alignment[0],
        sim_matrix=sim_matrix[0],
        score=float(score[0]),
        species=None,
    )


def embed_fn(pdbfile: str, chains: str) -> types.MPNNEmbeddings:
    """
    Generate per-residue MPNN embeddings for a single PDB chain.

    This function parses a PDB file, extracts the specified chain, computes
    MPNN embeddings using the SoftAlign model's encoder, and returns them in a
    :class:`sabr.types.MPNNEmbeddings` container. The container includes the
    embedding array and the corresponding residue identifiers as produced by
    :func:`softalign.Input_MPNN.get_inputs_mpnn`.

    Parameters
    ----------
    pdbfile : str
        Path to the input PDB file to embed.
    chains : str
        Single-character chain identifier (e.g., ``"A"``, ``"H"``, ``"L"``).
        Multi-chain inputs are not supported by this wrapper.

    Returns
    -------
    sabr.types.MPNNEmbeddings
        An object with:
        - ``name``: fixed to ``"INPUT_PDB"``,
        - ``embeddings``: array of shape ``(L, D)`` where ``D`` is
          ``constants.EMBED_DIM``,
        - ``idxs``: sequence of residue IDs (as returned by
          :func:`Input_MPNN.get_inputs_mpnn`), typically strings that identify
          positions in the original structure.

    Raises
    ------
    NotImplementedError
        If ``chains`` contains more than one character (i.e., multi-chain
        embedding is requested).
    ValueError
        May be raised by underlying parsing/featurization if the PDB/chain
        cannot be processed.

    Notes
    -----
    - :func:`softalign.Input_MPNN.get_inputs_mpnn` returns a tuple
      ``(X, mask, chain_idx, res_one_letter, ids)``. Only ``ids`` are used
      here for metadata; they should align with the first axis of the returned
      embedding array.
    - If the number of ``ids`` does not match the number of embedding rows,
      a diagnostic listing is logged at ``INFO`` level.
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
