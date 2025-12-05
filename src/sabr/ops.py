#!/usr/bin/env python3

import logging

from jax import numpy as jnp
from softalign import END_TO_END_MODELS, Input_MPNN

from sabr import constants, mpnn_embeddings, softalign_output

LOGGER = logging.getLogger(__name__)


def align_fn(
    input: mpnn_embeddings.MPNNEmbeddings,
    target: mpnn_embeddings.MPNNEmbeddings,
    temperature: float = 10**-4,
) -> softalign_output.SoftAlignOutput:
    """Align two embedding sets with the SoftAlign model and return result."""
    input_array = input.embeddings
    target_array = target.embeddings
    target_stdev = jnp.array(target.stdev)
    target_array = target_array / target_stdev

    # we want to look at two schemes
    # one is where we divide each embedding by stdev
    # the other is where we scale the similarity matrix by stdev

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


def embed_fn(
    pdbfile: str, chains: str, max_residues: int = 0
) -> mpnn_embeddings.MPNNEmbeddings:
    """Return MPNN embeddings for ``chains`` in ``pdbfile`` using SoftAlign.

    Args:
        pdbfile: Path to the PDB file.
        chains: Chain identifier(s) to embed.
        max_residues: Maximum number of residues to embed. If 0, embed all.

    Returns:
        MPNNEmbeddings for the specified chain.
    """
    LOGGER.info(f"Embedding PDB {pdbfile} chain {chains}")
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
        raise ValueError(
            (
                f"IDs length ({len(ids)}) does not match embeddings rows"
                f" ({embeddings.shape[0]})"
            )
        )

    # Truncate to max_residues if specified
    if max_residues > 0 and len(ids) > max_residues:
        LOGGER.info(
            f"Truncating embeddings from {len(ids)} to {max_residues} residues"
        )
        embeddings = embeddings[:max_residues]
        ids = ids[:max_residues]

    return mpnn_embeddings.MPNNEmbeddings(
        name="INPUT_PDB",
        embeddings=embeddings,
        idxs=ids,
        stdev=jnp.ones_like(embeddings),
    )
