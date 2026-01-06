#!/usr/bin/env python3
"""JAX/Haiku backend for alignment operations.

This module provides the AlignmentBackend class which encapsulates all
JAX and Haiku dependencies for running soft alignment between embedding sets.

Public interfaces accept and return numpy arrays only.
"""

import logging
from typing import Tuple

import haiku as hk
import jax
import numpy as np
from jax import numpy as jnp

from sabr import constants
from sabr.nn.end_to_end import END_TO_END

LOGGER = logging.getLogger(__name__)


def _run_alignment_fn(
    input_embeddings: np.ndarray,
    target_embeddings: np.ndarray,
    target_stdev: np.ndarray,
    temperature: float,
    penalize_start_gap: bool = False,
    penalize_end_gap: bool = False,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Run soft alignment between embedding sets.

    This function runs inside hk.transform and uses the END_TO_END model
    to align query embeddings against reference embeddings.

    Args:
        input_embeddings: Query embeddings [N, embed_dim].
        target_embeddings: Reference embeddings [M, embed_dim].
        target_stdev: Standard deviation for normalization [M, embed_dim].
        temperature: Alignment temperature (lower = more deterministic).
        penalize_start_gap: Penalize alignments starting after position 1
            of the reference (N-terminus anchoring).
        penalize_end_gap: Penalize alignments ending before the last
            position of the reference (C-terminus anchoring).

    Returns:
        Tuple of (alignment_matrix, similarity_matrix, alignment_score).
    """
    model = END_TO_END(
        constants.EMBED_DIM,
        constants.EMBED_DIM,
        constants.EMBED_DIM,
        constants.N_MPNN_LAYERS,
        constants.EMBED_DIM,
        affine=True,
        soft_max=False,
        dropout=0.0,
        augment_eps=0.0,
        penalize_start_gap=penalize_start_gap,
        penalize_end_gap=penalize_end_gap,
    )

    target_stdev_jax = jnp.array(target_stdev)
    target_normalized = target_embeddings / target_stdev_jax

    lens = jnp.array([input_embeddings.shape[0], target_embeddings.shape[0]])[
        None, :
    ]
    batched_input = jnp.array(input_embeddings[None, :])
    batched_target = jnp.array(target_normalized[None, :])

    alignment, sim_matrix, score = model.align(
        batched_input, batched_target, lens, temperature
    )

    return alignment[0], sim_matrix[0], score[0]


class AlignmentBackend:
    """Backend for performing soft alignment between embedding sets.

    This class encapsulates the JAX/Haiku operations needed to run
    the SoftAlign alignment algorithm.

    Attributes:
        gap_extend: Gap extension penalty for Smith-Waterman.
        gap_open: Gap opening penalty for Smith-Waterman.
        penalize_start_gap: Penalize alignments starting after position 1.
        penalize_end_gap: Penalize alignments ending before the last position.
        key: JAX PRNG key for random operations.
    """

    def __init__(
        self,
        gap_extend: float = constants.SW_GAP_EXTEND,
        gap_open: float = constants.SW_GAP_OPEN,
        random_seed: int = 0,
        penalize_start_gap: bool = True,
        penalize_end_gap: bool = True,
    ) -> None:
        """Initialize the alignment backend.

        Args:
            gap_extend: Gap extension penalty.
            gap_open: Gap opening penalty.
            random_seed: Random seed for JAX PRNG.
            penalize_start_gap: Penalize alignments starting after position 1
                of the reference (N-terminus anchoring). Default True.
            penalize_end_gap: Penalize alignments ending before the last
                position of the reference (C-terminus anchoring). Default True.
        """
        self.gap_extend = gap_extend
        self.gap_open = gap_open
        self.penalize_start_gap = penalize_start_gap
        self.penalize_end_gap = penalize_end_gap
        self.key = jax.random.PRNGKey(random_seed)
        self._params = {
            "~": {
                "gap": jnp.array([self.gap_extend]),
                "open": jnp.array([self.gap_open]),
            }
        }
        self._transformed_fn = hk.transform(_run_alignment_fn)
        LOGGER.info("Initialized AlignmentBackend")

    def align(
        self,
        input_embeddings: np.ndarray,
        target_embeddings: np.ndarray,
        target_stdev: np.ndarray,
        temperature: float = constants.DEFAULT_TEMPERATURE,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Align input embeddings against target embeddings.

        Args:
            input_embeddings: Query embeddings [N, embed_dim].
            target_embeddings: Reference embeddings [M, embed_dim].
            target_stdev: Standard deviation for normalization [M, embed_dim].
            temperature: Alignment temperature parameter.

        Returns:
            Tuple of (alignment, similarity_matrix, score) as numpy.
        """
        self.key, subkey = jax.random.split(self.key)
        alignment, sim_matrix, score = self._transformed_fn.apply(
            self._params,
            subkey,
            input_embeddings,
            target_embeddings,
            target_stdev,
            temperature,
            self.penalize_start_gap,
            self.penalize_end_gap,
        )

        return (
            np.asarray(alignment),
            np.asarray(sim_matrix),
            float(score),
        )
