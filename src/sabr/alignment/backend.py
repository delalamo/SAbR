#!/usr/bin/env python3
"""JAX/Haiku backend for alignment operations.

This module provides the AlignmentBackend class which encapsulates all
JAX and Haiku dependencies for running soft alignment between embedding sets.

Public interfaces accept and return numpy arrays only.
"""

import logging
from typing import List, Optional, Tuple

import haiku as hk
import jax
import numpy as np
from jax import numpy as jnp

from sabr import constants
from sabr.nn.end_to_end import END_TO_END

LOGGER = logging.getLogger(__name__)


def create_gap_penalty_for_reduced_reference(
    query_len: int,
    idxs: List[int],
    include_anchors: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create gap penalty matrices with position-dependent penalties.

    Gap open penalties are set to zero in CDR regions (IMGT 27-38, 56-65,
    105-117) and at position 10. This allows free gap openings in variable
    loop regions while still penalizing gap extensions to prevent excessive
    insertions.

    When include_anchors=True, the idxs list should include anchor positions
    0 and 129 at the start and end. The gap penalties between anchors and
    adjacent positions encode the overhang cost.

    Args:
        query_len: Length of the query sequence.
        idxs: List of IMGT position integers for the reduced reference.
            If include_anchors=True, should start with 0 and end with 129.
        include_anchors: If True, handle anchor positions 0 and 129.

    Returns:
        Tuple of (gap_extend_matrix, gap_open_matrix) with shape
        (query_len, target_len).
    """
    target_len = len(idxs)

    # Start with normal penalties (as numpy arrays)
    gap_extend = np.full(
        (query_len, target_len), constants.SW_GAP_EXTEND, dtype=np.float32
    )
    gap_open = np.full(
        (query_len, target_len), constants.SW_GAP_OPEN, dtype=np.float32
    )

    # Build set of CDR positions for fast lookup
    cdr_positions = set()
    for cdr_name in ["CDR1", "CDR2", "CDR3"]:
        cdr_positions.update(constants.IMGT_REGIONS[cdr_name])

    # Zero gap_open for CDR positions and position 10
    # Keep gap_extend penalized to limit insertions
    for col_idx, pos in enumerate(idxs):
        if pos in cdr_positions or pos == 10:
            gap_open[:, col_idx] = 0.0

    return gap_extend, gap_open


def _run_alignment_fn(
    input_embeddings: np.ndarray,
    target_embeddings: np.ndarray,
    temperature: float,
    gap_matrix: Optional[np.ndarray] = None,
    open_matrix: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Run soft alignment between embedding sets.

    This function runs inside hk.transform and uses the END_TO_END model
    to align query embeddings against reference embeddings.

    Args:
        input_embeddings: Query embeddings [N, embed_dim].
        target_embeddings: Reference embeddings [M, embed_dim].
        temperature: Alignment temperature (lower = more deterministic).
        gap_matrix: Optional position-dependent gap extension penalties [N, M].
        open_matrix: Optional position-dependent gap open penalties [N, M].

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
    )

    lens = jnp.array([input_embeddings.shape[0], target_embeddings.shape[0]])[
        None, :
    ]
    batched_input = jnp.array(input_embeddings[None, :])
    batched_target = jnp.array(target_embeddings[None, :])

    # Batch gap matrices if provided
    batched_gap_matrix = None
    batched_open_matrix = None
    if gap_matrix is not None and open_matrix is not None:
        batched_gap_matrix = jnp.array(gap_matrix[None, :])
        batched_open_matrix = jnp.array(open_matrix[None, :])

    alignment, sim_matrix, score = model.align(
        batched_input,
        batched_target,
        lens,
        temperature,
        gap_matrix=batched_gap_matrix,
        open_matrix=batched_open_matrix,
    )

    return alignment[0], sim_matrix[0], score[0]


class AlignmentBackend:
    """Backend for performing soft alignment between embedding sets.

    This class encapsulates the JAX/Haiku operations needed to run
    the SoftAlign alignment algorithm.

    Attributes:
        gap_extend: Gap extension penalty for Smith-Waterman.
        gap_open: Gap opening penalty for Smith-Waterman.
        key: JAX PRNG key for random operations.
    """

    def __init__(
        self,
        gap_extend: float = constants.SW_GAP_EXTEND,
        gap_open: float = constants.SW_GAP_OPEN,
        random_seed: int = 0,
    ) -> None:
        """Initialize the alignment backend.

        Args:
            gap_extend: Gap extension penalty.
            gap_open: Gap opening penalty.
            random_seed: Random seed for JAX PRNG.
        """
        self.gap_extend = gap_extend
        self.gap_open = gap_open
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
        temperature: float = constants.DEFAULT_TEMPERATURE,
        gap_matrix: Optional[np.ndarray] = None,
        open_matrix: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Align input embeddings against target embeddings.

        Args:
            input_embeddings: Query embeddings [N, embed_dim].
            target_embeddings: Reference embeddings [M, embed_dim].
            temperature: Alignment temperature parameter.
            gap_matrix: Optional position-dependent gap extension penalties.
            open_matrix: Optional position-dependent gap open penalties.

        Returns:
            Tuple of (alignment, similarity_matrix, score) as numpy.
        """
        self.key, subkey = jax.random.split(self.key)
        alignment, sim_matrix, score = self._transformed_fn.apply(
            self._params,
            subkey,
            input_embeddings,
            target_embeddings,
            temperature,
            gap_matrix,
            open_matrix,
        )

        return (
            np.asarray(alignment),
            np.asarray(sim_matrix),
            float(score),
        )
