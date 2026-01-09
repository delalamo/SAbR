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


def compute_overhang_penalty(
    alignment: np.ndarray,
    idxs: List[int],
    gap_open: float = constants.SW_GAP_OPEN,
    gap_extend: float = constants.SW_GAP_EXTEND,
) -> float:
    """Compute penalty for unaligned positions at N- and C-termini.

    Smith-Waterman local alignment doesn't penalize unaligned ends. This
    function adds a penalty for "overhangs" - reference positions that are
    skipped at the beginning or end of the alignment.

    The penalty is computed as if there were anchor positions at IMGT 0 and
    IMGT 129. If the alignment starts at IMGT position 2, that's treated as
    a gap opening (skipping position 1). If it starts at IMGT position 4,
    that's gap open + 2 * gap extend (skipping positions 1, 2, 3).

    Args:
        alignment: Alignment matrix [query_len, target_len]. Should be
            rounded/binarized so we can find first/last aligned columns.
        idxs: List of IMGT position integers for the reference columns.
        gap_open: Gap opening penalty (negative value).
        gap_extend: Gap extension penalty (negative value).

    Returns:
        Total overhang penalty (negative value, to be added to score).
    """
    # Find which reference columns have any alignment
    col_sums = alignment.sum(axis=0)
    aligned_cols = np.where(col_sums > 0.5)[0]

    if len(aligned_cols) == 0:
        # No alignment at all - shouldn't happen, but handle gracefully
        return 0.0

    first_aligned_col = aligned_cols[0]
    last_aligned_col = aligned_cols[-1]

    # Get IMGT positions for first and last aligned columns
    first_imgt = idxs[first_aligned_col]
    last_imgt = idxs[last_aligned_col]

    penalty = 0.0

    # N-terminal overhang: positions skipped before first alignment
    # Reference "position 0" is the anchor, so IMGT 1 is the first real position
    # If first_imgt = 1, no penalty. If first_imgt = 2, skip 1 position, etc.
    n_skipped_start = first_imgt - 1
    if n_skipped_start > 0:
        penalty += gap_open + (n_skipped_start - 1) * gap_extend

    # C-terminal overhang: positions skipped after last alignment
    # Reference "position 129" is the anchor, so IMGT 128 is the last position
    # If last_imgt = 128, no penalty. If last_imgt = 123, skip 5 positions.
    n_skipped_end = 128 - last_imgt
    if n_skipped_end > 0:
        penalty += gap_open + (n_skipped_end - 1) * gap_extend

    return penalty


def create_gap_penalty_for_reduced_reference(
    query_len: int,
    idxs: List[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Create gap penalty matrices with zero penalty at position 10.

    Gap penalties are set to zero only for IMGT position 10, which is
    commonly absent in antibody sequences. This allows a single free
    insertion/deletion at position 10 (gap open is zero, but gap extend
    remains penalized to prevent multiple free gaps).

    Args:
        query_len: Length of the query sequence.
        idxs: List of IMGT position integers for the reduced reference.

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

    # Position 10 (FR1 gap position commonly absent in antibodies)
    # Only zero the gap_open penalty, keep gap_extend to ensure only
    # one free insertion at this position
    for col_idx, pos in enumerate(idxs):
        if pos == 10:
            gap_open[:, col_idx] = 0.0
            # gap_extend remains penalized to limit to single insertion
            break  # Only one position 10 column

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
