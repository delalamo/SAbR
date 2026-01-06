#!/usr/bin/env python3
"""End-to-end neural alignment model.

This module provides the END_TO_END class which combines the MPNN encoder
with a differentiable Smith-Waterman alignment algorithm to perform
structure-based sequence alignment.

The model can:
1. Generate per-residue embeddings from backbone coordinates
2. Align embeddings using soft Smith-Waterman with affine gap penalties
3. Return alignment matrices, similarity scores, and alignment scores
"""

import haiku as hk
import jax
import jax.numpy as jnp

from sabr.nn import encoder, smith_waterman


class END_TO_END:
    """End-to-end model for structure embedding and alignment.

    This class combines an MPNN encoder for generating per-residue embeddings
    with a differentiable Smith-Waterman algorithm for soft alignment.

    The model supports both linear and affine gap penalties, controlled by
    the `affine` parameter. Affine gaps distinguish between gap opening
    (expensive) and gap extension (cheaper).
    """

    def __init__(
        self,
        node_features: int,
        edge_features: int,
        hidden_dim: int,
        num_encoder_layers: int = 3,
        k_neighbors: int = 64,
        augment_eps: float = 0.05,
        dropout: float = 0.0,
        affine: bool = False,
        soft_max: bool = False,
        penalize_start_gap: bool = False,
        penalize_end_gap: bool = False,
    ):
        """Initialize the end-to-end model.

        Args:
            node_features: Dimension of node features.
            edge_features: Dimension of edge features.
            hidden_dim: Hidden dimension for encoder layers.
            num_encoder_layers: Number of MPNN encoder layers.
            k_neighbors: Number of nearest neighbors in graph.
            augment_eps: Coordinate noise augmentation.
            dropout: Dropout rate for encoder layers.
            affine: Use affine gap penalties (gap open + extend).
            soft_max: Use softmax alignment instead of Smith-Waterman.
            penalize_start_gap: Penalize alignments starting after position 1
                of the reference (N-terminus anchoring).
            penalize_end_gap: Penalize alignments ending before the last
                position of the reference (C-terminus anchoring).
        """
        super(END_TO_END, self).__init__()

        self.MPNN = encoder.ENC(
            node_features,
            edge_features,
            hidden_dim,
            num_encoder_layers,
            k_neighbors,
            augment_eps,
            dropout,
        )
        self.affine = affine
        if affine:
            self.my_sw_func = jax.jit(smith_waterman.sw_affine(batch=True))
        else:
            self.my_sw_func = jax.jit(smith_waterman.sw(batch=True))
        self.siz = node_features
        self.soft_max = soft_max
        self.penalize_start_gap = penalize_start_gap
        self.penalize_end_gap = penalize_end_gap

    def align(self, h_V1, h_V2, lens, t):
        """Align two sets of embeddings.

        Args:
            h_V1: First embedding set [B, N, D].
            h_V2: Second embedding set [B, M, D].
            lens: Lengths of sequences [B, 2].
            t: Temperature for soft alignment.

        Returns:
            Tuple of (soft_alignment, similarity_matrix, scores).
        """
        gap = hk.get_parameter(
            "gap", shape=[1], init=hk.initializers.RandomNormal(0.1, -1)
        )
        if self.affine:
            popen = hk.get_parameter(
                "open", shape=[1], init=hk.initializers.RandomNormal(0.1, -3)
            )

        sim_matrix = jnp.einsum("nia,nja->nij", h_V1, h_V2)

        if not self.soft_max:
            if self.affine:
                scores, soft_aln = self.my_sw_func(
                    sim_matrix,
                    lens,
                    gap[0],
                    popen[0],
                    t,
                    self.penalize_start_gap,
                    self.penalize_end_gap,
                )
            else:
                scores, soft_aln = self.my_sw_func(sim_matrix, lens, gap[0], t)
            return soft_aln, sim_matrix, scores
        else:
            soft_aln = jax.vmap(_soft_max_single, in_axes=(0, 0, None))(
                sim_matrix, lens, t
            )
            scores = _max_ali(soft_aln)
            return soft_aln, sim_matrix, scores

    def __call__(self, x1, x2, lens, t):
        """Run end-to-end embedding and alignment.

        Args:
            x1: First structure tuple (X, mask, res, chain).
            x2: Second structure tuple (X, mask, res, chain).
            lens: Lengths of sequences [B, 2].
            t: Temperature for soft alignment.

        Returns:
            Tuple of (soft_alignment, similarity_matrix, scores).
        """
        X1, mask1, res1, ch1 = x1
        X2, mask2, res2, ch2 = x2
        h_V1 = self.MPNN(X1, mask1, res1, ch1)
        h_V2 = self.MPNN(X2, mask2, res2, ch2)

        return self.align(h_V1, h_V2, lens, t)


def _soft_max_single(sim_matrix, lens, t):
    """Compute softmax alignment on a single similarity matrix."""
    max_len_1, max_len_2 = sim_matrix.shape

    mask_1 = jnp.arange(max_len_1) < lens[0]
    mask_2 = jnp.arange(max_len_2) < lens[1]

    mask = mask_1[:, None] * mask_2[None, :]
    masked_sim_matrix = jnp.where(mask, sim_matrix, -100000)

    soft_aln = jnp.sqrt(
        10**-9
        + jax.nn.softmax(t**-1 * masked_sim_matrix, axis=-1)
        * jax.nn.softmax(t**-1 * masked_sim_matrix, axis=-2)
    )
    return jnp.where(mask, soft_aln, 0)


@jax.jit
def _max_ali(aln):
    """Compute maximum alignment score."""
    return jnp.sum(jnp.max(aln, axis=-1), axis=-1)
