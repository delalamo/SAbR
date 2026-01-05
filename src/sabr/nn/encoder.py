#!/usr/bin/env python3
"""MPNN (Message Passing Neural Network) encoder for protein structures.

This module provides the ENC class which encodes protein backbone structures
into per-residue embeddings using a graph neural network architecture.

The encoder processes protein structures by:
1. Extracting geometric features from backbone atoms (N, CA, C, CB)
2. Building a k-nearest neighbor graph based on CA distances
3. Computing radial basis function (RBF) features for atom-atom distances
4. Running message passing layers to produce per-residue embeddings

Adapted from SoftAlign/ProteinMPNN implementations.
"""

import functools

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

Gelu = functools.partial(jax.nn.gelu, approximate=False)


class SafeKey:
    """Safety wrapper for PRNG keys to prevent reuse."""

    def __init__(self, key):
        self._key = key
        self._used = False

    def _assert_not_used(self):
        if self._used:
            raise RuntimeError("Random key has been used previously.")

    def get(self):
        self._assert_not_used()
        self._used = True
        return self._key

    def split(self, num_keys: int = 2):
        self._assert_not_used()
        self._used = True
        new_keys = jax.random.split(self._key, num_keys)
        return jax.tree_util.tree_map(SafeKey, tuple(new_keys))

    def duplicate(self, num_keys: int = 2):
        self._assert_not_used()
        self._used = True
        return tuple(SafeKey(self._key) for _ in range(num_keys))


def gather_edges(edges, neighbor_idx):
    """Gather edge features at neighbor indices.

    Args:
        edges: Edge features [B, N, N, C].
        neighbor_idx: Neighbor indices [B, N, K].

    Returns:
        Neighbor edge features [B, N, K, C].
    """
    neighbors = jnp.tile(
        jnp.expand_dims(neighbor_idx, -1), [1, 1, 1, edges.shape[-1]]
    )
    edge_features = jnp.take_along_axis(edges, neighbors, 2)
    return edge_features


def gather_nodes(nodes, neighbor_idx):
    """Gather node features at neighbor indices.

    Args:
        nodes: Node features [B, N, C].
        neighbor_idx: Neighbor indices [B, N, K].

    Returns:
        Neighbor node features [B, N, K, C].
    """
    neighbors_flat = neighbor_idx.reshape([neighbor_idx.shape[0], -1])
    neighbors_flat = jnp.tile(
        jnp.expand_dims(neighbors_flat, -1), [1, 1, nodes.shape[2]]
    )
    neighbor_features = jnp.take_along_axis(nodes, neighbors_flat, 1)
    neighbor_features = neighbor_features.reshape(
        list(neighbor_idx.shape[:3]) + [-1]
    )
    return neighbor_features


def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    """Concatenate node features with gathered neighbor features."""
    h_nodes = gather_nodes(h_nodes, E_idx)
    h_nn = jnp.concatenate([h_neighbors, h_nodes], -1)
    return h_nn


class PositionalEncodings(hk.Module):
    """Relative positional encodings for sequence positions."""

    def __init__(self, num_embeddings: int, max_relative_feature: int = 32):
        super(PositionalEncodings, self).__init__()
        self.num_embeddings = num_embeddings
        self.max_relative_feature = max_relative_feature
        self.linear = hk.Linear(num_embeddings, name="embedding_linear")

    def __call__(self, offset, mask):
        d = jnp.clip(
            offset + self.max_relative_feature, 0, 2 * self.max_relative_feature
        ) * mask + (1 - mask) * (2 * self.max_relative_feature + 1)
        d_onehot = jax.nn.one_hot(d, 2 * self.max_relative_feature + 1 + 1)
        E = self.linear(jax.lax.convert_element_type(d_onehot, jnp.float32))
        return E


class ProteinFeatures(hk.Module):
    """Extract geometric features from protein backbone coordinates."""

    def __init__(
        self,
        edge_features: int,
        node_features: int,
        num_positional_embeddings: int = 16,
        num_rbf: int = 16,
        top_k: int = 30,
        augment_eps: float = 0.0,
        num_chain_embeddings: int = 16,
    ):
        super(ProteinFeatures, self).__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings

        self.embeddings = PositionalEncodings(num_positional_embeddings)
        # edge_in = num_positional_embeddings + num_rbf * 25 (for reference)
        self.edge_embedding = hk.Linear(
            edge_features, with_bias=False, name="edge_embedding"
        )
        self.norm_edges = hk.LayerNorm(
            -1, create_scale=True, create_offset=True, name="norm_edges"
        )

    def _dist(self, X, mask, eps=1e-6):
        """Compute pairwise distances and k-nearest neighbors."""
        mask_2D = jnp.expand_dims(mask, 1) * jnp.expand_dims(mask, 2)
        dX = jnp.expand_dims(X, 1) - jnp.expand_dims(X, 2)
        D = mask_2D * jnp.sqrt(jnp.sum(dX**2, 3) + eps)
        D_max = jnp.max(D, -1, keepdims=True)
        D_adjust = D + (1.0 - mask_2D) * D_max
        D_neighbors, E_idx = jax.lax.approx_min_k(
            D_adjust, np.minimum(self.top_k, X.shape[1]), reduction_dimension=-1
        )
        return D_neighbors, E_idx

    def _rbf(self, D):
        """Compute radial basis function features."""
        D_min, D_max, D_count = 2.0, 22.0, self.num_rbf
        D_mu = jnp.linspace(D_min, D_max, D_count)
        D_mu = D_mu.reshape([1, 1, 1, -1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = jnp.expand_dims(D, -1)
        RBF = jnp.exp(-(((D_expand - D_mu) / D_sigma) ** 2))
        return RBF

    def _get_rbf(self, A, B, E_idx):
        """Compute RBF features between two atom types."""
        D_A_B = jnp.sqrt(
            jnp.sum((A[:, :, None, :] - B[:, None, :, :]) ** 2, -1) + 1e-6
        )
        D_A_B_neighbors = gather_edges(D_A_B[:, :, :, None], E_idx)[:, :, :, 0]
        RBF_A_B = self._rbf(D_A_B_neighbors)
        return RBF_A_B

    def __call__(self, X, mask, residue_idx, chain_labels):
        """Extract edge features from backbone coordinates.

        Args:
            X: Backbone coordinates [B, N, 4, 3] (N, CA, C, O/CB).
            mask: Valid residue mask [B, N].
            residue_idx: Residue indices [B, N].
            chain_labels: Chain identifiers [B, N].

        Returns:
            Tuple of (edge_features, neighbor_indices).
        """
        if self.augment_eps > 0:
            use_key = hk.next_rng_key()
            X = X + self.augment_eps * jax.random.normal(use_key, X.shape)

        # Compute virtual CB position
        b = X[:, :, 1, :] - X[:, :, 0, :]
        c = X[:, :, 2, :] - X[:, :, 1, :]
        a = jnp.cross(b, c)
        Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + X[:, :, 1, :]

        Ca = X[:, :, 1, :]
        N = X[:, :, 0, :]
        C = X[:, :, 2, :]
        bb_O = X[:, :, 3, :]  # Backbone oxygen

        D_neighbors, E_idx = self._dist(Ca, mask)

        # Compute all pairwise RBF features (25 combinations)
        RBF_all = []
        RBF_all.append(self._rbf(D_neighbors))  # Ca-Ca
        RBF_all.append(self._get_rbf(N, N, E_idx))
        RBF_all.append(self._get_rbf(C, C, E_idx))
        RBF_all.append(self._get_rbf(bb_O, bb_O, E_idx))
        RBF_all.append(self._get_rbf(Cb, Cb, E_idx))
        RBF_all.append(self._get_rbf(Ca, N, E_idx))
        RBF_all.append(self._get_rbf(Ca, C, E_idx))
        RBF_all.append(self._get_rbf(Ca, bb_O, E_idx))
        RBF_all.append(self._get_rbf(Ca, Cb, E_idx))
        RBF_all.append(self._get_rbf(N, C, E_idx))
        RBF_all.append(self._get_rbf(N, bb_O, E_idx))
        RBF_all.append(self._get_rbf(N, Cb, E_idx))
        RBF_all.append(self._get_rbf(Cb, C, E_idx))
        RBF_all.append(self._get_rbf(Cb, bb_O, E_idx))
        RBF_all.append(self._get_rbf(bb_O, C, E_idx))
        RBF_all.append(self._get_rbf(N, Ca, E_idx))
        RBF_all.append(self._get_rbf(C, Ca, E_idx))
        RBF_all.append(self._get_rbf(bb_O, Ca, E_idx))
        RBF_all.append(self._get_rbf(Cb, Ca, E_idx))
        RBF_all.append(self._get_rbf(C, N, E_idx))
        RBF_all.append(self._get_rbf(bb_O, N, E_idx))
        RBF_all.append(self._get_rbf(Cb, N, E_idx))
        RBF_all.append(self._get_rbf(C, Cb, E_idx))
        RBF_all.append(self._get_rbf(bb_O, Cb, E_idx))
        RBF_all.append(self._get_rbf(C, bb_O, E_idx))
        RBF_all = jnp.concatenate(tuple(RBF_all), axis=-1)

        # Positional encodings
        offset = residue_idx[:, :, None] - residue_idx[:, None, :]
        offset = gather_edges(offset[:, :, :, None], E_idx)[:, :, :, 0]

        d_chains = (chain_labels[:, :, None] - chain_labels[:, None, :]) == 0
        d_chains = jax.lax.convert_element_type(d_chains, jnp.int64)
        E_chains = gather_edges(d_chains[:, :, :, None], E_idx)[:, :, :, 0]
        E_positional = self.embeddings(
            jax.lax.convert_element_type(offset, jnp.int64), E_chains
        )

        E = jnp.concatenate((E_positional, RBF_all), -1)
        E = self.edge_embedding(E)
        E = self.norm_edges(E)
        return E, E_idx


class PositionWiseFeedForward(hk.Module):
    """Position-wise feed-forward network."""

    def __init__(self, num_hidden: int, num_ff: int, name: str = None):
        super(PositionWiseFeedForward, self).__init__()
        self.W_in = hk.Linear(num_ff, with_bias=True, name=name + "_W_in")
        self.W_out = hk.Linear(num_hidden, with_bias=True, name=name + "_W_out")
        self.act = Gelu

    def __call__(self, h_V):
        h = self.act(self.W_in(h_V), approximate=False)
        h = self.W_out(h)
        return h


class DropoutCust(hk.Module):
    """Custom dropout with safe key management."""

    def __init__(self, rate: float):
        super().__init__()
        self.rate = rate
        self.safe_key = SafeKey(hk.next_rng_key())

    def __call__(self, x):
        self.safe_key, use_key = self.safe_key.split()
        return hk.dropout(use_key.get(), self.rate, x)


class EncLayer(hk.Module):
    """Encoder layer with message passing and feed-forward."""

    def __init__(
        self,
        num_hidden: int,
        num_in: int,
        dropout: float = 0.1,
        num_heads: int = None,
        scale: int = 30,
        name: str = None,
    ):
        super(EncLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale

        self.safe_key = SafeKey(hk.next_rng_key())

        self.dropout1 = DropoutCust(dropout)
        self.dropout2 = DropoutCust(dropout)
        self.dropout3 = DropoutCust(dropout)
        self.norm1 = hk.LayerNorm(
            -1, create_scale=True, create_offset=True, name=name + "_norm1"
        )
        self.norm2 = hk.LayerNorm(
            -1, create_scale=True, create_offset=True, name=name + "_norm2"
        )
        self.norm3 = hk.LayerNorm(
            -1, create_scale=True, create_offset=True, name=name + "_norm3"
        )

        self.W1 = hk.Linear(num_hidden, with_bias=True, name=name + "_W1")
        self.W2 = hk.Linear(num_hidden, with_bias=True, name=name + "_W2")
        self.W3 = hk.Linear(num_hidden, with_bias=True, name=name + "_W3")
        self.W11 = hk.Linear(num_hidden, with_bias=True, name=name + "_W11")
        self.W12 = hk.Linear(num_hidden, with_bias=True, name=name + "_W12")
        self.W13 = hk.Linear(num_hidden, with_bias=True, name=name + "_W13")
        self.act = Gelu
        self.dense = PositionWiseFeedForward(
            num_hidden, num_hidden * 4, name=name + "_dense"
        )

    def __call__(self, h_V, h_E, E_idx, mask_V=None, mask_attend=None):
        """Run encoder layer.

        Args:
            h_V: Node features [B, N, H].
            h_E: Edge features [B, N, K, H].
            E_idx: Edge indices [B, N, K].
            mask_V: Node mask [B, N].
            mask_attend: Attention mask [B, N, K].

        Returns:
            Updated (h_V, h_E).
        """
        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = jnp.tile(
            jnp.expand_dims(h_V, -2), [1, 1, h_EV.shape[-2], 1]
        )
        h_EV = jnp.concatenate([h_V_expand, h_EV], -1)

        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = jnp.expand_dims(mask_attend, -1) * h_message
        dh = jnp.sum(h_message, -2) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))

        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))
        if mask_V is not None:
            mask_V = jnp.expand_dims(mask_V, -1)
            h_V = mask_V * h_V

        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = jnp.tile(
            jnp.expand_dims(h_V, -2), [1, 1, h_EV.shape[-2], 1]
        )
        h_EV = jnp.concatenate([h_V_expand, h_EV], -1)
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        h_E = self.norm3(h_E + self.dropout3(h_message))
        return h_V, h_E


class ENC:
    """MPNN Encoder for protein structure embeddings.

    This class encodes protein backbone coordinates into per-residue
    embedding vectors using a message passing neural network architecture.

    The encoder:
    1. Extracts geometric features from backbone atoms
    2. Builds a k-nearest neighbor graph
    3. Runs multiple message passing layers
    4. Returns per-residue embeddings
    """

    def __init__(
        self,
        node_features: int,
        edge_features: int,
        hidden_dim: int,
        num_encoder_layers: int = 1,
        k_neighbors: int = 64,
        augment_eps: float = 0.05,
        dropout: float = 0.1,
    ):
        """Initialize the MPNN encoder.

        Args:
            node_features: Dimension of node features.
            edge_features: Dimension of edge features.
            hidden_dim: Hidden dimension for layers.
            num_encoder_layers: Number of encoder layers.
            k_neighbors: Number of nearest neighbors in graph.
            augment_eps: Coordinate noise augmentation.
            dropout: Dropout rate.
        """
        super(ENC, self).__init__()
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim

        self.features = ProteinFeatures(
            node_features,
            edge_features,
            top_k=k_neighbors,
            augment_eps=augment_eps,
        )

        self.W_e = hk.Linear(hidden_dim, with_bias=True, name="W_e")
        self.encoder_layers = [
            EncLayer(
                hidden_dim, hidden_dim * 2, dropout=dropout, name="enc" + str(i)
            )
            for i in range(num_encoder_layers)
        ]

    def __call__(self, X, mask, residue_idx, chain_encoding_all):
        """Encode protein structure to per-residue embeddings.

        Args:
            X: Backbone coordinates [B, N, 4, 3].
            mask: Valid residue mask [B, N].
            residue_idx: Residue indices [B, N].
            chain_encoding_all: Chain identifiers [B, N].

        Returns:
            Per-residue embeddings [B, N, hidden_dim].
        """
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = jnp.zeros((E.shape[0], E.shape[1], E.shape[-1]))
        h_E = self.W_e(E)

        mask_attend = gather_nodes(jnp.expand_dims(mask, -1), E_idx).squeeze(-1)
        mask_attend = jnp.expand_dims(mask, -1) * mask_attend

        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        return h_V
