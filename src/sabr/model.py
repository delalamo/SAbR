#!/usr/bin/env python3
"""Model factory functions for creating SoftAlign models."""

from typing import Any

from softalign import END_TO_END_MODELS

from sabr import constants


def create_e2e_model() -> Any:
    """Create and return an END_TO_END model with standard parameters.

    Returns:
        An END_TO_END model instance configured with:
        - embed_dim: 64 (from constants.EMBED_DIM)
        - n_layers: 3 (from constants.N_MPNN_LAYERS)
        - affine: True
        - soft_max: False
        - dropout: 0.0
        - augment_eps: 0.0
    """
    return END_TO_END_MODELS.END_TO_END(
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
