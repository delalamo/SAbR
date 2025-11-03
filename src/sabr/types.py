#!/usr/bin/env python3

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from jax import numpy as jnp

from sabr import constants

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class MPNNEmbeddings:
    """Per-residue embedding tensor and matching residue identifiers."""

    name: str
    embeddings: np.ndarray
    idxs: List[str]

    def __post_init__(self) -> None:
        if self.embeddings.shape[0] != len(self.idxs):
            raise ValueError(
                f"embeddings.shape[0] ({self.embeddings.shape[0]}) must match "
                f"len(idxs) ({len(self.idxs)}). "
                f"Error raised for {self.name}"
            )
        if self.embeddings.shape[1] != constants.EMBED_DIM:
            raise ValueError(
                f"embeddings.shape[1] ({self.embeddings.shape[1]}) must match "
                f"constants.EMBED_DIM ({constants.EMBED_DIM}). "
                f"Error raised for {self.name}"
            )
        LOGGER.debug(
            f"Initialized MPNNEmbeddings for {self.name} "
            f"(shape={self.embeddings.shape})"
        )


@dataclass(frozen=True)
class SoftAlignOutput:
    """Alignment matrix plus bookkeeping returned by SoftAlign."""

    alignment: jnp.ndarray
    score: float
    sim_matrix: Optional[jnp.ndarray]
    species: Optional[str]
    idxs1: List[str]
    idxs2: List[str]

    def __post_init__(self) -> None:
        if self.alignment.shape[0] != len(self.idxs1):
            raise ValueError(
                f"embeddings.shape[0] ({self.alignment.shape[0]}) must match "
                f"len(idxs1) ({len(self.idxs1)}). "
            )
        if self.alignment.shape[1] != len(self.idxs2):
            raise ValueError(
                f"embeddings.shape[1] ({self.alignment.shape[1]}) must match "
                f"len(idxs2) ({len(self.idxs2)}). "
            )
        LOGGER.debug(
            "Created SoftAlignOutput for "
            f"species={self.species}, alignment_shape="
            f"{getattr(self.alignment, 'shape', None)}, score={self.score}"
        )
