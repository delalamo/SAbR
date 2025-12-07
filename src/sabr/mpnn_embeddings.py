#!/usr/bin/env python3

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from sabr import constants

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class MPNNEmbeddings:
    """Per-residue embedding tensor and matching residue identifiers."""

    name: str
    embeddings: np.ndarray
    idxs: List[str]
    stdev: Optional[np.ndarray] = None
    sequence: Optional[str] = None

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

        if self.stdev is None:
            self.stdev = np.ones_like(self.embeddings)

        LOGGER.debug(
            f"Initialized MPNNEmbeddings for {self.name} "
            f"(shape={self.embeddings.shape})"
        )
