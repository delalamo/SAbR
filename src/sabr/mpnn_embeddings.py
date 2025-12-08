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

        n_rows = self.embeddings.shape[0]
        processed_stdev = self._process_stdev(self.stdev, n_rows)
        object.__setattr__(self, "stdev", processed_stdev)

        LOGGER.debug(
            f"Initialized MPNNEmbeddings for {self.name} "
            f"(shape={self.embeddings.shape})"
        )

    def _process_stdev(
        self, stdev: Optional[np.ndarray], n_rows: int
    ) -> np.ndarray:
        """Process and validate stdev, returning a properly shaped array."""
        if stdev is None:
            return np.ones_like(self.embeddings)

        stdev = np.asarray(stdev)

        if stdev.ndim == 1:
            if stdev.shape[0] != constants.EMBED_DIM:
                raise ValueError(
                    f"1D stdev must have length {constants.EMBED_DIM}, "
                    f"got {stdev.shape[0]}"
                )
            return np.broadcast_to(stdev, (n_rows, constants.EMBED_DIM)).copy()

        if stdev.ndim == 2:
            if stdev.shape[1] != constants.EMBED_DIM:
                raise ValueError(
                    f"stdev.shape[1] ({stdev.shape[1]}) must match "
                    f"constants.EMBED_DIM ({constants.EMBED_DIM})"
                )
            if stdev.shape[0] == 1:
                return np.broadcast_to(
                    stdev, (n_rows, constants.EMBED_DIM)
                ).copy()
            if stdev.shape[0] < n_rows:
                raise ValueError(
                    f"stdev rows fewer than embeddings rows are not allowed: "
                    f"stdev rows={stdev.shape[0]}, embeddings rows={n_rows}"
                )
            if stdev.shape[0] > n_rows:
                return stdev[:n_rows, :].copy()
            return stdev

        raise ValueError(
            f"stdev must be 1D or 2D array compatible with embeddings, "
            f"got ndim={stdev.ndim}"
        )
