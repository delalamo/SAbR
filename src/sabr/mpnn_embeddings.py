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

        stdev = (
            np.ones(
                self.embeddings.shape, dtype=np.asarray(self.embeddings).dtype
            )
            if self.stdev is None
            else np.asarray(self.stdev)
        )

        def _repeat_rows(arr: np.ndarray, n_rows: int) -> np.ndarray:
            """Broadcast a single row across all residues."""
            return np.broadcast_to(arr, (n_rows, arr.shape[-1]))

        if stdev.ndim == 1:
            if stdev.shape[0] != constants.EMBED_DIM:
                raise ValueError(
                    "1D stdev must have length "
                    f"{constants.EMBED_DIM}; got {stdev.shape[0]} "
                    f"for {self.name}"
                )
            stdev = _repeat_rows(stdev[None, :], self.embeddings.shape[0])
        elif stdev.ndim == 2:
            if stdev.shape[1] != constants.EMBED_DIM:
                raise ValueError(
                    f"stdev.shape[1] ({stdev.shape[1]}) must match "
                    f"constants.EMBED_DIM ({constants.EMBED_DIM}) "
                    f"for {self.name}"
                )
            if stdev.shape[0] == 1:
                stdev = _repeat_rows(stdev, self.embeddings.shape[0])
            elif stdev.shape[0] > self.embeddings.shape[0]:
                stdev = stdev[: self.embeddings.shape[0], :]
            elif stdev.shape[0] < self.embeddings.shape[0]:
                raise ValueError(
                    "stdev rows fewer than embeddings rows are not allowed "
                    f"(stdev rows={stdev.shape[0]}, embeddings rows="
                    f"{self.embeddings.shape[0]} in {self.name})"
                )
        else:
            raise ValueError(
                f"stdev must be 1D or 2D array compatible with embeddings; "
                f"got ndim={stdev.ndim} for {self.name}"
            )

        object.__setattr__(self, "stdev", stdev)

        LOGGER.debug(
            f"Initialized MPNNEmbeddings for {self.name} "
            f"(shape={self.embeddings.shape})"
        )
