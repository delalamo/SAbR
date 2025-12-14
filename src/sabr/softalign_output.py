#!/usr/bin/env python3

import logging
from dataclasses import dataclass
from typing import List, Optional, Union

import jax
import numpy as np

LOGGER = logging.getLogger(__name__)

# Type alias for arrays that can be either NumPy or JAX arrays
ArrayLike = Union[np.ndarray, jax.Array]


@dataclass(frozen=True)
class SoftAlignOutput:
    """Alignment matrix plus bookkeeping returned by SoftAlign."""

    alignment: ArrayLike
    score: float
    sim_matrix: Optional[ArrayLike]
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
