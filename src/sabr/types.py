from dataclasses import dataclass
from typing import List

import numpy as np
from jax import numpy as jnp


@dataclass(frozen=True)
class MPNNEmbeddings:
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


@dataclass(frozen=True)
class SoftAlignOutput:
    alignment: jnp.ndarray
    sim_matrix: jnp.ndarray
    score: float
