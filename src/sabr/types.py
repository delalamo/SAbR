from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from jax import numpy as jnp

from sabr import constants


@dataclass(frozen=True)
class MPNNEmbeddings:
    """Per-residue embeddings for a single sequence/structure.

    Attributes
    ----------
    name : str
        A human-readable identifier for the embedding set (e.g., species name,
        source model, or ``"INPUT_PDB"``).
    embeddings : np.ndarray
        Array of shape ``(L, D)`` holding per-residue embeddings, where ``L``
        is the number of residues and ``D`` is the embedding dimension
        (often ``constants.EMBED_DIM``).
    idxs : list[str]
        A list of length ``L`` containing residue identifiers (e.g., PDB/IMGT
        numbering tokens) corresponding one-to-one with the rows of
        ``embeddings``.
    """

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


@dataclass(frozen=True)
class SoftAlignOutput:
    """Outputs produced by the SoftAlign alignment routine.

    Attributes
    ----------
    alignment : jnp.ndarray
        Binary alignment matrix. Typical shape is ``(L_in, K)`` after any
        expansion/mapping step (e.g., IMGT-width); exact width is model- and
        pipeline-dependent.
    score : float
        Scalar alignment score reported by the model (higher is better).
    sim_matrix : Optional[jnp.ndarray]
        Optional similarity matrix produced during alignment, commonly of shape
        ``(L_in, L_tgt)`` before any width remapping. May be ``None`` if not
        returned by the underlying model.
    species : Optional[str]
        Optional label for the target embedding set (e.g., species name) used
        when comparing against multiple candidates.
    """

    alignment: jnp.ndarray
    score: float
    sim_matrix: Optional[jnp.ndarray]
    species: Optional[str]
