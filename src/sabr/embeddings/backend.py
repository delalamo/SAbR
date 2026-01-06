#!/usr/bin/env python3
"""JAX/Haiku backend for embedding generation.

This module provides the EmbeddingBackend class which encapsulates all
JAX and Haiku dependencies for generating MPNN embeddings from protein
structure coordinates.

Public interfaces accept and return numpy arrays only.
"""

import logging
from importlib.resources import files
from typing import Any, Dict, List, Tuple

import haiku as hk
import jax
import numpy as np
from jax import numpy as jnp

from sabr import constants
from sabr.nn.end_to_end import END_TO_END


def create_gap_penalty_for_reduced_reference(
    query_len: int,
    idxs: List[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Create gap penalty matrices with zeros where IMGT positions have jumps.

    When variable positions are removed from the reference, there are jumps
    in the IMGT position sequence. Gap penalties should be zero at these
    boundaries to avoid penalizing the natural gaps where CDRs were removed.

    Example: idxs = [1, 2, ..., 26, 39, 40, ...]
             Gap from 26→39 (CDR1 removed) should have zero penalty

    Args:
        query_len: Length of the query sequence.
        idxs: List of IMGT position integers for the reduced reference.

    Returns:
        Tuple of (gap_extend_matrix, gap_open_matrix) with shape
        (query_len, target_len). Positions where IMGT numbering has
        jumps have zero penalty.
    """
    target_len = len(idxs)

    # Start with normal penalties (as numpy arrays)
    gap_extend = np.full(
        (query_len, target_len), constants.SW_GAP_EXTEND, dtype=np.float32
    )
    gap_open = np.full(
        (query_len, target_len), constants.SW_GAP_OPEN, dtype=np.float32
    )

    # Find columns where IMGT positions have jumps (CDRs were removed)
    for i in range(1, target_len):
        if idxs[i] - idxs[i - 1] > 1:  # Jump in IMGT numbering
            # Set zero penalty for this column (crossing removed CDR region)
            gap_extend[:, i] = 0.0
            gap_open[:, i] = 0.0

    return gap_extend, gap_open


LOGGER = logging.getLogger(__name__)


def _unflatten_dict(d: Dict[str, Any], sep: str = ".") -> Dict[str, Any]:
    """Unflatten a dictionary with separator-joined keys.

    Args:
        d: Flat dictionary with keys like "a.b.c".
        sep: Separator used in keys.

    Returns:
        Nested dictionary structure.
    """
    result = {}
    for key, value in d.items():
        parts = key.split(sep)
        current = result
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
    return result


def _convert_numpy_to_jax(obj: Any) -> Any:
    """Recursively convert numpy arrays in nested structures to JAX arrays.

    Traverses dictionaries and converts numpy ndarrays to jnp.arrays
    while preserving all other values unchanged.

    Args:
        obj: Object that may contain numpy arrays (dict, ndarray, or other).

    Returns:
        Same structure with numpy arrays replaced by JAX arrays.
    """
    if isinstance(obj, dict):
        return {k: _convert_numpy_to_jax(v) for k, v in obj.items()}
    elif isinstance(obj, np.ndarray):
        return jnp.array(obj)
    else:
        return obj


def load_mpnn_params(
    params_name: str = "mpnn_encoder",
    params_path: str = "sabr.assets",
) -> Dict[str, Any]:
    """Load MPNN encoder parameters from package resources.

    Args:
        params_name: Name of the parameters file (without extension).
        params_path: Package path containing the parameters file.

    Returns:
        Dictionary containing the model parameters as JAX arrays.
    """
    package_files = files(params_path)
    npz_path = package_files / f"{params_name}.npz"

    with open(npz_path, "rb") as f:
        data = dict(np.load(f, allow_pickle=False))

    params = _unflatten_dict(data)
    params = _convert_numpy_to_jax(params)
    LOGGER.info(f"Loaded MPNN parameters from {npz_path}")
    return params


def _create_e2e_model() -> END_TO_END:
    """Create an END_TO_END model with standard SAbR configuration.

    Returns:
        An END_TO_END model instance configured for antibody embedding
        and alignment with 64-dimensional embeddings and 3 MPNN layers.
    """
    return END_TO_END(
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


def _compute_embeddings_fn(
    coords: np.ndarray,
    mask: np.ndarray,
    chain_ids: np.ndarray,
    residue_indices: np.ndarray,
) -> np.ndarray:
    """Compute MPNN embeddings from structure coordinates.

    This function runs inside hk.transform and uses the END_TO_END model
    to generate per-residue embeddings from backbone coordinates.

    Args:
        coords: Backbone coordinates [1, N, 4, 3] (N, CA, C, CB).
        mask: Binary mask for valid residues [1, N].
        chain_ids: Chain identifiers [1, N].
        residue_indices: Sequential residue indices [1, N].

    Returns:
        Embeddings array with shape [1, N, embed_dim].
    """
    model = _create_e2e_model()
    return model.MPNN(coords, mask, chain_ids, residue_indices)


class EmbeddingBackend:
    """Backend for generating MPNN embeddings from protein structures.

    This class encapsulates the JAX/Haiku operations needed to run
    the MPNN encoder on protein structure coordinates.

    Attributes:
        params: The loaded model parameters.
        key: JAX PRNG key for random operations.
    """

    def __init__(
        self,
        params_name: str = "mpnn_encoder",
        params_path: str = "sabr.assets",
        random_seed: int = 0,
    ) -> None:
        """Initialize the embedding backend.

        Args:
            params_name: Name of the parameters file.
            params_path: Package path containing the parameters.
            random_seed: Random seed for JAX PRNG.
        """
        self.params = load_mpnn_params(params_name, params_path)
        self.key = jax.random.PRNGKey(random_seed)
        self._transformed_fn = hk.transform(_compute_embeddings_fn)
        LOGGER.info("Initialized EmbeddingBackend")

    def compute_embeddings(
        self,
        coords: np.ndarray,
        mask: np.ndarray,
        chain_ids: np.ndarray,
        residue_indices: np.ndarray,
    ) -> np.ndarray:
        """Compute MPNN embeddings for protein structure coordinates.

        Args:
            coords: Backbone coordinates [1, N, 4, 3] (N, CA, C, CB).
            mask: Binary mask for valid residues [1, N].
            chain_ids: Chain identifiers [1, N].
            residue_indices: Sequential residue indices [1, N].

        Returns:
            Embeddings array [N, embed_dim] as numpy array.
        """
        result = self._transformed_fn.apply(
            self.params,
            self.key,
            coords,
            mask,
            chain_ids,
            residue_indices,
        )
        # Convert from JAX array to numpy and remove batch dimension
        return np.asarray(result[0])
