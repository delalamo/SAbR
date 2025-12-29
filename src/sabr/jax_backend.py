#!/usr/bin/env python3
"""JAX/Haiku backend for neural network operations.

This module encapsulates all JAX and Haiku dependencies, providing
numpy-based interfaces for the rest of the codebase. All JAX/Haiku
imports are contained within this module.

Key components:
- EmbeddingBackend: Generates MPNN embeddings from protein structures
- AlignmentBackend: Performs soft alignment between embedding sets

Public interfaces accept and return numpy arrays only.
"""

import logging
from importlib.resources import files
from typing import Any, Dict, Tuple

import haiku as hk
import jax
import numpy as np
from jax import numpy as jnp
from softalign import END_TO_END_MODELS

from sabr import constants

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


def create_e2e_model() -> END_TO_END_MODELS.END_TO_END:
    """Create an END_TO_END model with standard SAbR configuration.

    Returns:
        An END_TO_END model instance configured for antibody embedding
        and alignment with 64-dimensional embeddings and 3 MPNN layers.
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


# Module-level functions for Haiku transforms
# These must be defined at module level for hk.transform to work correctly


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
    model = create_e2e_model()
    return model.MPNN(coords, mask, chain_ids, residue_indices)


def _run_alignment_fn(
    input_embeddings: np.ndarray,
    target_embeddings: np.ndarray,
    target_stdev: np.ndarray,
    temperature: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Run soft alignment between embedding sets.

    This function runs inside hk.transform and uses the END_TO_END model
    to align query embeddings against reference embeddings.

    Args:
        input_embeddings: Query embeddings [N, embed_dim].
        target_embeddings: Reference embeddings [M, embed_dim].
        target_stdev: Standard deviation for normalization [M, embed_dim].
        temperature: Alignment temperature (lower = more deterministic).

    Returns:
        Tuple of (alignment_matrix, similarity_matrix, alignment_score).
    """
    model = create_e2e_model()

    # Normalize target by stdev
    target_stdev_jax = jnp.array(target_stdev)
    target_normalized = target_embeddings / target_stdev_jax

    # Prepare batched inputs (model expects batch dimension)
    lens = jnp.array([input_embeddings.shape[0], target_embeddings.shape[0]])[
        None, :
    ]
    batched_input = jnp.array(input_embeddings[None, :])
    batched_target = jnp.array(target_normalized[None, :])

    # Run alignment
    alignment, sim_matrix, score = model.align(
        batched_input, batched_target, lens, temperature
    )

    # Remove batch dimension from outputs
    return alignment[0], sim_matrix[0], score[0]


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


class AlignmentBackend:
    """Backend for performing soft alignment between embedding sets.

    This class encapsulates the JAX/Haiku operations needed to run
    the SoftAlign alignment algorithm.

    Attributes:
        gap_extend: Gap extension penalty for Smith-Waterman.
        gap_open: Gap opening penalty for Smith-Waterman.
        key: JAX PRNG key for random operations.
    """

    def __init__(
        self,
        gap_extend: float = constants.SW_GAP_EXTEND,
        gap_open: float = constants.SW_GAP_OPEN,
        random_seed: int = 0,
    ) -> None:
        """Initialize the alignment backend.

        Args:
            gap_extend: Gap extension penalty.
            gap_open: Gap opening penalty.
            random_seed: Random seed for JAX PRNG.
        """
        self.gap_extend = gap_extend
        self.gap_open = gap_open
        self.key = jax.random.PRNGKey(random_seed)

        # Create params dict with alignment penalties
        # These are the only parameters needed for the align operation
        self._params = {
            "~": {
                "gap": jnp.array([self.gap_extend]),
                "open": jnp.array([self.gap_open]),
            }
        }
        self._transformed_fn = hk.transform(_run_alignment_fn)
        LOGGER.info("Initialized AlignmentBackend")

    def align(
        self,
        input_embeddings: np.ndarray,
        target_embeddings: np.ndarray,
        target_stdev: np.ndarray,
        temperature: float = constants.DEFAULT_TEMPERATURE,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Align input embeddings against target embeddings.

        Args:
            input_embeddings: Query embeddings [N, embed_dim].
            target_embeddings: Reference embeddings [M, embed_dim].
            target_stdev: Standard deviation for normalization [M, embed_dim].
            temperature: Alignment temperature parameter.

        Returns:
            Tuple of (alignment, similarity_matrix, score) as numpy.
        """
        alignment, sim_matrix, score = self._transformed_fn.apply(
            self._params,
            self.key,
            input_embeddings,
            target_embeddings,
            target_stdev,
            temperature,
        )

        return (
            np.asarray(alignment),
            np.asarray(sim_matrix),
            float(score),
        )
