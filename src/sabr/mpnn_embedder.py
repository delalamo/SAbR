#!/usr/bin/env python3

import logging
import pickle
from importlib.resources import files
from typing import Any, Dict

import haiku as hk
import jax

from sabr import mpnn_embeddings, ops

LOGGER = logging.getLogger(__name__)


class MPNNEmbedder:
    """Generate MPNN embeddings for a query chain."""

    def __init__(
        self,
        params_name: str = "CONT_SW_05_T_3_1",
        params_path: str = "softalign.models",
        random_seed: int = 0,
    ) -> None:
        """
        Initialize the MPNNEmbedder by loading model parameters.

        Args:
            params_name: Name of the model parameters file.
            params_path: Package path containing the parameters file.
            random_seed: Random seed for JAX.
        """
        self.model_params = self._read_softalign_params(
            params_name=params_name, params_path=params_path
        )
        self.key = jax.random.PRNGKey(random_seed)
        self.transformed_embed_fn = hk.transform(ops.embed_fn)

    def _read_softalign_params(
        self,
        params_name: str = "CONT_SW_05_T_3_1",
        params_path: str = "softalign.models",
    ) -> Dict[str, Any]:
        """Load SoftAlign parameters from package resources."""
        path = files(params_path) / params_name
        params = pickle.load(open(path, "rb"))
        LOGGER.info(f"Loaded model parameters from {path}")
        return params

    def embed(
        self,
        input_pdb: str,
        input_chain: str,
        max_residues: int = 0,
    ) -> mpnn_embeddings.MPNNEmbeddings:
        """
        Generate MPNN embeddings for the specified chain.

        Args:
            input_pdb: Path to input PDB file.
            input_chain: Chain identifier to embed.
            max_residues: Maximum residues to embed. If 0, embed all.

        Returns:
            MPNNEmbeddings for the specified chain.
        """
        input_data = self.transformed_embed_fn.apply(
            self.model_params, self.key, input_pdb, input_chain, max_residues
        )
        LOGGER.info(
            f"Computed embeddings for {input_pdb} chain {input_chain} "
            f"(length={input_data.embeddings.shape[0]})"
        )
        return input_data
