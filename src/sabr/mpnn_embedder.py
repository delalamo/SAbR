#!/usr/bin/env python3

import logging
import pickle
from importlib.resources import files
from pathlib import Path
from typing import Any, Dict

import haiku as hk
import jax
import numpy as np

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

    @staticmethod
    def save_to_npz(
        embedding: mpnn_embeddings.MPNNEmbeddings,
        output_path: str,
    ) -> None:
        """
        Save MPNNEmbeddings to an NPZ file.

        Args:
            embedding: The MPNNEmbeddings object to save.
            output_path: Path where the NPZ file will be saved.
        """
        output_path = Path(output_path)
        np.savez(
            output_path,
            name=embedding.name,
            embeddings=embedding.embeddings,
            idxs=np.array(embedding.idxs),
            stdev=embedding.stdev,
        )
        LOGGER.info(f"Saved embeddings to {output_path}")

    @staticmethod
    def load_from_npz(input_path: str) -> mpnn_embeddings.MPNNEmbeddings:
        """
        Load MPNNEmbeddings from an NPZ file.

        Args:
            input_path: Path to the NPZ file to load.

        Returns:
            MPNNEmbeddings object loaded from the file.
        """
        input_path = Path(input_path)
        data = np.load(input_path, allow_pickle=True)

        # Convert numpy scalar to string if needed
        name = str(data["name"])

        # Convert idxs back to list of strings
        idxs = [str(idx) for idx in data["idxs"]]

        embedding = mpnn_embeddings.MPNNEmbeddings(
            name=name,
            embeddings=data["embeddings"],
            idxs=idxs,
            stdev=data["stdev"],
        )
        LOGGER.info(
            f"Loaded embeddings from {input_path} "
            f"(name={name}, length={len(idxs)})"
        )
        return embedding
