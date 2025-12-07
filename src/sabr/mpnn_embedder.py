#!/usr/bin/env python3

import logging
import pickle
from importlib.resources import files
from pathlib import Path
from typing import Any, Dict

import haiku as hk
import jax
import numpy as np
from Bio import SeqIO

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
            MPNNEmbeddings for the specified chain, including sequence.
        """
        # Get embeddings from the model
        input_data = self.transformed_embed_fn.apply(
            self.model_params, self.key, input_pdb, input_chain, max_residues
        )

        # Extract sequence from PDB file
        try:
            sequence = self.fetch_sequence_from_pdb(input_pdb, input_chain)
        except Exception as e:
            LOGGER.warning(
                f"Could not extract sequence from PDB: {e}. "
                "Continuing without sequence."
            )
            sequence = None

        # Create new MPNNEmbeddings with sequence included
        input_data_with_seq = mpnn_embeddings.MPNNEmbeddings(
            name=input_data.name,
            embeddings=input_data.embeddings,
            idxs=input_data.idxs,
            stdev=input_data.stdev,
            sequence=sequence,
        )

        LOGGER.info(
            f"Computed embeddings for {input_pdb} chain {input_chain} "
            f"(length={input_data_with_seq.embeddings.shape[0]})"
        )
        return input_data_with_seq

    @staticmethod
    def fetch_sequence_from_pdb(pdb_file: str, chain: str) -> str:
        """
        Extract the sequence for a chain from a PDB file.

        Args:
            pdb_file: Path to the PDB file.
            chain: Chain identifier to extract.

        Returns:
            The sequence as a string, with X residues removed.

        Raises:
            ValueError: If the chain is not found in the PDB file.
        """
        for record in SeqIO.parse(pdb_file, "pdb-atom"):
            if record.id.endswith(chain):
                sequence = str(record.seq).replace("X", "")
                LOGGER.info(
                    f"Extracted sequence from {pdb_file} chain {chain} "
                    f"(length={len(sequence)})"
                )
                return sequence
        ids = [r.id for r in SeqIO.parse(pdb_file, "pdb-atom")]
        raise ValueError(
            f"Chain {chain} not found in {pdb_file} (contains {ids})"
        )

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
            sequence=embedding.sequence if embedding.sequence else "",
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

        # Load sequence if present, otherwise None
        sequence = None
        if "sequence" in data:
            seq_str = str(data["sequence"])
            sequence = seq_str if seq_str else None

        embedding = mpnn_embeddings.MPNNEmbeddings(
            name=name,
            embeddings=data["embeddings"],
            idxs=idxs,
            stdev=data["stdev"],
            sequence=sequence,
        )
        LOGGER.info(
            f"Loaded embeddings from {input_path} "
            f"(name={name}, length={len(idxs)})"
        )
        return embedding
