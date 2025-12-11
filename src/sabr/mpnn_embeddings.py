#!/usr/bin/env python3

import logging
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from sabr import constants

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class MPNNEmbeddings:
    """Per-residue embedding tensor and matching residue identifiers.

    Can be instantiated from either:
    1. A PDB file (via from_pdb classmethod)
    2. An NPZ file (via from_npz classmethod)
    3. Direct construction with embeddings data
    """

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

    @classmethod
    def from_pdb(
        cls,
        pdb_file: str,
        chain: str,
        max_residues: int = 0,
        params_name: str = "CONT_SW_05_T_3_1",
        params_path: str = "softalign.models",
        random_seed: int = 0,
    ) -> "MPNNEmbeddings":
        """
        Create MPNNEmbeddings from a PDB file.

        Args:
            pdb_file: Path to input PDB file.
            chain: Chain identifier to embed.
            max_residues: Maximum residues to embed. If 0, embed all.
            params_name: Name of the model parameters file.
            params_path: Package path containing the parameters file.
            random_seed: Random seed for JAX.

        Returns:
            MPNNEmbeddings for the specified chain.
        """
        # Lazy imports to avoid JAX dependency for NPZ loading
        import haiku as hk
        import jax

        from sabr import ops

        # Load model parameters
        model_params = cls._read_softalign_params(
            params_name=params_name, params_path=params_path
        )
        key = jax.random.PRNGKey(random_seed)
        transformed_embed_fn = hk.transform(ops.embed_fn)

        # Get embeddings from the model
        input_data = transformed_embed_fn.apply(
            model_params, key, pdb_file, chain, max_residues
        )

        # Extract sequence from PDB file
        try:
            sequence = cls._fetch_sequence_from_pdb(pdb_file, chain)
        except Exception as e:
            LOGGER.warning(
                f"Could not extract sequence from PDB: {e}. "
                "Continuing without sequence."
            )
            sequence = None

        # Create new MPNNEmbeddings with sequence included
        result = cls(
            name=input_data.name,
            embeddings=input_data.embeddings,
            idxs=input_data.idxs,
            stdev=input_data.stdev,
            sequence=sequence,
        )

        LOGGER.info(
            f"Computed embeddings for {pdb_file} chain {chain} "
            f"(length={result.embeddings.shape[0]})"
        )
        return result

    @classmethod
    def from_npz(cls, npz_file: str) -> "MPNNEmbeddings":
        """
        Create MPNNEmbeddings from an NPZ file.

        Args:
            npz_file: Path to the NPZ file to load.

        Returns:
            MPNNEmbeddings object loaded from the file.
        """
        input_path = Path(npz_file)
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

        embedding = cls(
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

    def save(self, output_path: str) -> None:
        """
        Save this MPNNEmbeddings to an NPZ file.

        Args:
            output_path: Path where the NPZ file will be saved.
        """
        output_path = Path(output_path)
        np.savez(
            output_path,
            name=self.name,
            embeddings=self.embeddings,
            idxs=np.array(self.idxs),
            stdev=self.stdev,
            sequence=self.sequence if self.sequence else "",
        )
        LOGGER.info(f"Saved embeddings to {output_path}")

    @staticmethod
    def _read_softalign_params(
        params_name: str = "CONT_SW_05_T_3_1",
        params_path: str = "softalign.models",
    ) -> Dict[str, Any]:
        """Load SoftAlign parameters from package resources."""
        from sabr import util

        path = files(params_path) / params_name
        with open(path, "rb") as f:
            params = util.JaxBackwardsCompatUnpickler(f).load()
        LOGGER.info(f"Loaded model parameters from {path}")
        return params

    @staticmethod
    def _fetch_sequence_from_pdb(pdb_file: str, chain: str) -> str:
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
        from Bio import SeqIO

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
