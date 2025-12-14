#!/usr/bin/env python3

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import haiku as hk
import jax
import numpy as np
from jax import numpy as jnp
from softalign import Input_MPNN

from sabr import constants, model, util

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class MPNNEmbeddings:
    """Per-residue embedding tensor and matching residue identifiers.

    Can be instantiated from either:
    1. A PDB file (via from_pdb function)
    2. An NPZ file (via from_npz function)
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

    def save(self, output_path: str) -> None:
        """
        Save MPNNEmbeddings to an NPZ file.

        Args:
            output_path: Path where the NPZ file will be saved.
        """
        output_path_obj = Path(output_path)
        np.savez(
            output_path_obj,
            name=self.name,
            embeddings=self.embeddings,
            idxs=np.array(self.idxs),
            stdev=self.stdev,
            sequence=self.sequence if self.sequence else "",
        )
        LOGGER.info(f"Saved embeddings to {output_path_obj}")


def _embed_pdb(
    pdbfile: str, chains: str, max_residues: int = 0
) -> MPNNEmbeddings:
    """Return MPNN embeddings for chains in pdbfile using SoftAlign.

    Args:
        pdbfile: Path to the PDB file.
        chains: Chain identifier(s) to embed.
        max_residues: Maximum number of residues to embed. If 0, embed all.

    Returns:
        MPNNEmbeddings for the specified chain.
    """
    LOGGER.info(f"Embedding PDB {pdbfile} chain {chains}")
    e2e_model = model.create_e2e_model()
    if len(chains) > 1:
        raise NotImplementedError(
            f"Only single chain embedding is supported. "
            f"Got {len(chains)} chains: '{chains}'. "
            f"Please specify a single chain identifier."
        )
    X1, mask1, chain1, res1, ids = Input_MPNN.get_inputs_mpnn(
        pdbfile, chain=chains
    )
    embeddings = e2e_model.MPNN(X1, mask1, chain1, res1)[0]
    if len(ids) != embeddings.shape[0]:
        raise ValueError(
            f"IDs length ({len(ids)}) does not match embeddings rows"
            f" ({embeddings.shape[0]})"
        )

    if max_residues > 0 and len(ids) > max_residues:
        LOGGER.info(
            f"Truncating embeddings from {len(ids)} to {max_residues} residues"
        )
        embeddings = embeddings[:max_residues]
        ids = ids[:max_residues]

    return MPNNEmbeddings(
        name="INPUT_PDB",
        embeddings=embeddings,
        idxs=ids,
        stdev=jnp.ones_like(embeddings),
    )


def from_pdb(
    pdb_file: str,
    chain: str,
    max_residues: int = 0,
    params_name: str = "CONT_SW_05_T_3_1",
    params_path: str = "softalign.models",
    random_seed: int = 0,
) -> MPNNEmbeddings:
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
    model_params = util.read_softalign_params(
        params_name=params_name, params_path=params_path
    )
    key = jax.random.PRNGKey(random_seed)
    transformed_embed_fn = hk.transform(_embed_pdb)

    input_data = transformed_embed_fn.apply(
        model_params, key, pdb_file, chain, max_residues
    )

    try:
        sequence = util.fetch_sequence_from_pdb(pdb_file, chain)
    except Exception as e:
        LOGGER.warning(
            f"Could not extract sequence from PDB: {e}. "
            "Continuing without sequence."
        )
        sequence = None

    result = MPNNEmbeddings(
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


def from_npz(npz_file: str) -> MPNNEmbeddings:
    """
    Create MPNNEmbeddings from an NPZ file.

    Args:
        npz_file: Path to the NPZ file to load.

    Returns:
        MPNNEmbeddings object loaded from the file.
    """
    input_path = Path(npz_file)
    data = np.load(input_path, allow_pickle=True)

    name = str(data["name"])
    idxs = [str(idx) for idx in data["idxs"]]

    sequence = None
    if "sequence" in data:
        seq_str = str(data["sequence"])
        sequence = seq_str if seq_str else None

    embedding = MPNNEmbeddings(
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
