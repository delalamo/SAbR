#!/usr/bin/env python3
"""SoftAlign-based antibody sequence alignment module.

This module provides the SoftAligner class which aligns query antibody
embeddings against unified reference embeddings to generate IMGT-compatible
alignments.

Key components:
- SoftAligner: Main class for running alignments

The alignment process includes:
1. Embedding comparison against unified reference
2. Deterministic corrections for CDR loops, DE loop, FR1, and C-terminus
3. Expansion to full 128-position IMGT alignment matrix
4. Chain type detection from DE loop occupancy
"""

import logging
from importlib.resources import as_file, files
from typing import List

import numpy as np

from sabr.alignment import corrections
from sabr.alignment.backend import AlignmentBackend
from sabr.alignment.output import SoftAlignOutput
from sabr.core import constants
from sabr.core.util import detect_chain_type

LOGGER = logging.getLogger(__name__)


class SoftAligner:
    """Align a query embedding against unified reference embeddings."""

    def __init__(
        self,
        embeddings_name: str = "embeddings.npz",
        embeddings_path: str = "sabr.assets",
        temperature: float = constants.DEFAULT_TEMPERATURE,
        random_seed: int = 0,
    ) -> None:
        """
        Initialize the SoftAligner by loading reference embeddings and backend.

        Args:
            embeddings_name: Name of the reference embeddings file.
            embeddings_path: Package path containing the embeddings file.
            temperature: Alignment temperature parameter.
            random_seed: Random seed for reproducibility.
        """
        self.unified_embedding = self.read_embeddings(
            embeddings_name=embeddings_name,
            embeddings_path=embeddings_path,
        )
        self.temperature = temperature
        self._backend = AlignmentBackend(random_seed=random_seed)

    def read_embeddings(
        self,
        embeddings_name: str = "embeddings.npz",
        embeddings_path: str = "sabr.assets",
    ):
        """Load packaged reference embeddings."""
        # Import here to avoid circular dependency
        from sabr.embeddings.mpnn import MPNNEmbeddings

        path = files(embeddings_path) / embeddings_name
        with as_file(path) as p:
            data = np.load(p, allow_pickle=True)
            embedding = MPNNEmbeddings(
                name=str(data["name"]),
                embeddings=data["array"],
                stdev=data["stdev"],
                idxs=list(data["idxs"]),
            )
        LOGGER.info(f"Loaded embeddings from {path}")
        return embedding

    def fix_aln(self, old_aln: np.ndarray, idxs: List[int]) -> np.ndarray:
        """Expand an alignment onto IMGT positions using saved indices."""
        aln = np.zeros(
            (old_aln.shape[0], constants.IMGT_MAX_POSITION), dtype=old_aln.dtype
        )
        aln[:, np.asarray(idxs, dtype=int) - 1] = old_aln

        return aln

    def __call__(
        self,
        input_data,
        deterministic_loop_renumbering: bool = True,
    ) -> SoftAlignOutput:
        """Align input embeddings against the unified reference embedding.

        Args:
            input_data: Pre-computed MPNN embeddings for the query chain.
            deterministic_loop_renumbering: Whether to apply deterministic
                renumbering corrections for CDR loops, FR1, FR3, and
                C-terminus. Default is True.

        Returns:
            SoftAlignOutput with the best alignment.
        """
        LOGGER.info(
            f"Aligning embeddings with length={input_data.embeddings.shape[0]}"
        )

        alignment, sim_matrix, score = self._backend.align(
            input_embeddings=input_data.embeddings,
            target_embeddings=self.unified_embedding.embeddings,
            target_stdev=self.unified_embedding.stdev,
            temperature=self.temperature,
        )

        aln = self.fix_aln(alignment, self.unified_embedding.idxs)
        aln = np.round(aln).astype(int)

        gap_indices = input_data.gap_indices
        if gap_indices:
            LOGGER.info(
                f"Detected {len(gap_indices)} structural gap(s) in input"
            )

        if deterministic_loop_renumbering:
            aln, detected_chain_type = (
                corrections.apply_deterministic_corrections(
                    aln, gap_indices=gap_indices
                )
            )
        else:
            detected_chain_type = detect_chain_type(aln)
            LOGGER.info(f"Detected chain type: {detected_chain_type}")

        return SoftAlignOutput(
            chain_type=detected_chain_type,
            alignment=aln,
            score=score,
            sim_matrix=sim_matrix,
            idxs1=input_data.idxs,
            idxs2=[str(x) for x in range(1, constants.IMGT_MAX_POSITION + 1)],
        )
