#!/usr/bin/env python3
"""SoftAlign-based antibody sequence alignment module.

This module provides the SoftAligner class which aligns query antibody
embeddings against unified reference embeddings to generate IMGT-compatible
alignments.

Key components:
- SoftAligner: Main class for running alignments
- SoftAlignOutput: Dataclass for alignment results

The alignment process includes:
1. Embedding comparison against unified reference
2. Deterministic corrections for CDR loops, DE loop, FR1, and C-terminus
3. Expansion to full 128-position IMGT alignment matrix
4. Chain type detection from DE loop occupancy
"""

import logging
from dataclasses import dataclass
from importlib.resources import as_file, files
from typing import List, Optional

import numpy as np

from sabr import constants
from sabr.alignment import corrections
from sabr.alignment.backend import (
    AlignmentBackend,
    create_gap_penalty_for_reduced_reference,
)
from sabr.embeddings.mpnn import MPNNEmbeddings
from sabr.util import detect_chain_type, validate_array_shape

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SoftAlignOutput:
    """Alignment matrix and metadata returned by SoftAlign.

    This dataclass stores the results of aligning a query antibody sequence
    against a reference embedding using the SoftAlign neural alignment model.

    Attributes:
        alignment: Binary alignment matrix of shape (n_query, n_reference).
            A value of 1 at position (i, j) indicates that query residue i
            aligns to reference position j. For IMGT alignment, columns
            correspond to IMGT positions 1-128.
        score: Alignment score from the SoftAlign model. Higher scores
            indicate better alignments.
        sim_matrix: Optional similarity matrix of shape (n_query, n_reference)
            containing pairwise similarity scores between query and reference
            embeddings. May be None if not computed.
        chain_type: Detected antibody chain type: "H" (heavy), "K" (kappa),
            or "L" (lambda). May be None if not yet detected.
        idxs1: List of residue identifiers for the query sequence (rows).
            These correspond to PDB residue numbers/insertion codes.
        idxs2: List of position identifiers for the reference (columns).
            For IMGT alignment, these are strings "1" through "128".
    """

    alignment: np.ndarray
    score: float
    sim_matrix: Optional[np.ndarray]
    chain_type: Optional[str]
    idxs1: List[str]
    idxs2: List[str]

    def __post_init__(self) -> None:
        validate_array_shape(
            self.alignment, 0, len(self.idxs1), "alignment", "len(idxs1)"
        )
        validate_array_shape(
            self.alignment, 1, len(self.idxs2), "alignment", "len(idxs2)"
        )
        LOGGER.debug(
            "Created SoftAlignOutput for "
            f"chain_type={self.chain_type}, alignment_shape="
            f"{getattr(self.alignment, 'shape', None)}, score={self.score}"
        )


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
        path = files(embeddings_path) / embeddings_name
        with as_file(path) as p:
            data = np.load(p, allow_pickle=True)
            embedding = MPNNEmbeddings(
                name=str(data["name"]),
                embeddings=data["array"],
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
        use_custom_gap_penalties: bool = True,
    ) -> SoftAlignOutput:
        """Align input embeddings against the unified reference embedding.

        Args:
            input_data: Pre-computed MPNN embeddings for the query chain.
            deterministic_loop_renumbering: Whether to apply deterministic
                renumbering corrections for CDR loops, FR1, FR3, and
                C-terminus. Default is True.
            use_custom_gap_penalties: If True, apply custom gap penalties
                including zero gap open in CDR regions, zero gap open at
                position 10, and overhang penalties. If False, use uniform
                gap penalties. Default is True.

        Returns:
            SoftAlignOutput with the best alignment.
        """
        LOGGER.info(
            f"Aligning embeddings with length={input_data.embeddings.shape[0]}"
        )

        if use_custom_gap_penalties:
            # Add anchor columns to reference for overhang penalties.
            # Anchors at positions 0 and 129 have zero embeddings, so gap
            # penalties between anchors and real positions enforce overhang
            # cost.
            query_len = input_data.embeddings.shape[0]
            idxs_int = [int(x) for x in self.unified_embedding.idxs]
            augmented_idxs = [0] + idxs_int + [129]

            embed_dim = self.unified_embedding.embeddings.shape[1]
            anchor = np.zeros(
                (1, embed_dim), dtype=self.unified_embedding.embeddings.dtype
            )
            augmented_embeddings = np.concatenate(
                [anchor, self.unified_embedding.embeddings, anchor], axis=0
            )

            # Create position-dependent gap penalty matrices for augmented
            # reference
            gap_matrix, open_matrix = create_gap_penalty_for_reduced_reference(
                query_len, augmented_idxs, include_anchors=True
            )

            alignment, sim_matrix, score = self._backend.align(
                input_embeddings=input_data.embeddings,
                target_embeddings=augmented_embeddings,
                temperature=self.temperature,
                gap_matrix=gap_matrix,
                open_matrix=open_matrix,
            )

            # Strip anchor columns from alignment result
            alignment = alignment[:, 1:-1]
        else:
            # Use uniform gap penalties without custom modifications
            LOGGER.info(
                "Using uniform gap penalties (custom penalties disabled)"
            )
            alignment, sim_matrix, score = self._backend.align(
                input_embeddings=input_data.embeddings,
                target_embeddings=self.unified_embedding.embeddings,
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
