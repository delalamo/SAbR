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
from typing import Dict, List, Optional

import numpy as np

from sabr import constants
from sabr.alignment import corrections
from sabr.alignment.backend import (
    AlignmentBackend,
    create_gap_penalty_for_reduced_reference,
)
from sabr.alignment.validation import validate_alignment_matrix
from sabr.embeddings.mpnn import MPNNEmbeddings
from sabr.embeddings.schema import load_reference_embeddings
from sabr.types import ChainType, chain_type_value
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
        selected_reference: Reference embedding key selected for alignment.
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
    selected_reference: str = ""

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
    """Align a query embedding against reference embeddings.

    Supports both unified (single) reference embeddings and split embeddings
    with separate references for Heavy (H), Kappa (K), and Lambda (L) chains.
    """

    def __init__(
        self,
        embeddings_name: str = "embeddings.npz",
        embeddings_path: str = "sabr.assets",
        temperature: float = constants.DEFAULT_TEMPERATURE,
        random_seed: int = 0,
        backend: AlignmentBackend | None = None,
        reference_embeddings: Dict[str, MPNNEmbeddings] | None = None,
    ) -> None:
        """
        Initialize the SoftAligner by loading reference embeddings and backend.

        Args:
            embeddings_name: Name of the reference embeddings file.
            embeddings_path: Package path containing the embeddings file.
            temperature: Alignment temperature parameter.
            random_seed: Random seed for reproducibility.
        """
        self.embeddings = reference_embeddings or self.read_embeddings(
            embeddings_name=embeddings_name, embeddings_path=embeddings_path
        )
        self.temperature = temperature
        self._backend = backend
        self._random_seed = random_seed

    def _get_backend(self) -> AlignmentBackend:
        if self._backend is None:
            self._backend = AlignmentBackend(random_seed=self._random_seed)
        return self._backend

    def read_embeddings(
        self,
        embeddings_name: str = "embeddings.npz",
        embeddings_path: str = "sabr.assets",
    ) -> Dict[str, MPNNEmbeddings]:
        """Load packaged reference embeddings.

        Automatically detects the file format:
        - Old format (has 'array' key): Returns {"unified": embedding}
        - New split format (has 'arr_0' key): Returns {"H", "K", "L"} dict

        Returns:
            Dictionary mapping chain type keys to MPNNEmbeddings objects.
        """
        embeddings = load_reference_embeddings(embeddings_name, embeddings_path)
        LOGGER.info(f"Loaded reference embeddings: {list(embeddings)}")
        return embeddings

    def fix_aln(self, old_aln: np.ndarray, idxs: List[int]) -> np.ndarray:
        """Expand an alignment onto IMGT positions using saved indices."""
        aln = np.zeros(
            (old_aln.shape[0], constants.IMGT_MAX_POSITION), dtype=old_aln.dtype
        )
        aln[:, np.asarray(idxs, dtype=int) - 1] = old_aln

        return aln

    def _align_single(
        self,
        input_data,
        reference_embedding: MPNNEmbeddings,
        use_custom_gap_penalties: bool,
    ):
        """Align input embeddings against a single reference embedding.

        Args:
            input_data: Pre-computed MPNN embeddings for the query chain.
            reference_embedding: Reference embedding to align against.
            use_custom_gap_penalties: Whether to apply custom gap penalties.

        Returns:
            Tuple of (alignment, sim_matrix, score, reference_idxs).
        """
        if use_custom_gap_penalties:
            query_len = input_data.embeddings.shape[0]
            idxs_int = [int(x) for x in reference_embedding.idxs]

            gap_matrix, open_matrix = create_gap_penalty_for_reduced_reference(
                query_len, idxs_int
            )

            alignment, sim_matrix, score = self._get_backend().align(
                input_embeddings=input_data.embeddings,
                target_embeddings=reference_embedding.embeddings,
                temperature=self.temperature,
                gap_matrix=gap_matrix,
                open_matrix=open_matrix,
            )
        else:
            alignment, sim_matrix, score = self._get_backend().align(
                input_embeddings=input_data.embeddings,
                target_embeddings=reference_embedding.embeddings,
                temperature=self.temperature,
            )

        return alignment, sim_matrix, score, reference_embedding.idxs

    def __call__(
        self,
        input_data,
        deterministic_loop_renumbering: bool = True,
        use_custom_gap_penalties: bool = True,
        reference_chain_type: str | ChainType = "auto",
    ) -> SoftAlignOutput:
        """Align input embeddings against reference embeddings.

        Args:
            input_data: Pre-computed MPNN embeddings for the query chain.
            deterministic_loop_renumbering: Whether to apply deterministic
                renumbering corrections for CDR loops, FR1, FR3, and
                C-terminus. Default is True.
            use_custom_gap_penalties: If True, apply custom gap penalties
                by setting gap-open to zero in IMGT CDR regions only.
                If False, use uniform gap penalties. Default is True.
            reference_chain_type: Which reference embeddings to use.
                "auto" (default): Try all available and pick best by score.
                "H", "K", "L": Use the specified chain type's embeddings.
                "unified": Use unified embeddings (for old format files).

        Returns:
            SoftAlignOutput with the best alignment.
        """
        LOGGER.info(f"Aligning embeddings with length={input_data.embeddings.shape[0]}")

        if not use_custom_gap_penalties:
            LOGGER.info("Using uniform gap penalties (custom penalties disabled)")

        # Determine which embeddings to use
        reference_key = (
            chain_type_value(reference_chain_type)
            if isinstance(reference_chain_type, ChainType)
            else reference_chain_type
        )

        if reference_key == "auto":
            # Try all available embeddings and pick best by score
            best_result = None
            best_score = float("-inf")
            best_chain_type = None

            for chain_type, embedding in self.embeddings.items():
                alignment, sim_matrix, score, idxs = self._align_single(
                    input_data, embedding, use_custom_gap_penalties
                )
                LOGGER.debug(f"Alignment score for {chain_type} reference: {score:.4f}")
                if score > best_score:
                    best_score = score
                    best_result = (alignment, sim_matrix, score, idxs)
                    best_chain_type = chain_type

            LOGGER.info(
                f"Selected {best_chain_type} reference embeddings "
                f"with score {best_score:.4f}"
            )
            alignment, sim_matrix, score, ref_idxs = best_result
            selected_reference = best_chain_type
        else:
            # Use specified chain type
            if reference_key not in self.embeddings:
                available = list(self.embeddings.keys())
                raise ValueError(
                    f"Reference chain type '{reference_key}' not found. "
                    f"Available: {available}"
                )
            embedding = self.embeddings[reference_key]
            alignment, sim_matrix, score, ref_idxs = self._align_single(
                input_data, embedding, use_custom_gap_penalties
            )
            LOGGER.info(
                f"Using {reference_key} reference embeddings (score: {score:.4f})"
            )
            selected_reference = reference_key

        aln = self.fix_aln(alignment, ref_idxs)
        aln = np.round(aln).astype(int)
        validate_alignment_matrix(aln)

        gap_indices = input_data.gap_indices
        if gap_indices:
            LOGGER.info(f"Detected {len(gap_indices)} structural gap(s) in input")

        if deterministic_loop_renumbering:
            aln, detected_chain_type = corrections.apply_deterministic_corrections(
                aln, gap_indices=gap_indices
            )
        else:
            detected_chain_type = detect_chain_type(aln)
            LOGGER.info(f"Detected chain type: {detected_chain_type}")

        return SoftAlignOutput(
            chain_type=detected_chain_type,
            selected_reference=selected_reference,
            alignment=aln,
            score=score,
            sim_matrix=sim_matrix,
            idxs1=input_data.idxs,
            idxs2=[str(x) for x in range(1, constants.IMGT_MAX_POSITION + 1)],
        )
