#!/usr/bin/env python3
"""SoftAlign-based antibody sequence alignment module.

This module provides the SoftAligner class which aligns query antibody
embeddings against chain-labelled reference embeddings to generate IMGT-compatible
alignments.

Key components:
- SoftAligner: Main class for running alignments
- AlignmentResult: Dataclass for alignment results

The alignment process includes:
1. Embedding comparison against labelled H/K/L references
2. Deterministic corrections for CDR loops, DE loop, FR1, and C-terminus
3. Expansion to full 128-position IMGT alignment matrix
4. Chain type selection from the chosen reference embedding label
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from sabr.alignment import corrections
from sabr.alignment.backend import (
    DEFAULT_TEMPERATURE,
    AlignmentBackend,
    create_gap_penalty_for_reduced_reference,
)
from sabr.alignment.validation import validate_alignment_matrix
from sabr.embeddings.mpnn import QueryEmbeddings
from sabr.embeddings.references import (
    VALID_REFERENCE_LABELS,
    ReferenceEmbeddings,
    load_reference_embeddings,
)
from sabr.numbering.imgt import IMGT_MAX_POSITION
from sabr.types import ChainType, parse_chain_type

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class AlignmentResult:
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
        selected_chain_type: Chain type selected from the reference embedding
            label.
    """

    alignment: np.ndarray
    score: float
    selected_chain_type: ChainType
    sim_matrix: Optional[np.ndarray]

    def __post_init__(self) -> None:
        if self.alignment.ndim != 2:
            raise ValueError(
                f"alignment must be two-dimensional. Got shape {self.alignment.shape}."
            )
        LOGGER.debug(
            "Created AlignmentResult for "
            f"chain_type={self.selected_chain_type.value}, alignment_shape="
            f"{getattr(self.alignment, 'shape', None)}, score={self.score}"
        )


class SoftAligner:
    """Align a query embedding against reference embeddings.

    Supports split embeddings with separate references for Heavy (H),
    Kappa (K), and Lambda (L) chains.
    """

    def __init__(
        self,
        embeddings_name: str = "embeddings.npz",
        embeddings_path: str = "sabr.assets",
        temperature: float = DEFAULT_TEMPERATURE,
        random_seed: int = 0,
        reference_embeddings: Dict[str, ReferenceEmbeddings] | None = None,
    ) -> None:
        """
        Initialize the SoftAligner by loading reference embeddings and backend.

        Args:
            embeddings_name: Name of the reference embeddings file.
            embeddings_path: Package path containing the embeddings file.
            temperature: Alignment temperature parameter.
            random_seed: Random seed for reproducibility.
        """
        if reference_embeddings is None:
            reference_embeddings = load_reference_embeddings(
                embeddings_name, embeddings_path
            )
            LOGGER.info(f"Loaded reference embeddings: {list(reference_embeddings)}")

        self.embeddings = reference_embeddings
        self._validate_reference_labels(self.embeddings)
        self.temperature = temperature
        self._backend: AlignmentBackend | None = None
        self._random_seed = random_seed

    @staticmethod
    def _validate_reference_labels(
        embeddings: Dict[str, ReferenceEmbeddings],
    ) -> None:
        labels = set(embeddings)
        if labels != VALID_REFERENCE_LABELS:
            raise ValueError(
                "Reference embeddings must be labelled exactly H, K, and L. "
                f"Got labels: {sorted(labels)}."
            )

    def _get_backend(self) -> AlignmentBackend:
        if self._backend is None:
            self._backend = AlignmentBackend(random_seed=self._random_seed)
        return self._backend

    def _fix_aln(self, old_aln: np.ndarray, idxs: List[int]) -> np.ndarray:
        """Expand an alignment onto IMGT positions using saved indices."""
        aln = np.zeros((old_aln.shape[0], IMGT_MAX_POSITION), dtype=old_aln.dtype)
        aln[:, np.asarray(idxs, dtype=int) - 1] = old_aln

        return aln

    def _align_single(
        self,
        input_data: QueryEmbeddings,
        reference_embedding: ReferenceEmbeddings,
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
        gap_matrix = None
        open_matrix = None
        if use_custom_gap_penalties:
            gap_matrix, open_matrix = create_gap_penalty_for_reduced_reference(
                input_data.embeddings.shape[0], reference_embedding.positions
            )

        alignment, sim_matrix, score = self._get_backend().align(
            input_embeddings=input_data.embeddings,
            target_embeddings=reference_embedding.embeddings,
            temperature=self.temperature,
            gap_matrix=gap_matrix,
            open_matrix=open_matrix,
        )

        return alignment, sim_matrix, score, reference_embedding.positions

    def __call__(
        self,
        input_data: QueryEmbeddings,
        deterministic_loop_renumbering: bool = True,
        use_custom_gap_penalties: bool = True,
        chain_type: str | ChainType | None = None,
    ) -> AlignmentResult:
        """Align input embeddings against reference embeddings.

        Args:
            input_data: Pre-computed MPNN embeddings for the query chain.
            deterministic_loop_renumbering: Whether to apply deterministic
                renumbering corrections for CDR loops, FR1, FR3, and
                C-terminus. Default is True.
            use_custom_gap_penalties: If True, apply custom gap penalties
                by setting gap-open to zero in IMGT CDR regions only.
                If False, use uniform gap penalties. Default is True.
            chain_type: Which reference embeddings to use.
                None/"auto" (default): Try all available and pick best by score.
                "H", "K", "L": Use the specified chain type's embeddings.

        Returns:
            AlignmentResult with the best alignment.
        """
        LOGGER.info(f"Aligning embeddings with length={input_data.embeddings.shape[0]}")

        if not use_custom_gap_penalties:
            LOGGER.info("Using uniform gap penalties (custom penalties disabled)")

        # Determine which embeddings to use
        requested_chain_type = parse_chain_type(chain_type)
        requested_label = (
            None if requested_chain_type is None else requested_chain_type.value
        )

        if requested_label is None:
            # Try all available embeddings and pick best by score
            best_result = None
            best_score = float("-inf")
            best_label = None

            for label, embedding in self.embeddings.items():
                alignment, sim_matrix, score, idxs = self._align_single(
                    input_data, embedding, use_custom_gap_penalties
                )
                LOGGER.debug(f"Alignment score for {label} reference: {score:.4f}")
                if score > best_score:
                    best_score = score
                    best_result = (alignment, sim_matrix, score, idxs)
                    best_label = label

            LOGGER.info(
                f"Selected {best_label} reference embeddings "
                f"with score {best_score:.4f}"
            )
            alignment, sim_matrix, score, ref_idxs = best_result
            selected_chain_type = parse_chain_type(best_label)
            if selected_chain_type is None:
                raise ValueError("Alignment selected no concrete chain type.")
        else:
            # Use specified chain type
            if requested_label not in self.embeddings:
                available = list(self.embeddings.keys())
                raise ValueError(
                    f"Chain type embedding label '{requested_label}' not found. "
                    f"Available: {available}"
                )
            embedding = self.embeddings[requested_label]
            alignment, sim_matrix, score, ref_idxs = self._align_single(
                input_data, embedding, use_custom_gap_penalties
            )
            LOGGER.info(
                f"Using {requested_label} reference embeddings (score: {score:.4f})"
            )
            selected_chain_type = requested_chain_type

        aln = self._fix_aln(alignment, ref_idxs)
        aln = np.round(aln).astype(int)
        validate_alignment_matrix(aln)

        gap_indices = input_data.gap_indices
        if gap_indices:
            LOGGER.info(f"Detected {len(gap_indices)} structural gap(s) in input")

        if deterministic_loop_renumbering:
            aln = corrections.apply_deterministic_corrections(
                aln, selected_chain_type, gap_indices=gap_indices
            )

        return AlignmentResult(
            alignment=aln,
            score=score,
            selected_chain_type=selected_chain_type,
            sim_matrix=sim_matrix,
        )
