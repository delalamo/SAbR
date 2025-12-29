#!/usr/bin/env python3
"""SoftAlign output dataclass module.

This module defines the SoftAlignOutput dataclass which holds the results
of a SoftAlign alignment operation, including the alignment matrix,
similarity scores, and chain type information.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

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
        if self.alignment.shape[0] != len(self.idxs1):
            raise ValueError(
                f"alignment.shape[0] ({self.alignment.shape[0]}) must match "
                f"len(idxs1) ({len(self.idxs1)}). "
            )
        if self.alignment.shape[1] != len(self.idxs2):
            raise ValueError(
                f"alignment.shape[1] ({self.alignment.shape[1]}) must match "
                f"len(idxs2) ({len(self.idxs2)}). "
            )
        LOGGER.debug(
            "Created SoftAlignOutput for "
            f"chain_type={self.chain_type}, alignment_shape="
            f"{getattr(self.alignment, 'shape', None)}, score={self.score}"
        )
