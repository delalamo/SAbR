#!/usr/bin/env python3
"""Alignment engine for SAbR.

This package provides the alignment functionality for SAbR including:
- SoftAligner: Main alignment class
- SoftAlignOutput: Output dataclass
- AlignmentBackend: JAX/Haiku backend for alignment
- HMM conversion utilities
- Deterministic correction functions
"""

from sabr.alignment.aln2hmm import (
    Aln2HmmOutput,
    State,
    alignment_matrix_to_state_vector,
)
from sabr.alignment.backend import AlignmentBackend
from sabr.alignment.corrections import (
    apply_deterministic_corrections,
    correct_c_terminus,
    correct_cdr_loop,
    correct_fr1_alignment,
    correct_fr3_alignment,
    correct_gap_numbering,
    find_nearest_occupied_column,
)
from sabr.alignment.output import SoftAlignOutput
from sabr.alignment.soft_aligner import SoftAligner

__all__ = [
    # Main classes
    "SoftAligner",
    "SoftAlignOutput",
    "AlignmentBackend",
    # HMM conversion
    "State",
    "Aln2HmmOutput",
    "alignment_matrix_to_state_vector",
    # Corrections
    "apply_deterministic_corrections",
    "correct_cdr_loop",
    "correct_fr1_alignment",
    "correct_fr3_alignment",
    "correct_c_terminus",
    "correct_gap_numbering",
    "find_nearest_occupied_column",
]
