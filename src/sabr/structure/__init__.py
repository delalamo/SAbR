#!/usr/bin/env python3
"""Structure I/O for SAbR.

This package provides structure file handling including:
- Reading PDB/mmCIF files
- Threading alignments onto structures
- Writing renumbered structures
"""

from sabr.structure.io import read_structure_biopython, read_structure_gemmi
from sabr.structure.threading import (
    has_extended_insertion_codes,
    thread_alignment,
    thread_onto_chain,
    validate_output_format,
)

__all__ = [
    # I/O
    "read_structure_gemmi",
    "read_structure_biopython",
    # Threading
    "thread_alignment",
    "thread_onto_chain",
    "validate_output_format",
    "has_extended_insertion_codes",
]
