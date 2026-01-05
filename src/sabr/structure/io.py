#!/usr/bin/env python3
"""Structure file reading utilities.

This module provides functions for reading protein structure files
in PDB and mmCIF formats using both Gemmi and BioPython parsers.
"""

import logging

import gemmi
from Bio.PDB import PDBParser
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.Structure import Structure

LOGGER = logging.getLogger(__name__)


def read_structure_gemmi(file_path: str) -> gemmi.Structure:
    """Read a structure file using Gemmi.

    Args:
        file_path: Path to PDB or mmCIF file.

    Returns:
        Gemmi Structure object.
    """
    structure = gemmi.read_structure(file_path)
    LOGGER.info(f"Read structure from {file_path} using Gemmi")
    return structure


def read_structure_biopython(file_path: str) -> Structure:
    """Read a structure file using BioPython.

    Args:
        file_path: Path to PDB or mmCIF file.

    Returns:
        BioPython Structure object.
    """
    if file_path.endswith(".cif"):
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)

    structure = parser.get_structure("structure", file_path)
    LOGGER.info(f"Read structure from {file_path} using BioPython")
    return structure
