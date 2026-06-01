#!/usr/bin/env python3
"""Structure file I/O utilities.

This module provides functions for reading protein structure files
in PDB and mmCIF formats using BioPython parsers.
"""

import logging
from pathlib import Path

from Bio.PDB import PDBIO, PDBParser
from Bio.PDB.mmcifio import MMCIFIO
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.Structure import Structure

from sabr.errors import OutputFormatError

LOGGER = logging.getLogger(__name__)


def read_structure(file_path: str) -> Structure:
    """Read a PDB or mmCIF file into a BioPython structure."""
    path = Path(file_path)
    if path.suffix.lower() not in {".pdb", ".cif"}:
        raise OutputFormatError(
            f"Input file must be a PDB (.pdb) or mmCIF (.cif) file. Got: {path}"
        )
    parser = (
        MMCIFParser(QUIET=True)
        if path.suffix.lower() == ".cif"
        else PDBParser(QUIET=True)
    )
    structure = parser.get_structure("structure", str(path))
    LOGGER.info(f"Read structure from {path} using BioPython")
    return structure


def write_structure(structure: Structure, output_path: str) -> None:
    """Write a BioPython structure as PDB or mmCIF."""
    path = Path(output_path)
    suffix = path.suffix.lower()
    if suffix == ".cif":
        io = MMCIFIO()
    elif suffix == ".pdb":
        io = PDBIO()
    else:
        raise OutputFormatError(
            f"Output file must have extension .pdb or .cif. Got: {path}"
        )

    io.set_structure(structure)
    io.save(str(path))
    LOGGER.info(f"Saved structure to {path}")
