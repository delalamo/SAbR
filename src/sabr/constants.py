#!/usr/bin/env python3
"""Constants and configuration values for SAbR.

This module defines constants used throughout the SAbR package including:
- Neural network embedding dimensions
- IMGT numbering scheme definitions
- Amino acid mappings
- Alignment parameters
"""

from enum import Enum

# Neural network configuration
EMBED_DIM = 64
N_MPNN_LAYERS = 3

# IMGT numbering constants
IMGT_MAX_POSITION = 128  # Maximum position in IMGT numbering scheme

# Default alignment temperature for SoftAlign
DEFAULT_TEMPERATURE = 1e-4

# DE loop correction positions (0-indexed columns in alignment matrix)
DE_LOOP_POSITIONS = (80, 81, 82)

# Light chain FR1 positions for correction (0-indexed)
# Positions 6-10 in IMGT (0-indexed: 5-9)
LIGHT_CHAIN_FR1_START = 5  # 0-indexed column for position 6
LIGHT_CHAIN_FR1_END = 9  # 0-indexed column for position 10


class ChainType(Enum):
    """Antibody chain type filter options."""

    HEAVY = "heavy"
    LIGHT = "light"
    AUTO = "auto"


IMGT_FRAMEWORKS = {
    "FW1": list(range(1, 27)),
    "FW2": list(range(39, 56)),
    "FW3": list(range(66, 105)),
    "FW4": list(range(118, 129)),
}

# Loop definitions are inclusive
IMGT_LOOPS = {
    "CDR1": (27, 38),
    "CDR2": (56, 65),
    "CDR3": (105, 117),
}

NON_CDR_RESIDUES = sum(IMGT_FRAMEWORKS.values(), [])
CDR_RESIDUES = [x for x in range(1, 129) if x not in NON_CDR_RESIDUES]

# For testing purposes
ADDITIONAL_GAPS = []

AA_3TO1 = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}
