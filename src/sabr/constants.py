#!/usr/bin/env python3
"""Constants and configuration values for SAbR.

This module defines constants used throughout the SAbR package including:
- Neural network embedding dimensions
- IMGT numbering scheme definitions
- Amino acid mappings
- Alignment parameters
"""

from typing import List, Tuple

# Type alias for ANARCI alignment output:
# list of ((residue_number, insertion_code), amino_acid)
AnarciAlignment = List[Tuple[Tuple[int, str], str]]

# Neural network configuration
EMBED_DIM = 64
N_MPNN_LAYERS = 3

# IMGT numbering constants
IMGT_MAX_POSITION = 128  # Maximum position in IMGT numbering scheme

# Default alignment temperature for SoftAlign
DEFAULT_TEMPERATURE = 1e-4

# Light chain FR1 positions for correction (0-indexed)
# Positions 6-10 in IMGT (0-indexed: 5-9)
LIGHT_CHAIN_FR1_START = 5  # 0-indexed column for position 6
LIGHT_CHAIN_FR1_END = 9  # 0-indexed column for position 10

# C-terminus correction positions (0-indexed)
# Used to detect and fix unassigned residues at the end of FW4
# If residues after the last assigned position (around 125/126) are unassigned,
# they should be deterministically assigned to positions 127, 128
C_TERMINUS_ANCHOR_POSITION = 124  # 0-indexed for IMGT position 125
C_TERMINUS_LAST_POSITIONS = (
    125,
    126,
    127,
)  # 0-indexed for positions 126, 127, 128


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
