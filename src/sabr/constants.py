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
C_TERMINUS_ANCHOR_POSITION = 124  # 0-indexed for IMGT position 125

# Loop definitions are inclusive
IMGT_LOOPS = {
    "CDR1": (27, 38),
    "CDR2": (56, 65),
    "CDR3": (105, 117),
}

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
