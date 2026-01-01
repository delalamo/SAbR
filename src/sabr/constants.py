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

# CB (C-beta) position calculation constants (standard protein geometry)
CB_BOND_LENGTH = 1.522  # C-CA bond length in Angstroms
CB_BOND_ANGLE = 1.927  # N-CA-CB angle in radians (~110.5 degrees)
CB_DIHEDRAL = -2.143  # N-CA-C-CB dihedral angle in radians

# Peptide bond geometry for gap detection
# Standard C-N peptide bond length is ~1.32-1.35 Å
# A gap (missing residue) is detected when C-N distance exceeds threshold
PEPTIDE_BOND_MAX_DISTANCE = 2.0  # Angstroms; above = structural gap

# Smith-Waterman alignment parameters from fine-tuned SoftAlign model
# These are the gap penalties for the affine gap scoring function
# Fine-tuned from CONT_SW_05_T_3_1 with LR=3e-5, achieving val loss 115.81
# (4.8% improvement over baseline 121.69)
SW_GAP_EXTEND = -0.175027  # Gap extension penalty (negative = penalty)
SW_GAP_OPEN = -2.525591  # Gap opening penalty

# IMGT numbering constants
IMGT_MAX_POSITION = 128  # Maximum position in IMGT numbering scheme

# Default alignment temperature for SoftAlign
DEFAULT_TEMPERATURE = 1e-4

# FR1 region constants for alignment correction (0-indexed columns)
# FR1 spans IMGT positions 1-26, but correction focuses on positions 6-12
FR1_ANCHOR_START_COL = 5  # 0-indexed column for IMGT position 6
FR1_ANCHOR_END_COL = 11  # 0-indexed column for IMGT position 12
FR1_KAPPA_RESIDUE_COUNT = 7  # Kappa chains have 7 residues in positions 6-12

# FR3 positions 81-84 (0-indexed columns)
FR3_POS81_COL = 80
FR3_POS82_COL = 81
FR3_POS83_COL = 82
FR3_POS84_COL = 83

# C-terminus correction positions (0-indexed)
# Used to detect and fix unassigned residues at the end of FW4
C_TERMINUS_ANCHOR_POSITION = 124  # 0-indexed for IMGT position 125

# Loop definitions are inclusive (CDR start position, CDR end position)
IMGT_LOOPS = {
    "CDR1": (27, 38),
    "CDR2": (56, 65),
    "CDR3": (105, 117),
}

# Full IMGT region definitions including frameworks and CDRs
# Framework positions: FR1=1-26, FR2=39-55, FR3=66-104, FR4=118-128
IMGT_REGIONS = {
    "FR1": list(range(1, 27)),
    "CDR1": list(range(27, 39)),
    "FR2": list(range(39, 56)),
    "CDR2": list(range(56, 66)),
    "FR3": list(range(66, 105)),
    "CDR3": list(range(105, 118)),
    "FR4": list(range(118, 129)),
}

# Framework anchor positions for CDR renumbering
# These are conserved framework residues used to identify CDR boundaries.
# CDR residues are determined by counting rows between these anchors.
# Anchors are the last FW residue before CDR and first FW residue after,
# except CDR1 which uses Cys23 (conserved) requiring FW fill-in for 24-26.
CDR_ANCHORS = {
    "CDR1": (
        23,
        40,
    ),  # Cys23 (conserved) and position 40 (first FW2 after CDR1)
    "CDR2": (
        54,
        66,
    ),  # position 54 (55 is last FW2, added linearly) and 66 (first FW3)
    "CDR3": (104, 118),  # position 104 (last FW3) and 118 (first FW4)
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
