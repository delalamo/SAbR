#!/usr/bin/env python3
"""Core utilities and configuration for SAbR.

This package contains shared utilities, constants, and type definitions
used throughout the SAbR codebase.
"""

from sabr.core.constants import (
    AA_3TO1,
    BACKBONE_C_IDX,
    BACKBONE_CA_IDX,
    BACKBONE_CB_IDX,
    BACKBONE_N_IDX,
    C_TERMINUS_ANCHOR_POSITION,
    CB_BOND_ANGLE,
    CB_BOND_LENGTH,
    CB_DIHEDRAL,
    CDR_ANCHORS,
    DEFAULT_TEMPERATURE,
    EMBED_DIM,
    FR1_ANCHOR_END_COL,
    FR1_ANCHOR_START_COL,
    FR1_KAPPA_RESIDUE_COUNT,
    FR3_POS81_COL,
    FR3_POS82_COL,
    FR3_POS83_COL,
    FR3_POS84_COL,
    IMGT_LOOPS,
    IMGT_MAX_POSITION,
    IMGT_REGIONS,
    N_MPNN_LAYERS,
    PEPTIDE_BOND_MAX_DISTANCE,
    SW_GAP_EXTEND,
    SW_GAP_OPEN,
)
from sabr.core.util import (
    configure_logging,
    detect_backbone_gaps,
    detect_chain_type,
    has_gap_in_region,
)

__all__ = [
    # Constants
    "EMBED_DIM",
    "N_MPNN_LAYERS",
    "CB_BOND_LENGTH",
    "CB_BOND_ANGLE",
    "CB_DIHEDRAL",
    "PEPTIDE_BOND_MAX_DISTANCE",
    "BACKBONE_N_IDX",
    "BACKBONE_CA_IDX",
    "BACKBONE_C_IDX",
    "BACKBONE_CB_IDX",
    "SW_GAP_EXTEND",
    "SW_GAP_OPEN",
    "IMGT_MAX_POSITION",
    "DEFAULT_TEMPERATURE",
    "FR1_ANCHOR_START_COL",
    "FR1_ANCHOR_END_COL",
    "FR1_KAPPA_RESIDUE_COUNT",
    "FR3_POS81_COL",
    "FR3_POS82_COL",
    "FR3_POS83_COL",
    "FR3_POS84_COL",
    "C_TERMINUS_ANCHOR_POSITION",
    "IMGT_LOOPS",
    "IMGT_REGIONS",
    "CDR_ANCHORS",
    "AA_3TO1",
    # Utilities
    "configure_logging",
    "detect_chain_type",
    "detect_backbone_gaps",
    "has_gap_in_region",
]
