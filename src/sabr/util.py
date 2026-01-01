#!/usr/bin/env python3
"""Utility functions for SAbR.

This module provides helper functions for:
- Configuring logging
- Detecting antibody chain types from alignments
- Detecting structural gaps from backbone coordinates
"""

import logging
from typing import Set

import numpy as np

from sabr import constants

LOGGER = logging.getLogger(__name__)


def configure_logging(verbose: bool) -> None:
    """Configure logging level based on verbosity flag.

    Args:
        verbose: If True, set logging level to INFO. Otherwise, set to WARNING.
    """
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, force=True)


def detect_chain_type(alignment: np.ndarray) -> str:
    """Detect antibody chain type from alignment.

    Uses the NUMBER of residues in DE loop region (positions 79-84) to determine
    chain type. With unified embeddings, soft alignment places residues at
    positions 81-82 for ALL chains, so we cannot simply check occupancy.
    Instead, we count total residues:
    - Heavy chains: 6 residues in DE loop (79, 80, 81, 82, 83, 84)
    - Light chains: 4 residues in DE loop (79, 80, 83, 84) - skip 81, 82

    For light chains, position 10 distinguishes kappa from lambda:
    - Kappa chains have position 10 occupied
    - Lambda chains lack position 10

    Args:
        alignment: The alignment matrix (rows=sequence, cols=IMGT positions).

    Returns:
        Chain type: "H" (heavy), "K" (kappa), or "L" (lambda).
    """
    # Count residues in DE loop region (positions 79-84, 0-indexed 78-83)
    # Each row can only align to one column, so we count rows with any alignment
    # in the DE loop columns
    de_loop_start = 78  # 0-indexed for position 79
    de_loop_end = 84  # 0-indexed for position 84 (exclusive end = 84)

    # Sum across DE loop columns for each row, then count rows with alignment
    de_loop_region = alignment[:, de_loop_start:de_loop_end]
    n_residues_in_de_loop = (de_loop_region.sum(axis=1) > 0).sum()

    LOGGER.info(f"DE loop residue count: {n_residues_in_de_loop}")

    # Heavy chains have 6 residues (79-84), light chains have 4 (79, 80, 83, 84)
    # Use threshold of 5 to distinguish
    if n_residues_in_de_loop >= 5:
        LOGGER.info(
            f"Detected chain type: H (heavy) based on {n_residues_in_de_loop} "
            "residues in DE loop region (>= 5)"
        )
        return "H"
    else:
        # Light chain - check position 10 to distinguish kappa from lambda
        pos10_col = 9  # 0-indexed column for IMGT position 10
        pos10_occupied = alignment[:, pos10_col].sum() >= 1

        if pos10_occupied:
            LOGGER.info(
                f"Detected chain type: K (kappa) - {n_residues_in_de_loop} "
                "residues in DE loop (< 5), position 10 occupied"
            )
            return "K"
        else:
            LOGGER.info(
                f"Detected chain type: L (lambda) - {n_residues_in_de_loop} "
                "residues in DE loop (< 5), position 10 unoccupied"
            )
            return "L"


def detect_backbone_gaps(
    coords: np.ndarray,
    threshold: float = constants.PEPTIDE_BOND_MAX_DISTANCE,
) -> Set[int]:
    """Detect structural gaps in backbone by checking C-N bond distances.

    A gap indicates missing residues in the structure, where the backbone
    carbonyl carbon (C) of residue i is too far from the backbone nitrogen
    (N) of residue i+1. Standard peptide bond C-N distance is ~1.32-1.35 Å.

    Args:
        coords: Backbone coordinates with shape [N, 4, 3] where the 4
            atoms are [N, CA, C, CB] in order. Can also be [1, N, 4, 3].
        threshold: Maximum allowed C-N distance in Angstroms. Distances
            above this indicate a structural gap.

    Returns:
        Set of row indices where gaps occur. Each index i in the set means
        there is a gap AFTER residue i (between residue i and i+1).
    """
    # Handle batch dimension if present
    if coords.ndim == 4 and coords.shape[0] == 1:
        coords = coords[0]

    if coords.ndim != 3 or coords.shape[1] != 4 or coords.shape[2] != 3:
        raise ValueError(f"Expected coords shape [N, 4, 3], got {coords.shape}")

    n_residues = coords.shape[0]
    gap_indices = set()

    # Atom indices: N=0, CA=1, C=2, CB=3
    c_atom_idx = 2
    n_atom_idx = 0

    for i in range(n_residues - 1):
        # C atom of residue i
        c_coord = coords[i, c_atom_idx, :]
        # N atom of residue i+1
        n_coord = coords[i + 1, n_atom_idx, :]

        # Calculate Euclidean distance
        distance = np.linalg.norm(c_coord - n_coord)

        if distance > threshold:
            LOGGER.info(
                f"Structural gap detected between residues {i} and {i + 1}: "
                f"C-N distance = {distance:.2f} Å (threshold: {threshold} Å)"
            )
            gap_indices.add(i)

    return gap_indices


def has_gap_in_region(
    gap_indices: Set[int],
    start_row: int,
    end_row: int,
) -> bool:
    """Check if there is a structural gap within a region of residues.

    Args:
        gap_indices: Set of row indices where gaps occur (from
            detect_backbone_gaps).
        start_row: First row index of the region (inclusive).
        end_row: Last row index of the region (inclusive).

    Returns:
        True if any gap exists between start_row and end_row-1 (since
        gaps are between residue i and i+1, we check up to end_row-1).
    """
    for i in range(start_row, end_row):
        if i in gap_indices:
            return True
    return False
