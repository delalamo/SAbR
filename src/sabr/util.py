#!/usr/bin/env python3
"""Utility functions for SAbR.

This module provides helper functions for:
- Configuring logging
- Detecting antibody chain types from alignments
- Detecting structural gaps from backbone coordinates
"""

import logging
from typing import FrozenSet

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
) -> FrozenSet[int]:
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
        FrozenSet of row indices where gaps occur. Each index i in the set
        means there is a gap AFTER residue i (between residue i and i+1).

    Raises:
        ValueError: If coords has invalid shape.
    """
    # Handle batch dimension if present
    if coords.ndim == 4 and coords.shape[0] == 1:
        coords = coords[0]

    if coords.ndim != 3 or coords.shape[1] != 4 or coords.shape[2] != 3:
        raise ValueError(f"Expected coords shape [N, 4, 3], got {coords.shape}")

    n_residues = coords.shape[0]
    if n_residues <= 1:
        return frozenset()

    # Extract C atoms (all but last residue) and N atoms (all but first)
    c_coords = coords[:-1, constants.BACKBONE_C_IDX, :]  # [N-1, 3]
    n_coords = coords[1:, constants.BACKBONE_N_IDX, :]  # [N-1, 3]

    # Vectorized distance calculation
    distances = np.linalg.norm(c_coords - n_coords, axis=1)

    # Find gaps (where distance exceeds threshold)
    gap_mask = distances > threshold
    gap_indices = set(np.where(gap_mask)[0])

    for idx in gap_indices:
        LOGGER.info(
            f"Structural gap detected between residues {idx} and {idx + 1}: "
            f"C-N distance = {distances[idx]:.2f} Å (threshold: {threshold} Å)"
        )

    return frozenset(gap_indices)


def has_gap_in_region(
    gap_indices: FrozenSet[int],
    start_row: int,
    end_row: int,
) -> bool:
    """Check if there is a structural gap within a region of residues.

    A gap at index i represents a structural break between residue i and
    residue i+1 (the C-N peptide bond distance is too large). This function
    checks whether any such gap would split the region.

    Args:
        gap_indices: FrozenSet of indices where gaps occur. A gap at index i
            indicates a break between residues at rows i and i+1.
        start_row: First row index of the region (inclusive).
        end_row: Last row index of the region (inclusive).

    Returns:
        True if any internal gap exists. We check gap indices from start_row
        to end_row-1 because a gap at index i affects residues i and i+1;
        a gap at end_row would be between end_row and end_row+1, which extends
        outside the region.

    Example:
        For region [5, 10] inclusive, we check gap indices 5-9:
        - gap 5: break between residues 5-6 (both in region)
        - gap 9: break between residues 9-10 (both in region)
        - gap 10: break between residues 10-11 (11 is outside region)
    """
    for i in range(start_row, end_row):
        if i in gap_indices:
            return True
    return False
