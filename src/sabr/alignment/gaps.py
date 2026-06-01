"""Structural gap detection helpers."""

from __future__ import annotations

import logging
from typing import FrozenSet

import numpy as np

LOGGER = logging.getLogger(__name__)
PEPTIDE_BOND_LENGTH = 1.33
PEPTIDE_BOND_MAX_DISTANCE = 2 * PEPTIDE_BOND_LENGTH
BACKBONE_N_IDX = 0
BACKBONE_C_IDX = 2


def detect_backbone_gaps(
    coords: np.ndarray,
    threshold: float = PEPTIDE_BOND_MAX_DISTANCE,
) -> FrozenSet[int]:
    """Detect structural gaps by checking C-N peptide bond distances."""
    if coords.ndim == 4 and coords.shape[0] == 1:
        coords = coords[0]

    if coords.ndim != 3 or coords.shape[1] != 4 or coords.shape[2] != 3:
        raise ValueError(f"Expected coords shape [N, 4, 3], got {coords.shape}")

    n_residues = coords.shape[0]
    if n_residues <= 1:
        return frozenset()

    c_coords = coords[:-1, BACKBONE_C_IDX, :]
    n_coords = coords[1:, BACKBONE_N_IDX, :]
    distances = np.linalg.norm(c_coords - n_coords, axis=1)

    gap_indices = set(np.where(distances > threshold)[0])
    if gap_indices:
        if len(gap_indices) <= 3:
            gap_list = ", ".join(str(i) for i in sorted(gap_indices))
            LOGGER.info(
                "Detected %s structural gap(s) at: %s", len(gap_indices), gap_list
            )
        else:
            gap_list = ", ".join(str(i) for i in sorted(gap_indices)[:3])
            LOGGER.info(
                "Detected %s structural gaps (first 3: %s, ...)",
                len(gap_indices),
                gap_list,
            )
        for idx in sorted(gap_indices):
            LOGGER.debug(
                "Gap between residues %s and %s: C-N distance = %.2f A "
                "(threshold: %.2f A)",
                idx,
                idx + 1,
                distances[idx],
                threshold,
            )

    return frozenset(gap_indices)


def has_gap_in_region(
    gap_indices: FrozenSet[int],
    start_row: int,
    end_row: int,
) -> bool:
    """Return whether a structural gap splits an inclusive row region."""
    return any(i in gap_indices for i in range(start_row, end_row))
