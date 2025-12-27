#!/usr/bin/env python3
"""Utility functions for SAbR.

This module provides helper functions for:
- Loading SoftAlign model parameters
- Configuring logging
"""

import logging
from importlib.resources import files
from typing import Any, Dict

import numpy as np
from softalign.utils import convert_numpy_to_jax, unflatten_dict

LOGGER = logging.getLogger(__name__)


def configure_logging(verbose: bool) -> None:
    """Configure logging level based on verbosity flag.

    Args:
        verbose: If True, set logging level to INFO. Otherwise, set to WARNING.
    """
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, force=True)


def read_softalign_params(
    params_name: str = "CONT_SW_05_T_3_1",
    params_path: str = "softalign.models",
) -> Dict[str, Any]:
    """Load SoftAlign parameters from package resources.

    Args:
        params_name: Name of the model parameters file (without extension).
        params_path: Package path containing the parameters file.

    Returns:
        Dictionary containing the model parameters as JAX arrays.
    """
    package_files = files(params_path)
    npz_path = package_files / f"{params_name}.npz"

    with open(npz_path, "rb") as f:
        data = dict(np.load(f, allow_pickle=False))

    # Unflatten the dictionary structure and convert to JAX arrays
    params = unflatten_dict(data)
    params = convert_numpy_to_jax(params)
    LOGGER.info(f"Loaded model parameters from {npz_path}")
    return params


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
