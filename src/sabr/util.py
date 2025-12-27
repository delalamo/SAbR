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

from sabr import constants

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

    Uses DE loop region (positions 79-84) to determine chain type:
    - Heavy chains have 6 residues in DE loop (79, 80, 81, 82, 83, 84)
    - Light chains have 4 residues (79, 80, 83, 84 - skip 81, 82)

    The detection counts total residues in the DE loop region rather than
    just checking specific positions, which handles cases where the soft
    alignment may have misplaced residues.

    For light chains, position 10 distinguishes kappa from lambda:
    - Kappa chains have position 10 occupied
    - Lambda chains lack position 10

    Args:
        alignment: The alignment matrix (rows=sequence, cols=IMGT positions).

    Returns:
        Chain type: "H" (heavy), "K" (kappa), or "L" (lambda).
    """
    # Count residues in the DE loop region (positions 79-84)
    de_loop_cols = range(
        constants.DE_LOOP_START_COL, constants.DE_LOOP_END_COL + 1
    )
    n_residues_in_de_loop = sum(
        1 for col in de_loop_cols if alignment[:, col].sum() >= 1
    )

    # Heavy chains have 6 residues (79-84), light chains have 4 (skip 81-82)
    if n_residues_in_de_loop >= constants.DE_LOOP_HEAVY_THRESHOLD:
        # 5 or 6 residues -> heavy chain
        LOGGER.info(
            f"Detected chain type: H (heavy) based on DE loop "
            f"having {n_residues_in_de_loop} residues in positions 79-84"
        )
        return "H"
    else:
        # 4 or fewer residues -> light chain
        # Check position 10 to distinguish kappa from lambda
        pos10_col = constants.FR1_POSITION_10_COL
        pos10_occupied = alignment[:, pos10_col].sum() >= 1

        if pos10_occupied:
            LOGGER.info(
                f"Detected chain type: K (kappa) based on DE loop "
                f"having {n_residues_in_de_loop} residues and position 10 occupied"
            )
            return "K"
        else:
            LOGGER.info(
                f"Detected chain type: L (lambda) based on DE loop "
                f"having {n_residues_in_de_loop} residues and position 10 empty"
            )
            return "L"
