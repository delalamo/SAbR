#!/usr/bin/env python3
"""Utility functions for SAbR.

This module provides helper functions for:
- Loading SoftAlign model parameters
- Detecting ANARCI chain type from alignment
"""

import logging
from importlib.resources import files
from typing import Any, Dict

import numpy as np
from softalign.utils import convert_numpy_to_jax, unflatten_dict

LOGGER = logging.getLogger(__name__)


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
    """Detect antibody chain type from alignment based on DE loop length.

    The DE loop (IMGT positions 81-84) differs between chain types:
    - Heavy chains have 4 residues (81, 82, 83, 84)
    - Light chains have 2 residues (83, 84 only - skip 81, 82)

    Args:
        alignment: The alignment matrix (rows=sequence, cols=IMGT positions).

    Returns:
        Chain type string for ANARCI: "H" for heavy, "K" for kappa.
    """
    # Check alignment matrix for occupancy at positions 81 and 82
    pos81_col = 80  # 0-indexed column for IMGT position 81
    pos82_col = 81  # 0-indexed column for IMGT position 82
    pos81_occupied = alignment[:, pos81_col].sum() >= 1
    pos82_occupied = alignment[:, pos82_col].sum() >= 1

    if pos81_occupied or pos82_occupied:
        # DE loop has 4 residues -> heavy chain
        LOGGER.info(
            "Detected chain type: H (heavy) based on DE loop "
            "having residues at positions 81 or 82"
        )
        return "H"
    else:
        # DE loop has 2 residues -> light chain (default to kappa)
        LOGGER.info(
            "Detected chain type: K (kappa) based on DE loop "
            "lacking residues at positions 81 and 82"
        )
        return "K"
