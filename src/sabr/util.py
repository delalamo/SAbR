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
