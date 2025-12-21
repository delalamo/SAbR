#!/usr/bin/env python3
"""Utility functions for SAbR.

This module provides helper functions for:
- Loading SoftAlign model parameters
"""

import logging
import pickle
from importlib.resources import files
from typing import Any, Dict

LOGGER = logging.getLogger(__name__)


def read_softalign_params(
    params_name: str = "CONT_SW_05_T_3_1",
    params_path: str = "softalign.models",
) -> Dict[str, Any]:
    """Load SoftAlign parameters from package resources.

    Args:
        params_name: Name of the model parameters file.
        params_path: Package path containing the parameters file.

    Returns:
        Dictionary containing the model parameters.
    """
    path = files(params_path) / params_name
    with open(path, "rb") as f:
        params = pickle.load(f)
    LOGGER.info(f"Loaded model parameters from {path}")
    return params
