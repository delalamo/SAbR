"""Shared validation helpers."""

from __future__ import annotations

import numpy as np


def validate_array_shape(
    array: np.ndarray,
    dim: int,
    expected_size: int,
    array_name: str,
    size_name: str,
    context: str = "",
) -> None:
    """Validate that an array dimension matches an expected size."""
    actual_size = array.shape[dim]
    if actual_size != expected_size:
        msg = (
            f"{array_name}.shape[{dim}] ({actual_size}) must match "
            f"{size_name} ({expected_size})."
        )
        if context:
            msg += f" {context}"
        raise ValueError(msg)
