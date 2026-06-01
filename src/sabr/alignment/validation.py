"""Validation helpers for alignment matrices."""

from __future__ import annotations

import numpy as np

from sabr.errors import AlignmentError


def validate_alignment_matrix(matrix: np.ndarray) -> None:
    """Validate basic alignment matrix invariants."""
    if matrix.ndim != 2:
        raise AlignmentError(f"Alignment matrix must be 2D, got {matrix.shape}.")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise AlignmentError(f"Alignment matrix must be non-empty, got {matrix.shape}.")
    if not np.issubdtype(matrix.dtype, np.number) and matrix.dtype != np.bool_:
        raise AlignmentError(
            f"Alignment matrix must contain numeric or bool values, got {matrix.dtype}."
        )
    if not np.isfinite(matrix).all():
        raise AlignmentError("Alignment matrix contains NaN or infinite values.")

    rounded = np.round(matrix)
    if not np.isin(rounded, [0, 1]).all():
        raise AlignmentError("Alignment matrix values must round to binary 0/1.")
    if not rounded.any():
        raise AlignmentError(
            f"Alignment matrix contains no path, shape={matrix.shape}."
        )

    row_sums = rounded.sum(axis=1)
    duplicate_rows = np.where(row_sums > 1)[0]
    if len(duplicate_rows):
        raise AlignmentError(
            "Alignment matrix assigns multiple reference positions to query "
            f"rows: {duplicate_rows.tolist()}."
        )

    path = np.argwhere(rounded == 1)
    if len(path) > 1:
        cols_by_row = path[np.argsort(path[:, 0]), 1]
        if np.any(np.diff(cols_by_row) < 0):
            raise AlignmentError("Alignment matrix path must be monotonic.")
