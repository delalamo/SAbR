#!/usr/bin/env python3

import logging
from typing import List, Optional, Tuple

import numpy as np

State = Tuple[Tuple[int, str], Optional[int]]

LOGGER = logging.getLogger(__name__)


def alignment_matrix_to_state_vector(
    matrix: np.ndarray,
) -> Tuple[List[State], int, int]:
    """Return an HMMER-style state vector from a binary alignment matrix."""
    if matrix.ndim != 2:
        raise ValueError("matrix must be 2D")
    LOGGER.info(f"Converting alignment matrix with shape {matrix.shape}")

    mat = np.transpose(matrix)
    path = np.argwhere(mat == 1)
    if path.size == 0:
        raise RuntimeError("Alignment matrix contains no path")

    path = sorted(path.tolist())
    out: List[Tuple[Tuple[int, str], Optional[int]]] = []

    b_end = max(b for b, _ in path)

    for (b, a), (b2, a2) in zip(path[:-1], path[1:]):
        db, da = b2 - b, a2 - a

        # 1) Diagonal steps -> matches
        while db > 0 and da > 0:
            b += 1
            a += 1
            db -= 1
            da -= 1
            out.append(((b, "m"), a - 1))  # report pre-move A index

        # 2) A-only steps -> inserts (emit current A, then advance A)
        while da > 0:
            if b == b_end:
                out.append(((path[-1][0] + 1, "m"), a))
                report_output(out)
                return out, path[0][0], a + 1 + path[0][0]
            out.append(((b + 1, "i"), a))
            a += 1
            da -= 1

        # 3) B-only steps -> deletes
        while db > 0:
            b += 1
            db -= 1
            out.append(((b, "d"), None))

    report_output(out)
    LOGGER.debug(
        "Generated state vector with "
        f"{len(out)} entries, b_start={path[0][0]}, "
        f"a_end={path[-1][1] + path[0][0]}"
    )
    return out, path[0][0], path[-1][1] + path[0][0]


def report_output(
    out: List[Tuple[Tuple[int, str], Optional[int]]],
) -> None:
    """Log each HMM state in ``out`` at INFO level."""
    LOGGER.info(f"Reporting {len(out)} HMM states")
    for idx, st in enumerate(out):
        (seqB, code), seqA = st
        if seqA is None:
            LOGGER.info(f"{idx} (({seqB}, '{code}'), None)")
        else:
            LOGGER.info(f"{idx} (({seqB}, '{code}'), {seqA})")
