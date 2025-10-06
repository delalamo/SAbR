import logging
from typing import List, Optional, Tuple

import numpy as np

State = Tuple[Tuple[int, str], Optional[int]]

LOGGER = logging.getLogger(__name__)


def alignment_matrix_to_state_vector(
    matrix: np.ndarray,
) -> Tuple[List[State], int, int]:
    """
    WARNING: This code was written in large part by ChatGPT. Use with caution.

    Convert a binary alignment matrix into an HMMER state vector (m, i, d),
    returning entries as ((seqB_index, code), seqA_index).

    Assumes the input matrix has rows=SeqA (x) and cols=SeqB (y).
    We transpose internally so rows=SeqB, cols=SeqA.

    Conventions (0-based):
      - Diagonal (b,a)->(b+1,a+1): emit ((b+1,'m'), a)
      - A-only  (b,a)->(b,a+1):   emit ((b+1,'i'), a)
      - B-only  (b,a)->(b+1,a):   emit ((b+1,'d'), None)

    Premature termination rule:
      If the alignment hits the end of SeqB and attempts to continue with
      insertions ('i') only (i.e., trailing inserts beyond the final SeqB row),
      raise RuntimeError to abort early.
    """
    if matrix.ndim != 2:
        raise ValueError("matrix must be 2D")

    # Treat rows as SeqB and cols as SeqA
    mat = matrix.T

    # Coordinates of ones (alignment path), sorted by (SeqB, SeqA)
    path = np.argwhere(mat == 1)
    if path.size == 0:
        return []

    path = sorted(path.tolist())  # [(b, a), ...]
    out: List[Tuple[Tuple[int, str], Optional[int]]] = []

    # Determine the final SeqB row present in the alignment path.
    # Any attempt to emit 'i' while b == b_end implies trailing
    # inserts past end of SeqB.
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
            # Premature termination condition: we've reached final SeqB row,
            # and continuing would produce trailing insertions beyond SeqB
            if b == b_end:
                out.append(((path[-1][0] + 1, "m"), a))
                report_output(out)
                return out, path[0][0], a + 1 + path[0][0]
                # raise RuntimeError(
                #     f"Trailing insertions beyond end of SeqB detected "
                #     f"(b_end={b_end}, starting at A index {a}). Aborting."
                # )
            out.append(((b + 1, "i"), a))  # emit CURRENT 'a'
            a += 1
            da -= 1

        # 3) B-only steps -> deletes
        while db > 0:
            b += 1
            db -= 1
            out.append(((b, "d"), None))

    report_output(out)
    return out, path[0][0], path[-1][1] + path[0][0]


def report_output(
    out: List[Tuple[Tuple[int, str], Optional[int]]],
) -> None:
    for idx, st in enumerate(out):
        (seqB, code), seqA = st
        if seqA is None:
            LOGGER.info(f"{idx} (({seqB}, '{code}'), None)")
        else:
            LOGGER.info(f"{idx} (({seqB}, '{code}'), {seqA})")
