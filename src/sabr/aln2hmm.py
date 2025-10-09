import logging
from typing import List, Optional, Tuple

import numpy as np

State = Tuple[Tuple[int, str], Optional[int]]

LOGGER = logging.getLogger(__name__)


def alignment_matrix_to_state_vector(
    matrix: np.ndarray,
) -> Tuple[List[State], int, int]:
    """
    Convert a binary alignment matrix into an HMMER-style state vector.

    The function interprets an alignment path encoded as ones in a binary
    matrix and emits a sequence of HMMER states:
    match ``'m'``, insert ``'i'``, and delete ``'d'``.
    Input matrices are assumed to have rows = SeqA and cols = SeqB; the
    matrix is transposed internally so that iteration proceeds with
    rows = SeqB and cols = SeqA.

    Conventions (0-based indices)
    -----------------------------
    - Diagonal step ``(b, a) -> (b+1, a+1)`` → emit ``((b+1, 'm'), a)``
    - A-only step  ``(b, a) -> (b,   a+1)`` → emit ``((b+1, 'i'), a)``
    - B-only step  ``(b, a) -> (b+1, a)``   → emit ``((b+1, 'd'), None)``

    Premature termination rule
    --------------------------
    If the path reaches the final SeqB row and attempts to continue with
    inserts only (i.e., trailing ``'i'`` beyond the last SeqB row), the
    function emits a final match-like placeholder and returns early. This is
    intended to prevent unbounded insertion runs at the end of the sequence.

    Parameters
    ----------
    matrix : np.ndarray
        Binary alignment matrix of shape ``(len(SeqA), len(SeqB))`` with ones
        along the alignment path.

    Returns
    -------
    tuple[list[State], int, int]
        A 3-tuple ``(states, b_start, a_end)`` where:
        - ``states`` is a list of states of the form
          ``((seqB_index, code), seqA_index_or_None)``;
        - ``b_start`` is the first SeqB index encountered in the path
          (0-based, before the internal +1 used in the state encoding);
        - ``a_end`` is the terminal SeqA index *offset-adjusted* in the same
          manner as the original implementation
          (``path[-1][1] + path[0][0]``).

        When the input has no ones (no path), the function returns
        ``([], 0, 0)``.

    Raises
    ------
    ValueError
        If ``matrix`` is not two-dimensional.

    Notes
    -----
    - The implementation assumes a monotonic alignment path (non-negative
      steps in both axes).
    - The logger will emit a human-readable dump of the produced states at
      ``INFO`` level via :func:`report_output`.
    """
    if matrix.ndim != 2:
        raise ValueError("matrix must be 2D")
    LOGGER.info(f"Converting alignment matrix with shape {matrix.shape}")

    # Treat rows as SeqB and cols as SeqA
    mat = np.transpose(matrix)

    # Coordinates of ones (alignment path), sorted by (SeqB, SeqA)
    path = np.argwhere(mat == 1)
    if path.size == 0:
        raise RuntimeError("Alignment matrix contains no path")

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
            out.append(((b + 1, "i"), a))  # emit CURRENT 'a'
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
    """
    Log a human-readable listing of HMMER state outputs.

    Each entry is printed at ``INFO`` level in the format:
    ``<idx> ((<seqB>, '<code>'), <seqA_or_None>)``.

    Parameters
    ----------
    out : list[State]
        The state vector produced by
        :func:`alignment_matrix_to_state_vector`.
    """
    LOGGER.info(f"Reporting {len(out)} HMM states")
    for idx, st in enumerate(out):
        (seqB, code), seqA = st
        if seqA is None:
            LOGGER.info(f"{idx} (({seqB}, '{code}'), None)")
        else:
            LOGGER.info(f"{idx} (({seqB}, '{code}'), {seqA})")
