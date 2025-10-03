from typing import List, Optional, Tuple

import numpy as np

State = Tuple[Tuple[int, str], Optional[int]]


def alignment_matrix_to_state_vector(
    matrix: np.ndarray,
) -> List[Tuple[Tuple[int, str], Optional[int]]]:
    """
    Convert a binary alignment matrix into an HMMER state vector (m, i, d),
    returning entries as ((seqB_index, code), seqA_index).

    Assumes the input matrix has rows=SeqA (x) and cols=SeqB (y).
    We transpose internally so rows=SeqB, cols=SeqA.

    Conventions (0-based):
      - Diagonal (b,a)->(b+1,a+1): emit ((b+1,'m'), a)
      - A-only  (b,a)->(b,a+1):   emit ((b+1,'i'), a)
      - B-only  (b,a)->(b+1,a):   emit ((b+1,'d'), None)

    Parameters
    ----------
    matrix : np.ndarray
        2D binary numpy array with 1s indicating alignment path.

    Returns
    -------
    List[Tuple[Tuple[int, str], Optional[int]]]
        List of state vector entries as ((seqB_index, code), seqA_index).
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

    # Track the last MATCH position emitted in post-move coordinates (b+1, a+1)
    last_emitted_match_post: Optional[Tuple[int, int]] = None

    for (b, a), (b2, a2) in zip(path[:-1], path[1:]):
        db, da = b2 - b, a2 - a

        # 1) Diagonal steps -> matches
        while db > 0 and da > 0:
            b += 1
            a += 1
            db -= 1
            da -= 1
            out.append(((b, "m"), a - 1))  # report pre-move A index
            last_emitted_match_post = (b, a)  # post-move (b, a)

        # 2) A-only steps -> inserts (emit current A, then advance A)
        while da > 0:
            out.append(((b + 1, "i"), a))  # emit CURRENT 'a'
            a += 1
            da -= 1

        # 3) B-only steps -> deletes
        while db > 0:
            b += 1
            db -= 1
            out.append(((b, "d"), None))

    # ---- Explicitly handle the terminal node (b_last, a_last) ----
    b_last, a_last = path[-1]
    terminal_match_post = (b_last + 1, a_last + 1)
    if last_emitted_match_post != terminal_match_post:
        # Emit the final match row representing arrival at the terminal node
        out.append(((b_last + 1, "m"), a_last))

    for idx, st in enumerate(out):
        (seqB, code), seqA = st
        if seqA is None:
            print(f"{idx} (({seqB}, '{code}'), None)")
        else:
            print(f"{idx} (({seqB}, '{code}'), {seqA})")

    return out


def print_state_vector(
    states: List[State], header: str = "STATE_VECTOR"
) -> None:
    print(header)
    for idx, st in enumerate(states):
        # st is ((seqB, code), seqA_or_None)
        (seqB, code), seqA = st
        # match your exact quoting/None style
        if seqA is None:
            print(f"{idx} (({seqB}, '{code}'), None)")
        else:
            print(f"{idx} (({seqB}, '{code}'), {seqA})")
