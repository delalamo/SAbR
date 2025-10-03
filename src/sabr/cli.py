import argparse
from typing import List, Optional, Tuple

import numpy as np

# np.set_printoptions(threshold=sys.maxsize)
from ANARCI import anarci
from Bio import SeqIO

from sabr import softaligner

State = Tuple[Tuple[int, str], int]


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


def fetch_sequence_from_pdb(pdb_file: str, chain: str) -> str:
    """
    Fetch the sequence from a PDB file for a given chain.
    """
    for record in SeqIO.parse(pdb_file, "pdb-atom"):
        if record.id.endswith(chain):
            return str(record.seq).replace("X", "")
    ids = [r.id for r in SeqIO.parse(pdb_file, "pdb-atom")]
    raise ValueError(f"Chain {chain} not found in {pdb_file} (contains {ids})")


def parse_args() -> argparse.Namespace:
    """
    Parse arguments
    """
    argparser = argparse.ArgumentParser(prog="sabr", description="SAbR CLI")
    argparser.add_argument(
        "-i", "--input_pdb", required=True, help="input pdb file"
    )
    argparser.add_argument(
        "-c", "--input_chain", help="input chain", required=True
    )
    argparser.add_argument(
        "-o", "--output_pdb", help="output pdb file", required=True
    )
    argparser.add_argument(
        "-n",
        "--numbering_scheme",
        help="numbering scheme, default is imgt",
        default="imgt",
    )
    argparser.add_argument(
        "-d",
        "--cdr_definition",
        help="cdr definition, default is imgt",
        default="imgt",
    )
    argparser.add_argument(
        "-v", "--verbose", help="verbose output", action="store_true"
    )
    args = argparser.parse_args()
    return args


def main():

    args = parse_args()
    sequence = fetch_sequence_from_pdb(args.input_pdb, args.input_chain)
    soft_aligner = softaligner.SoftAligner()
    best_match, alignment, query_idxs, match_idxs = soft_aligner(
        args.input_pdb, args.input_chain
    )
    # print(alignment)
    # state_vector = aln2sv(alignment, query_idxs, match_idxs)
    state_vector = alignment_matrix_to_state_vector(np.array(alignment))
    print_state_vector(state_vector)

    anarci_out = anarci.number_sequence_from_alignment(
        state_vector,
        sequence,
        scheme=args.numbering_scheme,
        chain_type=best_match[-1],
    )

    print(len(anarci_out), anarci_out)

    # check if file ends with pdb
    # transform chain to arrays
    # run softaligner
    # replace residue IDs as appropriate
    # save to output PDB


if __name__ == "__main__":
    main()
