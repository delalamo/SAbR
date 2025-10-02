import argparse
from typing import Tuple

import numpy as np

from sabr import softaligner


def aln2sv(alignment: np.ndarray) -> Tuple[Tuple[str, str], str]:
    """
    Convert numpy alignment matrix to HMMer-style state vector
    Parameters
    ----------
    matrix : np.ndarray
        A 2D array where ones define the alignment path between two sequences.

    Returns
    -------
    List[Tuple[Tuple[int, str], int]]
        The state vector as a list of tuples.
    """
    # Find coordinates of the alignment path (the ones)
    path = np.argwhere(alignment == 1)
    if path.size == 0:
        return []

    # Sort path in order (row, col)
    path = sorted(path.tolist())

    state_vector = []
    for (i, j), (i2, j2) in zip(path[:-1], path[1:]):
        di, dj = i2 - i, j2 - j

        # Diagonal step = match
        if di == 1 and dj == 1:
            state_vector.append(((i2, "m"), j2))
        # Horizontal step = insertion in sequence A (x-axis)
        elif di == 1 and dj == 0:
            state_vector.append(((i2, "i"), j))
        # Vertical step = deletion in sequence A (y-axis)
        elif di == 0 and dj == 1:
            state_vector.append(((i, "d"), j2))
        else:
            # Handle jumps > 1 (multiple insertions/deletions)
            if di > 0 and dj > 0 and di == dj:
                for k in range(1, di + 1):
                    state_vector.append(((i + k, "m"), j + k))
            elif di > 0 and dj == 0:
                for k in range(1, di + 1):
                    state_vector.append(((i + k, "i"), j))
            elif di == 0 and dj > 0:
                for _ in range(1, dj + 1):
                    state_vector.append(((i, "d"), None))
            else:
                raise ValueError(f"Unusual step from {(i, j)} to {(i2, j2)}")

    return state_vector


def parse_args() -> argparse.Namespace:
    """
    Parse arguments
    """
    argparser = argparse.ArgumentParser(prog="sabr", description="SAbR CLI")
    argparser = argparse.ArgumentParser(
        "-i", "--input_pdb", required=True, help="input pdb file"
    )
    argparser = argparse.ArgumentParser(
        "-c", "--input_chain", help="input chain", required=True
    )
    argparser = argparse.ArgumentParser(
        "-o", "--output_pdb", help="output pdb file", required=True
    )
    argparser = argparse.ArgumentParser(
        "-n",
        "--numbering_scheme",
        help="numbering scheme, default is imgt",
        default="imgt",
    )
    argparser = argparse.ArgumentParser(
        "-d",
        "--cdr_definition",
        help="cdr definition, default is imgt",
        default="imgt",
    )
    argparser = argparse.ArgumentParser(
        "-v", "--verbose", help="verbose output", action="store_true"
    )
    args = argparser.parse_args()
    return args


def main():

    args = parse_args()
    soft_aligner = softaligner.SoftAligner()
    alignment = soft_aligner(args.input_pdb, args.input_chain)
    state_vector = aln2sv(alignment)
    print(state_vector)

    # check if file ends with pdb
    # transform chain to arrays
    # run softaligner
    # replace residue IDs as appropriate
    # save to output PDB


if __name__ == "__main__":
    main()
