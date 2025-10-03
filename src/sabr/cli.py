import argparse
import sys
from typing import List, Tuple

from ANARCI import anarci
from Bio import SeqIO

from sabr import aln2hmm, edit_pdb, softaligner

State = Tuple[Tuple[int, str], int]


def verify_output(anarci_out: List[Tuple[State, str]], sequence: str) -> None:
    """
    Verify that the ANARCI output matches the input sequence.
    """
    anarci_seq = "".join([a_entry for (_, a_entry) in anarci_out]).replace(
        "-", ""
    )
    if anarci_seq != sequence:
        raise ValueError(
            f"ANARCI output sequence does not match input sequence! "
            f"({anarci_seq} != {sequence})"
        )


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
        "-t",
        "--trim",
        help="Remove regions outside V-region",
        action="store_true",
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
    best_match, alignment = soft_aligner(args.input_pdb, args.input_chain)
    state_vector = aln2hmm.alignment_matrix_to_state_vector(alignment)

    print(args.numbering_scheme)
    anarci_out, start_res, end_res = anarci.number_sequence_from_alignment(
        state_vector,
        sequence,
        scheme=args.numbering_scheme,
        chain_type=best_match[-1],
    )

    anarci_out = [a for a in anarci_out if a[1] != "-"]
    if len(anarci_out) != len(sequence):
        raise ValueError(
            (
                f"ANARCI output length does not match sequence length!"
                f" ({len(anarci_out)} != {len(sequence)})"
            )
        )
    print(anarci_out)
    edit_pdb.thread_alignment(
        args.input_pdb,
        args.input_chain,
        anarci_out,
        args.output_pdb,
        start_res,
        end_res,
    )

    sys.exit(0)

    # check if file ends with pdb
    # transform chain to arrays
    # run softaligner
    # replace residue IDs as appropriate
    # save to output PDB


if __name__ == "__main__":
    main()
