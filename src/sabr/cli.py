import argparse
import logging
import sys

from ANARCI import anarci
from Bio import SeqIO

from sabr import aln2hmm, edit_pdb, softaligner


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
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)
    sequence = fetch_sequence_from_pdb(args.input_pdb, args.input_chain)
    soft_aligner = softaligner.SoftAligner()
    best_match, alignment = soft_aligner(args.input_pdb, args.input_chain)
    state_vector = aln2hmm.alignment_matrix_to_state_vector(alignment)

    anarci_out, start_res, end_res = anarci.number_sequence_from_alignment(
        state_vector,
        sequence,
        scheme=args.numbering_scheme,
        chain_type=best_match[-1],
    )

    anarci_out = [a for a in anarci_out if a[1] != "-"]

    edit_pdb.thread_alignment(
        args.input_pdb,
        args.input_chain,
        anarci_out,
        args.output_pdb,
        start_res,
        end_res,
    )

    sys.exit(0)


if __name__ == "__main__":
    main()
