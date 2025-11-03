#!/usr/bin/env python3

import argparse
import logging
import os
import sys

from ANARCI import anarci
from Bio import SeqIO

from sabr import aln2hmm, edit_pdb, softaligner

LOGGER = logging.getLogger(__name__)


def fetch_sequence_from_pdb(pdb_file: str, chain: str) -> str:
    """Return the sequence for chain in pdb_file without X residues."""
    for record in SeqIO.parse(pdb_file, "pdb-atom"):
        if record.id.endswith(chain):
            return str(record.seq).replace("X", "")
    ids = [r.id for r in SeqIO.parse(pdb_file, "pdb-atom")]
    raise ValueError(f"Chain {chain} not found in {pdb_file} (contains {ids})")


def parse_args() -> argparse.Namespace:
    """Return parsed CLI arguments for the SAbR entry point."""
    description = (
        "Structure-based Antibody Renumbering (SAbR) renumbers antibody "
        "PDB files using the 3D coordinate of backbone atoms."
    )
    argparser = argparse.ArgumentParser(prog="sabr", description=description)
    argparser.add_argument(
        "-i", "--input_pdb", required=True, help="Input pdb file"
    )
    argparser.add_argument(
        "-c", "--input_chain", help="Input chain", required=True
    )
    argparser.add_argument(
        "-o", "--output_pdb", help="Output pdb file", required=True
    )
    argparser.add_argument(
        "-n",
        "--numbering_scheme",
        help=(
            "Numbering scheme, default is IMGT. Supports IMGT, Chothia, "
            "Kabat, Martin, AHo, and Wolfguy."
        ),
        default="imgt",
    )
    argparser.add_argument(
        "--overwrite", help="Overwrite output PDB", action="store_true"
    )
    argparser.add_argument(
        "-v", "--verbose", help="Verbose output", action="store_true"
    )
    args = argparser.parse_args()
    return args


def main():
    """Run the command-line workflow for renumbering antibody structures."""
    args = parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO, force=True)
    else:
        logging.basicConfig(level=logging.WARNING, force=True)
    start_msg = (
        f"Starting SAbR CLI with input={args.input_pdb} "
        f"chain={args.input_chain} output={args.output_pdb} "
        f"scheme={args.numbering_scheme}"
    )
    LOGGER.info(start_msg)
    if os.path.exists(args.output_pdb) and not args.overwrite:
        raise RuntimeError(
            f"Error: {args.output_pdb} exists, use --overwrite to overwrite"
        )
    sequence = fetch_sequence_from_pdb(args.input_pdb, args.input_chain)
    LOGGER.info(f">input_seq (len {len(sequence)})\n{sequence}")
    LOGGER.info(
        f"Fetched sequence of length {len(sequence)} from "
        f"{args.input_pdb} chain {args.input_chain}"
    )
    soft_aligner = softaligner.SoftAligner()
    out = soft_aligner(args.input_pdb, args.input_chain)
    sv, start, end = aln2hmm.alignment_matrix_to_state_vector(out.alignment)

    subsequence = "-" * start + sequence[start:end]
    LOGGER.info(f">identified_seq (len {len(subsequence)})\n{subsequence}")

    anarci_out, start_res, end_res = anarci.number_sequence_from_alignment(
        sv,
        subsequence,
        scheme=args.numbering_scheme,
        chain_type=out.name[-1],
    )

    anarci_out = [a for a in anarci_out if a[1] != "-"]

    edit_pdb.thread_alignment(
        args.input_pdb,
        args.input_chain,
        anarci_out,
        args.output_pdb,
        start_res,
        end_res,
        alignment_start=start,
    )
    LOGGER.info(f"Finished renumbering; output written to {args.output_pdb}")

    sys.exit(0)


if __name__ == "__main__":
    main()
