#!/usr/bin/env python3
"""Command-line interface for SAbR antibody renumbering.

This module provides the CLI entry point for the SAbR (Structure-based
Antibody Renumbering) tool. It orchestrates the full renumbering pipeline:

1. Load structure (PDB or mmCIF format) and extract sequence
2. Generate MPNN embeddings for the target chain
3. Align embeddings against unified reference using SoftAlign
4. Convert alignment to HMM state vector
5. Apply ANARCI numbering scheme (IMGT, Chothia, Kabat, etc.)
6. Write renumbered structure to output file

Usage:
    sabr -i input.pdb -c A -o output.pdb -n imgt
    sabr -i input.cif -c A -o output.cif -n imgt
"""

import logging

import click
from ANARCI import anarci

from sabr import (
    aln2hmm,
    edit_pdb,
    mpnn_embeddings,
    options,
    softaligner,
    util,
)

LOGGER = logging.getLogger(__name__)


@click.command(
    context_settings={"help_option_names": ["-h", "--help"]},
    help=(
        "Structure-based Antibody Renumbering (SAbR) renumbers antibody "
        "structure files using the 3D coordinates of backbone atoms. "
        "Supports both PDB and mmCIF input formats."
    ),
)
@click.option(
    "-i",
    "--input-pdb",
    "input_pdb",
    required=True,
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=str),
    help="Input structure file (PDB or mmCIF format).",
)
@click.option(
    "-c",
    "--input-chain",
    "input_chain",
    required=True,
    callback=lambda ctx, _, value: (
        value
        if len(value) == 1
        else ctx.fail("Chain identifier must be exactly one character.")
    ),
    help="Chain identifier to renumber (single character).",
)
@click.option(
    "-o",
    "--output",
    "output_file",
    required=True,
    type=click.Path(dir_okay=False, writable=True, path_type=str),
    help=(
        "Destination structure file. Use .pdb extension for PDB format "
        "or .cif extension for mmCIF format. mmCIF is required when using "
        "--extended-insertions."
    ),
)
@click.option(
    "-n",
    "--numbering-scheme",
    "numbering_scheme",
    default="imgt",
    show_default="IMGT",
    type=click.Choice(
        ["imgt", "chothia", "kabat", "martin", "aho", "wolfguy"],
        case_sensitive=False,
    ),
    help="Numbering scheme.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Overwrite the output PDB if it already exists.",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Enable verbose logging.",
)
@click.option(
    "--max-residues",
    "max_residues",
    type=int,
    default=0,
    help=(
        "Maximum number of residues to process from the chain. "
        "If 0 (default), process all residues."
    ),
)
@click.option(
    "--extended-insertions",
    "extended_insertions",
    is_flag=True,
    help=(
        "Enable extended insertion codes (AA, AB, ..., ZZ, AAA, etc.) "
        "for antibodies with very long CDR loops. Requires mmCIF output "
        "format (.cif extension). Standard PDB format only supports "
        "single-character insertion codes (A-Z, max 26 insertions per position)"
    ),
)
@click.option(
    "--disable-deterministic-renumbering",
    "disable_deterministic_renumbering",
    is_flag=True,
    help=(
        "Disable deterministic renumbering corrections for loop regions. "
        "By default, corrections are applied for: "
        "light chain FR1 positions 7-10, DE loop positions 80-85 (all chains), "
        "and CDR loops (CDR1, CDR2, CDR3). "
        "Use this flag to use raw alignment output without corrections."
    ),
)
@click.option(
    "-t",
    "--chain-type",
    "chain_type",
    default="auto",
    show_default=True,
    type=click.Choice(
        ["H", "K", "L", "heavy", "kappa", "lambda", "auto"],
        case_sensitive=False,
    ),
    callback=lambda ctx, param, value: options.normalize_chain_type(value),
    help=(
        "Chain type for ANARCI numbering. H/heavy=heavy chain, K/kappa=kappa "
        "light, L/lambda=lambda light. Use 'auto' (default) to detect from "
        "DE loop occupancy."
    ),
)
def main(
    input_pdb: str,
    input_chain: str,
    output_file: str,
    numbering_scheme: str,
    overwrite: bool,
    verbose: bool,
    max_residues: int,
    extended_insertions: bool,
    disable_deterministic_renumbering: bool,
    chain_type: str,
) -> None:
    """Run the command-line workflow for renumbering antibody structures."""
    util.configure_logging(verbose)
    options.validate_inputs(
        input_pdb,
        input_chain,
        output_file,
        max_residues,
        extended_insertions,
        overwrite,
    )

    start_msg = (
        f"Starting SAbR CLI with input={input_pdb} "
        f"chain={input_chain} output={output_file} "
        f"scheme={numbering_scheme}"
    )
    if extended_insertions:
        start_msg += " (extended insertion codes enabled)"
    LOGGER.info(start_msg)

    input_data = mpnn_embeddings.from_pdb(input_pdb, input_chain, max_residues)
    sequence = input_data.sequence

    LOGGER.info(f">input_seq (len {len(sequence)})\n{sequence}")
    if max_residues > 0:
        LOGGER.info(
            f"Will truncate output to {max_residues} residues "
            f"(max_residues flag)"
        )
    LOGGER.info(
        f"Fetched sequence of length {len(sequence)} from "
        f"{input_pdb} chain {input_chain}"
    )

    aligner = softaligner.SoftAligner()
    alignment_result = aligner(
        input_data,
        deterministic_loop_renumbering=not disable_deterministic_renumbering,
    )
    state_vector, imgt_start, imgt_end, first_aligned_row = (
        aln2hmm.alignment_matrix_to_state_vector(alignment_result.alignment)
    )

    n_aligned = imgt_end - imgt_start
    subsequence = "-" * imgt_start + sequence[:n_aligned]
    LOGGER.info(f">identified_seq (len {len(subsequence)})\n{subsequence}")

    # Detect chain type from DE loop for ANARCI numbering if not specified
    if chain_type == "auto":
        chain_type = util.detect_chain_type(alignment_result.alignment)
    else:
        LOGGER.info(f"Using user-specified chain type: {chain_type}")

    # TODO introduce extended insertion code handling here
    anarci_out, start_res, end_res = anarci.number_sequence_from_alignment(
        state_vector,
        subsequence,
        scheme=numbering_scheme,
        chain_type=chain_type,
    )

    anarci_out = [a for a in anarci_out if a[1] != "-"]

    edit_pdb.thread_alignment(
        input_pdb,
        input_chain,
        anarci_out,
        output_file,
        0,
        len(anarci_out),
        alignment_start=first_aligned_row,
        max_residues=max_residues,
    )
    LOGGER.info(f"Finished renumbering; output written to {output_file}")


if __name__ == "__main__":
    main()
