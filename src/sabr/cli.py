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
import random
from typing import Optional, Tuple

import click

from sabr import (
    edit_pdb,
    mpnn_embeddings,
    options,
    renumber,
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
    "--residue-range",
    "residue_range",
    nargs=2,
    type=int,
    default=(0, 0),
    help=(
        "Range of residues to process as START END in PDB numbering "
        "(inclusive). Use '0 0' (default) to process all residues. "
        "Example: --residue-range 1 120 processes residues 1-120."
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
    "--random-seed",
    "random_seed",
    type=int,
    default=None,
    help=(
        "Random seed for JAX operations. If not specified, a random seed "
        "will be generated. Set this for reproducible results."
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
    residue_range: Tuple[int, int],
    extended_insertions: bool,
    disable_deterministic_renumbering: bool,
    random_seed: Optional[int],
    chain_type: str,
) -> None:
    """Run the command-line workflow for renumbering antibody structures."""
    util.configure_logging(verbose)
    options.validate_inputs(
        input_pdb,
        input_chain,
        output_file,
        residue_range,
        extended_insertions,
        overwrite,
    )

    # Generate random seed if not specified
    if random_seed is None:
        random_seed = random.randint(0, 2**31 - 1)
        LOGGER.info(f"Generated random seed: {random_seed}")
    else:
        LOGGER.info(f"Using specified random seed: {random_seed}")

    start_msg = (
        f"Starting SAbR CLI with input={input_pdb} "
        f"chain={input_chain} output={output_file} "
        f"scheme={numbering_scheme}"
    )
    if extended_insertions:
        start_msg += " (extended insertion codes enabled)"
    LOGGER.info(start_msg)

    input_data = mpnn_embeddings.from_pdb(
        input_pdb,
        input_chain,
        residue_range=residue_range,
        random_seed=random_seed,
    )
    sequence = input_data.sequence

    LOGGER.info(f">input_seq (len {len(sequence)})\n{sequence}")
    if residue_range != (0, 0):
        LOGGER.info(
            f"Processing residues {residue_range[0]}-{residue_range[1]} "
            f"(residue_range flag)"
        )
    LOGGER.info(
        f"Fetched sequence of length {len(sequence)} from "
        f"{input_pdb} chain {input_chain}"
    )

    # Use shared renumbering pipeline
    use_deterministic = not disable_deterministic_renumbering
    anarci_out, detected_chain_type, first_aligned_row = (
        renumber.run_renumbering_pipeline(
            input_data,
            numbering_scheme=numbering_scheme,
            chain_type=chain_type,
            deterministic_loop_renumbering=use_deterministic,
        )
    )

    edit_pdb.thread_alignment(
        input_pdb,
        input_chain,
        anarci_out,
        output_file,
        0,
        len(anarci_out),
        alignment_start=first_aligned_row,
        residue_range=residue_range,
    )
    LOGGER.info(f"Finished renumbering; output written to {output_file}")


if __name__ == "__main__":
    main()
