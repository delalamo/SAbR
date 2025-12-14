#!/usr/bin/env python3

import logging
import os

import click
from ANARCI import anarci

from sabr import (
    aln2hmm,
    constants,
    edit_pdb,
    mpnn_embeddings,
    softaligner,
    util,
)

LOGGER = logging.getLogger(__name__)


@click.command(
    context_settings={"help_option_names": ["-h", "--help"]},
    help=(
        "Structure-based Antibody Renumbering (SAbR) renumbers antibody PDB "
        "files using the 3D coordinates of backbone atoms."
    ),
)
@click.option(
    "-i",
    "--input-pdb",
    "input_pdb",
    required=True,
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=str),
    help="Input PDB file.",
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
    "-t",
    "--chain-type",
    "chain_type",
    type=click.Choice(
        [ct.value for ct in constants.ChainType], case_sensitive=False
    ),
    default="auto",
    show_default=True,
    help=(
        "Restrict alignment to specific chain type embeddings. "
        "'heavy' searches only heavy chain (H) embeddings, "
        "'light' searches only light chain (K and L) embeddings, "
        "'auto' searches all embeddings and picks the best match."
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
    "--deterministic-loop-renumbering/--no-deterministic-loop-renumbering",
    "deterministic_loop_renumbering",
    default=True,
    show_default=True,
    help=(
        "Enable deterministic renumbering corrections for loop regions. "
        "When enabled (default), applies corrections for: "
        "light chain FR1 positions 7-10, DE loop positions 80-85 (all chains), "
        "and CDR loops (CDR1, CDR2, CDR3). "
        "When disabled, uses raw alignment output without corrections."
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
    chain_type: str,
    extended_insertions: bool,
    deterministic_loop_renumbering: bool,
) -> None:
    """Run the command-line workflow for renumbering antibody structures."""
    if verbose:
        logging.basicConfig(level=logging.INFO, force=True)
    else:
        logging.basicConfig(level=logging.WARNING, force=True)

    # === Input Validation ===

    # Validate input PDB file exists
    if not os.path.exists(input_pdb):
        raise click.ClickException(f"Input file '{input_pdb}' does not exist.")

    # Validate input file has correct extension
    if not input_pdb.lower().endswith(".pdb"):
        raise click.ClickException(
            f"Input file must be a PDB file (.pdb). Got: '{input_pdb}'"
        )

    # Validate chain identifier
    if input_chain and len(input_chain) != 1:
        raise click.ClickException(
            f"Chain identifier must be a single character. Got: '{input_chain}'"
        )

    # Validate output file extension
    valid_output_ext = (".pdb", ".cif")
    if not output_file.lower().endswith(valid_output_ext):
        raise click.ClickException(
            f"Output file must have extension .pdb or .cif. "
            f"Got: '{output_file}'"
        )

    # Validate extended insertions requires mmCIF format
    if extended_insertions and not output_file.endswith(".cif"):
        raise click.ClickException(
            "The --extended-insertions option requires mmCIF output format. "
            "Please use a .cif file extension for the output file."
        )

    # Validate max_residues is non-negative
    if max_residues < 0:
        raise click.ClickException(
            f"max_residues must be non-negative. Got: {max_residues}"
        )

    # === End Input Validation ===

    start_msg = (
        f"Starting SAbR CLI with input={input_pdb} "
        f"chain={input_chain} output={output_file} "
        f"scheme={numbering_scheme}"
    )
    if extended_insertions:
        start_msg += " (extended insertion codes enabled)"
    LOGGER.info(start_msg)
    if os.path.exists(output_file) and not overwrite:
        raise click.ClickException(
            f"{output_file} exists, rerun with --overwrite to replace it"
        )
    sequence = util.fetch_sequence_from_pdb(input_pdb, input_chain)
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
    # Convert chain_type string to enum
    chain_type_enum = constants.ChainType(chain_type)
    chain_type_filter = (
        None if chain_type_enum == constants.ChainType.AUTO else chain_type_enum
    )

    # Generate MPNN embeddings for the input chain
    input_data = mpnn_embeddings.from_pdb(input_pdb, input_chain, max_residues)

    # Align embeddings against species references
    soft_aligner = softaligner.SoftAligner()
    out = soft_aligner(
        input_data,
        chain_type=chain_type_filter,
        deterministic_loop_renumbering=deterministic_loop_renumbering,
    )
    sv, start, end = aln2hmm.alignment_matrix_to_state_vector(out.alignment)

    subsequence = "-" * start + sequence[start:end]
    LOGGER.info(f">identified_seq (len {len(subsequence)})\n{subsequence}")

    if not out.species:
        raise click.ClickException(
            "SoftAlign did not specify the matched species; "
            "cannot infer heavy/light chain type."
        )

    # TODO introduce extended insertion code handling here
    # Revert to default ANARCI behavior if extended_insertions is False
    anarci_out, start_res, end_res = anarci.number_sequence_from_alignment(
        sv,
        subsequence,
        scheme=numbering_scheme,
        chain_type=out.species[-1],
    )

    anarci_out = [a for a in anarci_out if a[1] != "-"]

    edit_pdb.thread_alignment(
        input_pdb,
        input_chain,
        anarci_out,
        output_file,
        start_res,
        end_res,
        alignment_start=start,
        max_residues=max_residues,
    )
    LOGGER.info(f"Finished renumbering; output written to {output_file}")


if __name__ == "__main__":
    main()
