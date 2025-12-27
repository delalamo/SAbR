#!/usr/bin/env python3
"""Command-line interface for SAbR antibody renumbering.

This module provides the CLI entry point for the SAbR (Structure-based
Antibody Renumbering) tool. It orchestrates the full renumbering pipeline:

1. Load structure (PDB or mmCIF format) and extract sequence
2. Generate MPNN embeddings for the target chain
3. Align embeddings against species references using SoftAlign
4. Convert alignment to HMM state vector
5. Apply ANARCI numbering scheme (IMGT, Chothia, Kabat, etc.)
6. Write renumbered structure to output file

Usage:
    sabr -i input.pdb -c A -o output.pdb -n imgt
    sabr -i input.cif -c A -o output.cif -n imgt
"""

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
        "CDR loops (CDR1, CDR2, CDR3), and C-terminus positions 126-128. "
        "When disabled, uses raw alignment output without corrections."
    ),
)
@click.option(
    "-s",
    "--anarci-species",
    "anarci_species",
    type=click.Choice(
        [s.value for s in constants.AnarciSpecies], case_sensitive=False
    ),
    default="human",
    show_default=True,
    help=(
        "Species for ANARCI numbering. This parameter is passed to ANARCI "
        "and affects germline gene identification. Choose from: human, mouse, "
        "rat, rabbit, pig, rhesus, alpaca."
    ),
)
@click.option(
    "-a",
    "--anarci-chain-type",
    "anarci_chain_type",
    type=click.Choice(
        [ct.value for ct in constants.AnarciChainType], case_sensitive=False
    ),
    default="auto",
    show_default=True,
    help=(
        "Chain type for ANARCI numbering. 'H' for heavy chain, 'K' for kappa "
        "light chain, 'L' for lambda light chain. 'auto' will detect based on "
        "DE loop length: heavy chains have 4 residues at positions 81-84, "
        "while light chains have 2 residues at positions 83-84 only. "
        "When 'auto' is used, heavy is selected if DE loop is 4 residues, "
        "kappa is selected if DE loop is 2 residues."
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
    anarci_species: str,
    anarci_chain_type: str,
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
    valid_input_ext = (".pdb", ".cif")
    if not input_pdb.lower().endswith(valid_input_ext):
        raise click.ClickException(
            f"Input file must be a PDB (.pdb) or mmCIF (.cif) file. "
            f"Got: '{input_pdb}'"
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

    # Convert chain_type string to enum
    chain_type_enum = constants.ChainType(chain_type)
    chain_type_filter = (
        None if chain_type_enum == constants.ChainType.AUTO else chain_type_enum
    )

    # Generate MPNN embeddings for the input chain (also extracts sequence)
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

    # Align embeddings against species references
    soft_aligner = softaligner.SoftAligner()
    out = soft_aligner(
        input_data,
        chain_type=chain_type_filter,
        deterministic_loop_renumbering=deterministic_loop_renumbering,
    )
    sv, start, end, first_aligned_row = (
        aln2hmm.alignment_matrix_to_state_vector(out.alignment)
    )

    # Create subsequence with leading dashes for missing IMGT positions
    # start = first IMGT column (0-indexed), used for leading dashes
    # end - start = number of aligned residues
    n_aligned = end - start
    subsequence = "-" * start + sequence[:n_aligned]
    LOGGER.info(f">identified_seq (len {len(subsequence)})\n{subsequence}")

    if not out.species:
        raise click.ClickException(
            "SoftAlign did not specify the matched species; "
            "cannot infer heavy/light chain type."
        )

    # Determine ANARCI chain type
    anarci_chain_type_enum = constants.AnarciChainType(anarci_chain_type)
    if anarci_chain_type_enum == constants.AnarciChainType.AUTO:
        # Auto-detect based on DE loop length (positions 81-84)
        # Heavy chains have 4 residues (81, 82, 83, 84)
        # Light chains have 2 residues (83, 84 only - skip 81, 82)
        # Check alignment matrix for occupancy at positions 81 and 82
        pos81_col = 80  # 0-indexed column for IMGT position 81
        pos82_col = 81  # 0-indexed column for IMGT position 82
        pos81_occupied = out.alignment[:, pos81_col].sum() >= 1
        pos82_occupied = out.alignment[:, pos82_col].sum() >= 1

        if pos81_occupied or pos82_occupied:
            # DE loop has 4 residues -> heavy chain
            resolved_chain_type = "H"
            LOGGER.info(
                "Auto-detected chain type: H (heavy) based on DE loop "
                "having residues at positions 81 or 82"
            )
        else:
            # DE loop has 2 residues -> light chain (default to kappa)
            resolved_chain_type = "K"
            LOGGER.info(
                "Auto-detected chain type: K (kappa) based on DE loop "
                "lacking residues at positions 81 and 82"
            )
    else:
        resolved_chain_type = anarci_chain_type_enum.value
        LOGGER.info(f"Using user-specified ANARCI chain type: {resolved_chain_type}")

    # Log the species parameter (informational - not currently used by ANARCI)
    anarci_species_enum = constants.AnarciSpecies(anarci_species)
    LOGGER.info(f"ANARCI species: {anarci_species_enum.value}")

    # TODO introduce extended insertion code handling here
    # Revert to default ANARCI behavior if extended_insertions is False
    anarci_out, start_res, end_res = anarci.number_sequence_from_alignment(
        sv,
        subsequence,
        scheme=numbering_scheme,
        chain_type=resolved_chain_type,
    )

    anarci_out = [a for a in anarci_out if a[1] != "-"]

    # After filtering, the ANARCI output starts at index 0, not start_res
    # Reset start_res and end_res to match the filtered output
    filtered_start_res = 0
    filtered_end_res = len(anarci_out)

    edit_pdb.thread_alignment(
        input_pdb,
        input_chain,
        anarci_out,
        output_file,
        filtered_start_res,
        filtered_end_res,
        alignment_start=first_aligned_row,
        max_residues=max_residues,
    )
    LOGGER.info(f"Finished renumbering; output written to {output_file}")


if __name__ == "__main__":
    main()
