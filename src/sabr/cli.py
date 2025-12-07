#!/usr/bin/env python3

import logging
import os

import click
from ANARCI import anarci

from sabr import aln2hmm, edit_pdb, mpnn_embedder, softaligner, util

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
    default=None,
    help="Chain identifier to renumber.",
)
@click.option(
    "-o",
    "--output-pdb",
    "output_pdb",
    required=True,
    type=click.Path(dir_okay=False, writable=True, path_type=str),
    help="Destination PDB file.",
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
    type=click.Choice(["heavy", "light", "auto"], case_sensitive=False),
    default="auto",
    show_default=True,
    help=(
        "Restrict alignment to specific chain type embeddings. "
        "'heavy' searches only heavy chain (H) embeddings, "
        "'light' searches only light chain (K and L) embeddings, "
        "'auto' searches all embeddings and picks the best match."
    ),
)
def main(
    input_pdb: str,
    input_chain: str,
    output_pdb: str,
    numbering_scheme: str,
    overwrite: bool,
    verbose: bool,
    max_residues: int,
    chain_type: str,
) -> None:
    """Run the command-line workflow for renumbering antibody structures."""
    if verbose:
        logging.basicConfig(level=logging.INFO, force=True)
    else:
        logging.basicConfig(level=logging.WARNING, force=True)
    start_msg = (
        f"Starting SAbR CLI with input={input_pdb} "
        f"chain={input_chain} output={output_pdb} "
        f"scheme={numbering_scheme}"
    )
    LOGGER.info(start_msg)
    if os.path.exists(output_pdb) and not overwrite:
        raise click.ClickException(
            f"{output_pdb} exists, rerun with --overwrite to replace it"
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
    # Convert chain_type to filter format for SoftAligner
    chain_type_filter = None if chain_type == "auto" else chain_type

    # Generate MPNN embeddings for the input chain
    embedder = mpnn_embedder.MPNNEmbedder()
    input_data = embedder.embed(input_pdb, input_chain, max_residues)

    # Align embeddings against species references
    soft_aligner = softaligner.SoftAligner()
    out = soft_aligner(
        input_data,
        chain_type=chain_type_filter,
    )
    sv, start, end = aln2hmm.alignment_matrix_to_state_vector(out.alignment)

    subsequence = "-" * start + sequence[start:end]
    LOGGER.info(f">identified_seq (len {len(subsequence)})\n{subsequence}")

    if not out.species:
        raise click.ClickException(
            "SoftAlign did not specify the matched species; "
            "cannot infer heavy/light chain type."
        )
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
        output_pdb,
        start_res,
        end_res,
        alignment_start=start,
        max_residues=max_residues,
    )
    LOGGER.info(f"Finished renumbering; output written to {output_pdb}")


if __name__ == "__main__":
    main()
