#!/usr/bin/env python3

import csv
import logging
import os
import warnings
from pathlib import Path

import click
from ANARCI import anarci
from Bio import BiopythonWarning, SeqIO

from sabr import aln2hmm, edit_pdb, softaligner

LOGGER = logging.getLogger(__name__)


def fetch_sequence_from_pdb(pdb_file: str, chain: str) -> str:
    """Return the sequence for chain in pdb_file without X residues."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BiopythonWarning)
        for record in SeqIO.parse(pdb_file, "pdb-atom"):
            if record.id.endswith(chain):
                return str(record.seq).replace("X", "")
    ids = [r.id for r in SeqIO.parse(pdb_file, "pdb-atom")]
    raise ValueError(f"Chain {chain} not found in {pdb_file} (contains {ids})")


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
    "input_path",
    required=True,
    type=click.Path(exists=True, readable=True, path_type=str),
    help="Input PDB file or directory containing PDB files.",
)
@click.option(
    "-c",
    "--input-chain",
    "input_chain",
    required=True,
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
    "--deviations-only",
    is_flag=True,
    help="Report residue numbering deviations without writing output.",
)
@click.option(
    "--deviations-csv",
    type=click.Path(dir_okay=False, writable=True, path_type=str),
    default=None,
    help=(
        "Optional path to write deviations as CSV "
        "(old/new residue IDs per row)."
    ),
)
def main(
    input_path: str,
    input_chain: str,
    output_pdb: str,
    numbering_scheme: str,
    overwrite: bool,
    verbose: bool,
    deviations_only: bool,
    deviations_csv: str | None,
) -> None:
    """Run the command-line workflow for renumbering antibody structures."""
    if verbose:
        logging.basicConfig(level=logging.INFO, force=True)
    else:
        logging.basicConfig(level=logging.WARNING, force=True)
    input_files = gather_input_files(input_path)
    start_msg = (
        f"Starting SAbR CLI with inputs={','.join(input_files)} "
        f"chain={input_chain} output={output_pdb} "
        f"scheme={numbering_scheme}"
    )
    LOGGER.info(start_msg)
    if len(input_files) > 1 and not deviations_only:
        raise click.ClickException(
            "Provide --deviations-only when processing multiple inputs; "
            "use one invocation per output otherwise."
        )

    soft_aligner = softaligner.SoftAligner()
    results: list[
        tuple[str, list[tuple[tuple[str, int, str], tuple[str, int, str]]]]
    ] = []
    for input_pdb in input_files:
        if (
            os.path.exists(output_pdb)
            and not overwrite
            and len(input_files) == 1
            and not deviations_only
        ):
            raise click.ClickException(
                f"{output_pdb} exists, rerun with --overwrite to replace it"
            )
        try:
            sequence = fetch_sequence_from_pdb(input_pdb, input_chain)
            LOGGER.info(f">input_seq (len {len(sequence)})\n{sequence}")
            LOGGER.info(
                f"Fetched sequence of length {len(sequence)} from "
                f"{input_pdb} chain {input_chain}"
            )
            out = soft_aligner(input_pdb, input_chain)
            sv, start, end = aln2hmm.alignment_matrix_to_state_vector(
                out.alignment
            )

            subsequence = "-" * start + sequence[start:end]
            LOGGER.info(
                f">identified_seq (len {len(subsequence)})\n{subsequence}"
            )

            if not out.species:
                raise click.ClickException(
                    "SoftAlign did not specify the matched species; "
                    "cannot infer heavy/light chain type."
                )
            anarci_out, start_res, end_res = (
                anarci.number_sequence_from_alignment(
                    sv,
                    subsequence,
                    scheme=numbering_scheme,
                    chain_type=out.species[-1],
                )
            )

            anarci_out = [a for a in anarci_out if a[1] != "-"]

            deviations = edit_pdb.thread_alignment(
                input_pdb,
                input_chain,
                anarci_out,
                output_pdb if len(input_files) == 1 else "",
                start_res,
                end_res,
                alignment_start=start,
                write_output=(len(input_files) == 1) and not deviations_only,
            )
            print(input_pdb, len(deviations))
            results.append((input_pdb, deviations))
            if not deviations_only:
                LOGGER.info(f"Finished renumbering {input_pdb} to {output_pdb}")
        except Exception:
            continue

    if deviations_csv:
        write_deviations_csv(deviations_csv, results)

    if deviations_only:
        if len(results) == 1:
            click.echo(results[0][1])
        else:
            click.echo(dict(results))
        return


def write_deviations_csv(
    csv_path: str,
    results: list[
        tuple[str, list[tuple[tuple[str, int, str], tuple[str, int, str]]]]
    ],
) -> None:
    """Write deviations summary to ``csv_path``."""
    with open(csv_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        for pdb_path, deviations in results:
            formatted = " ".join(
                _format_residue_id(old_id) for old_id, _new_id in deviations
            )
            writer.writerow([pdb_path, len(deviations), formatted])
    LOGGER.info(
        "Wrote deviation summary for %d inputs to %s", len(results), csv_path
    )


def _format_residue_id(res_id: tuple[str, int, str]) -> str:
    """Return ``resseq`` with optional insertion code appended."""
    resseq = res_id[1]
    icode = res_id[2].strip()
    return f"{resseq}{icode}"


def gather_input_files(path: str) -> list[str]:
    """Return files to process given a file or directory path."""
    path_obj = Path(path)
    if path_obj.is_file():
        return [str(path_obj)]
    if not path_obj.is_dir():
        raise click.ClickException(f"Path {path} is neither file nor directory")
    allowed = {".pdb", ".cif"}
    files = sorted(
        p
        for p in path_obj.iterdir()
        if p.is_file() and p.suffix.lower() in allowed
    )
    if not files:
        raise click.ClickException(f"No files found in directory {path}")
    return [str(f) for f in files]


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        LOGGER.warning("Process interrupted by user, exiting...")
        exit(1)
