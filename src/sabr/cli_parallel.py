#!/usr/bin/env python3
"""
Parallelized batch processing CLI for SAbR.

This script provides optimized parallel processing with two levels:
1. Species alignment parallelization (within each chain)
2. Multi-file/chain parallelization (across multiple inputs)

Key optimizations:
- Parallel species alignment using ThreadPoolExecutor
- ProcessPoolExecutor for multi-file processing
- Efficient memory usage with lazy loading
- Progress tracking and error reporting
"""

import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

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


@dataclass
class ProcessingJob:
    """Represents a single chain renumbering job."""

    input_pdb: str
    input_chain: str
    output_file: str
    numbering_scheme: str
    chain_type: str
    max_residues: int
    num_workers: int
    deterministic_loop_renumbering: bool


@dataclass
class ProcessingResult:
    """Result of processing a single chain."""

    job: ProcessingJob
    success: bool
    error_msg: Optional[str] = None


def process_single_chain(job: ProcessingJob) -> ProcessingResult:
    """
    Process a single antibody chain with parallelized species alignment.

    This function is designed to be run in a separate process for multi-file
    parallelization. It creates its own SoftAligner instance with the specified
    number of workers for parallel species alignment.
    """
    try:
        # Get sequence from PDB
        sequence = util.fetch_sequence_from_pdb(job.input_pdb, job.input_chain)
        LOGGER.info(
            f"Processing {job.input_pdb} chain {job.input_chain} "
            f"(len={len(sequence)})"
        )

        # Convert chain_type to enum
        chain_type_enum = constants.ChainType(job.chain_type)
        chain_type_filter = (
            None
            if chain_type_enum == constants.ChainType.AUTO
            else chain_type_enum
        )

        # Generate MPNN embeddings
        input_data = mpnn_embeddings.from_pdb(
            job.input_pdb, job.input_chain, job.max_residues
        )

        # Create SoftAligner with parallel species alignment
        soft_aligner = softaligner.SoftAligner(num_workers=job.num_workers)

        # Align embeddings against species references (parallelized internally)
        out = soft_aligner(
            input_data,
            chain_type=chain_type_filter,
            deterministic_loop_renumbering=job.deterministic_loop_renumbering,
        )

        # Convert alignment to state vector
        sv, start, end = aln2hmm.alignment_matrix_to_state_vector(out.alignment)
        subsequence = "-" * start + sequence[start:end]

        if not out.species:
            return ProcessingResult(
                job=job,
                success=False,
                error_msg="No species match found",
            )

        # Run ANARCI numbering
        anarci_out, start_res, end_res = anarci.number_sequence_from_alignment(
            sv,
            subsequence,
            scheme=job.numbering_scheme,
            chain_type=out.species[-1],
        )
        anarci_out = [a for a in anarci_out if a[1] != "-"]

        # Thread alignment onto PDB
        edit_pdb.thread_alignment(
            job.input_pdb,
            job.input_chain,
            anarci_out,
            job.output_file,
            start_res,
            end_res,
            alignment_start=start,
            max_residues=job.max_residues,
        )

        LOGGER.info(f"Successfully processed {job.output_file}")
        return ProcessingResult(job=job, success=True)

    except Exception as e:
        LOGGER.error(
            f"Error processing {job.input_pdb} chain {job.input_chain}: {e}"
        )
        return ProcessingResult(job=job, success=False, error_msg=str(e))


def parse_input_file(input_file: str) -> List[tuple]:
    """
    Parse input file containing PDB paths and chain IDs.

    Format: Each line should be "pdb_path,chain_id" or "pdb_path chain_id"
    Lines starting with # are ignored.
    """
    jobs = []
    with open(input_file) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Support both comma and whitespace separators
            if "," in line:
                parts = [p.strip() for p in line.split(",")]
            else:
                parts = line.split()

            if len(parts) < 2:
                LOGGER.warning(
                    f"Line {line_num}: Expected 'pdb_path,chain_id', "
                    f"got '{line}'. Skipping."
                )
                continue

            pdb_path, chain_id = parts[0], parts[1]
            if not os.path.exists(pdb_path):
                LOGGER.warning(
                    f"Line {line_num}: PDB file not found: {pdb_path}. "
                    "Skipping."
                )
                continue

            jobs.append((pdb_path, chain_id))

    return jobs


@click.command(
    context_settings={"help_option_names": ["-h", "--help"]},
    help=(
        "Parallelized batch processing for SAbR. "
        "Process multiple antibody chains with two levels of parallelization: "
        "(1) parallel species alignment within each chain, and "
        "(2) parallel processing across multiple files/chains."
    ),
)
@click.option(
    "-i",
    "--input",
    "input_source",
    required=True,
    type=click.Path(exists=True, readable=True, path_type=str),
    help=(
        "Input source. Can be: (1) a single PDB file (requires -c), "
        "(2) a directory of PDB files (requires -c), or "
        "(3) a text file listing 'pdb_path,chain_id' pairs (one per line)."
    ),
)
@click.option(
    "-c",
    "--chains",
    "chains",
    default=None,
    help=(
        "Chain identifier(s) to process. Can be comma-separated for multiple "
        "chains (e.g., 'H,L'). Required when input is a PDB file or directory."
    ),
)
@click.option(
    "-o",
    "--output-dir",
    "output_dir",
    required=True,
    type=click.Path(file_okay=False, path_type=str),
    help="Output directory for renumbered PDB files.",
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
    help="Overwrite existing output files.",
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
        "Maximum number of residues to process from each chain. "
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
    "-j",
    "--jobs",
    "num_jobs",
    type=int,
    default=1,
    show_default=True,
    help=(
        "Number of parallel jobs for processing multiple files/chains. "
        "Use 1 for sequential, >1 for parallel across multiple inputs."
    ),
)
@click.option(
    "-w",
    "--species-workers",
    "species_workers",
    type=int,
    default=1,
    show_default=True,
    help=(
        "Number of parallel workers for species alignment within each chain. "
        "Use 0 for sequential species alignment, >0 for parallel alignment. "
        "Higher values speed up alignment but use more memory."
    ),
)
@click.option(
    "--output-suffix",
    "output_suffix",
    default="_renumbered",
    show_default=True,
    help="Suffix to add to output filenames before extension.",
)
@click.option(
    "--deterministic-loop-renumbering/--no-deterministic-loop-renumbering",
    "deterministic_loop_renumbering",
    default=True,
    show_default=True,
    help=(
        "Enable deterministic renumbering corrections for loop regions. "
        "When enabled (default), applies corrections for FR1, DE, CDR loops."
    ),
)
def main(
    input_source: str,
    chains: Optional[str],
    output_dir: str,
    numbering_scheme: str,
    overwrite: bool,
    verbose: bool,
    max_residues: int,
    chain_type: str,
    num_jobs: int,
    species_workers: int,
    output_suffix: str,
    deterministic_loop_renumbering: bool,
) -> None:
    """Run parallelized batch processing for antibody structure renumbering."""
    # Setup logging
    if verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            force=True,
        )
    else:
        logging.basicConfig(
            level=logging.WARNING,
            format="%(asctime)s - %(levelname)s - %(message)s",
            force=True,
        )

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Collect jobs
    jobs: List[ProcessingJob] = []

    if os.path.isfile(input_source):
        # Check if it's a PDB file or a job list
        if input_source.lower().endswith((".pdb", ".ent", ".cif")):
            # Single PDB file
            if not chains:
                raise click.ClickException(
                    "When input is a PDB file, specify chain(s) with -c"
                )
            chain_list = [c.strip() for c in chains.split(",")]
            pdb_name = Path(input_source).stem

            for chain_id in chain_list:
                output_file = (
                    output_path / f"{pdb_name}_{chain_id}{output_suffix}.pdb"
                )
                if output_file.exists() and not overwrite:
                    LOGGER.warning(
                        f"Skipping {output_file} (exists, use --overwrite)"
                    )
                    continue

                det_loop = deterministic_loop_renumbering
                jobs.append(
                    ProcessingJob(
                        input_pdb=input_source,
                        input_chain=chain_id,
                        output_file=str(output_file),
                        numbering_scheme=numbering_scheme,
                        chain_type=chain_type,
                        max_residues=max_residues,
                        num_workers=species_workers,
                        deterministic_loop_renumbering=det_loop,
                    )
                )
        else:
            # Job list file
            parsed_jobs = parse_input_file(input_source)
            for pdb_path, chain_id in parsed_jobs:
                pdb_name = Path(pdb_path).stem
                output_file = (
                    output_path / f"{pdb_name}_{chain_id}{output_suffix}.pdb"
                )
                if output_file.exists() and not overwrite:
                    LOGGER.warning(
                        f"Skipping {output_file} (exists, use --overwrite)"
                    )
                    continue

                det_loop = deterministic_loop_renumbering
                jobs.append(
                    ProcessingJob(
                        input_pdb=pdb_path,
                        input_chain=chain_id,
                        output_file=str(output_file),
                        numbering_scheme=numbering_scheme,
                        chain_type=chain_type,
                        max_residues=max_residues,
                        num_workers=species_workers,
                        deterministic_loop_renumbering=det_loop,
                    )
                )
    elif os.path.isdir(input_source):
        # Directory of PDB files
        if not chains:
            raise click.ClickException(
                "When input is a directory, you must specify chain(s) with -c"
            )
        chain_list = [c.strip() for c in chains.split(",")]

        pdb_files = list(Path(input_source).glob("*.pdb"))
        pdb_files.extend(Path(input_source).glob("*.ent"))

        for pdb_file in sorted(pdb_files):
            pdb_name = pdb_file.stem
            for chain_id in chain_list:
                output_file = (
                    output_path / f"{pdb_name}_{chain_id}{output_suffix}.pdb"
                )
                if output_file.exists() and not overwrite:
                    LOGGER.warning(
                        f"Skipping {output_file} (exists, use --overwrite)"
                    )
                    continue

                det_loop = deterministic_loop_renumbering
                jobs.append(
                    ProcessingJob(
                        input_pdb=str(pdb_file),
                        input_chain=chain_id,
                        output_file=str(output_file),
                        numbering_scheme=numbering_scheme,
                        chain_type=chain_type,
                        max_residues=max_residues,
                        num_workers=species_workers,
                        deterministic_loop_renumbering=det_loop,
                    )
                )
    else:
        raise click.ClickException(f"Input not found: {input_source}")

    if not jobs:
        click.echo("No jobs to process.")
        return

    click.echo(f"Processing {len(jobs)} chain(s)...")
    click.echo(
        f"Parallelization: {num_jobs} file workers, "
        f"{species_workers} species workers per file"
    )

    # Process jobs
    results: List[ProcessingResult] = []

    if num_jobs == 1:
        # Sequential file processing (but parallel species alignment)
        click.echo(
            "Using sequential file processing with parallel species alignment"
        )
        for i, job in enumerate(jobs, 1):
            click.echo(
                f"Processing {i}/{len(jobs)}: "
                f"{job.input_pdb} chain {job.input_chain}"
            )
            result = process_single_chain(job)
            results.append(result)
            if not result.success:
                click.echo(f"  ERROR: {result.error_msg}", err=True)
    else:
        # Parallel file processing (each with parallel species alignment)
        click.echo(
            f"Using {num_jobs} parallel workers for file processing, "
            f"{species_workers} species workers each"
        )
        with ProcessPoolExecutor(max_workers=num_jobs) as executor:
            futures = {
                executor.submit(process_single_chain, job): job for job in jobs
            }

            with click.progressbar(
                length=len(futures), label="Processing"
            ) as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    pbar.update(1)
                    if not result.success:
                        click.echo(
                            f"\nFailed: {result.job.input_pdb} chain "
                            f"{result.job.input_chain}: {result.error_msg}",
                            err=True,
                        )

    # Summary
    successful = sum(1 for r in results if r.success)
    failed = len(results) - successful

    click.echo(f"\nCompleted: {successful} successful, {failed} failed")

    if failed > 0:
        click.echo("\nFailed jobs:")
        for r in results:
            if not r.success:
                click.echo(
                    f"  {r.job.input_pdb} chain {r.job.input_chain}: "
                    f"{r.error_msg}"
                )


if __name__ == "__main__":
    main()
