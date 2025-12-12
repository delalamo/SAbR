#!/usr/bin/env python3
"""Create NPZ embedding files from PDB structures.

This script generates MPNN embeddings from PDB files and saves them as NPZ files
for downstream processing. The embeddings can be used for renumbering benchmarks
or other analyses.

The script supports two modes:
1. Directory mode: Process all PDB files in a directory with a single chain
2. CSV mode: Process specific PDB files and chains from a CSV file
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

from sabr import mpnn_embeddings

logging.basicConfig(
    level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s"
)
LOGGER = logging.getLogger(__name__)


def process_from_directory(args):
    """Process PDB files from a directory."""
    pdb_dir = Path(args.pdb_dir)
    output_dir = Path(args.output_dir)

    # Validate input directory
    if not pdb_dir.exists():
        LOGGER.error(f"PDB directory not found: {pdb_dir}")
        sys.exit(1)

    if not pdb_dir.is_dir():
        LOGGER.error(f"PDB path is not a directory: {pdb_dir}")
        sys.exit(1)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info(f"Output directory: {output_dir}")

    # Find PDB files
    pdb_files = sorted(pdb_dir.glob(args.pattern))
    LOGGER.info(f"Found {len(pdb_files)} PDB files matching '{args.pattern}'")

    if len(pdb_files) == 0:
        LOGGER.error(f"No PDB files found in {pdb_dir}")
        sys.exit(1)

    # Build list of (pdb_file, chain) tuples
    tasks = [(pdb_file, args.chain) for pdb_file in pdb_files]

    return tasks, output_dir


def process_from_csv(args):
    """Process PDB files from a CSV file."""
    csv_path = Path(args.csv)
    pdb_dir = Path(args.pdb_dir)
    output_dir = Path(args.output_dir)

    # Validate CSV file
    if not csv_path.exists():
        LOGGER.error(f"CSV file not found: {csv_path}")
        sys.exit(1)

    # Validate PDB directory
    if not pdb_dir.exists():
        LOGGER.error(f"PDB directory not found: {pdb_dir}")
        sys.exit(1)

    if not pdb_dir.is_dir():
        LOGGER.error(f"PDB path is not a directory: {pdb_dir}")
        sys.exit(1)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info(f"Output directory: {output_dir}")

    # Read CSV file
    tasks = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)

        # Validate CSV has required columns
        if (
            "pdb_id" not in reader.fieldnames
            or "chain" not in reader.fieldnames
        ):
            LOGGER.error(
                "CSV file must have 'pdb_id' and 'chain' columns. "
                f"Found: {reader.fieldnames}"
            )
            sys.exit(1)

        for row in reader:
            pdb_id = row["pdb_id"].lower()
            chain = row["chain"]

            # Find PDB file (try different extensions)
            pdb_file = None
            for ext in [".pdb", ".ent", ".pdb.gz"]:
                candidate = pdb_dir / f"{pdb_id}{ext}"
                if candidate.exists():
                    pdb_file = candidate
                    break

            if pdb_file is None:
                LOGGER.warning(
                    f"PDB file not found for {pdb_id} (chain {chain}), skipping"
                )
                continue

            tasks.append((pdb_file, chain))

    LOGGER.info(f"Found {len(tasks)} entries from CSV")

    if len(tasks) == 0:
        LOGGER.error("No valid PDB files found from CSV")
        sys.exit(1)

    return tasks, output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Create NPZ embedding files from PDB structures",
        epilog="""
Examples:
  # Directory mode: process all PDB files with chain A
  %(prog)s --pdb-dir ./pdbs --output-dir ./embeddings --chain A

  # CSV mode: process specific PDB/chain combinations from CSV
  %(prog)s --pdb-dir ./pdbs --output-dir ./embeddings --csv antibodies.csv
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pdb-dir",
        type=str,
        required=True,
        help="Directory containing PDB files to process",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where NPZ files will be saved",
    )

    # CSV mode arguments
    parser.add_argument(
        "--csv",
        type=str,
        help=(
            "CSV file with 'pdb_id' and 'chain' columns. "
            "If provided, overrides --chain and --pattern options."
        ),
    )

    # Directory mode arguments
    parser.add_argument(
        "--chain",
        type=str,
        default="A",
        help="Chain identifier to embed (default: A, ignored if --csv is used)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.pdb",
        help=(
            "Glob pattern for PDB files "
            "(default: *.pdb, ignored if --csv is used)"
        ),
    )

    # Common arguments
    parser.add_argument(
        "--max-residues",
        type=int,
        default=0,
        help="Maximum residues to process (0 = no limit, default: 0)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files if NPZ already exists",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed output for each file",
    )

    args = parser.parse_args()

    # Determine mode and get tasks
    if args.csv:
        LOGGER.info("CSV mode: processing PDB/chain pairs from CSV file")
        tasks, output_dir = process_from_csv(args)
    else:
        LOGGER.info("Directory mode: processing all PDB files in directory")
        tasks, output_dir = process_from_directory(args)

    # Process each task
    processed = 0
    skipped = 0
    failed = 0

    for pdb_file, chain in tasks:
        # Generate output filename
        # For CSV mode: pdb_id_chain.npz (e.g., 1abc_H.npz)
        # For directory mode: original_filename.npz
        if args.csv:
            pdb_id = pdb_file.stem.split(".")[0]  # Remove .pdb or .ent
            base_name = f"{pdb_id}_{chain}"
        else:
            base_name = pdb_file.stem

        output_npz = output_dir / f"{base_name}.npz"

        # Check if already exists
        if args.skip_existing and output_npz.exists():
            if args.verbose:
                LOGGER.info(
                    f"Skipping {pdb_file.name} chain {chain} "
                    "(NPZ already exists)"
                )
            skipped += 1
            continue

        try:
            LOGGER.info(f"Processing {pdb_file.name} (chain {chain})...")

            # Generate embeddings
            embeddings = mpnn_embeddings.from_pdb(
                str(pdb_file),
                chain=chain,
                max_residues=args.max_residues,
            )

            # Save to NPZ
            embeddings.save(str(output_npz))

            if args.verbose:
                LOGGER.info(f"  Saved to: {output_npz.name}")
                LOGGER.info(f"  Shape: {embeddings.embeddings.shape}")
                LOGGER.info(f"  Residues: {embeddings.embeddings.shape[0]}")
                seq_len = (
                    len(embeddings.sequence) if embeddings.sequence else "N/A"
                )
                LOGGER.info(f"  Sequence length: {seq_len}")
            else:
                LOGGER.info(
                    f"  → {output_npz.name} "
                    f"({embeddings.embeddings.shape[0]} residues)"
                )

            processed += 1

        except Exception as e:
            LOGGER.error(
                f"  Failed to process {pdb_file.name} chain {chain}: {e}"
            )
            if args.verbose:
                import traceback

                traceback.print_exc()
            failed += 1
            continue

    # Summary
    LOGGER.info("")
    LOGGER.info("=" * 70)
    LOGGER.info("SUMMARY")
    LOGGER.info("=" * 70)
    LOGGER.info(f"Total tasks: {len(tasks)}")
    LOGGER.info(f"Successfully processed: {processed}")
    LOGGER.info(f"Skipped (already exist): {skipped}")
    LOGGER.info(f"Failed: {failed}")
    LOGGER.info(f"Output directory: {output_dir}")

    # Exit with error code if any failed
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
