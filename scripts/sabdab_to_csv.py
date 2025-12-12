#!/usr/bin/env python3
"""Generate a CSV file from SAbDab summary TSV for embedding creation.

This script processes the SAbDab summary file (downloadable from
http://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/summary/all/)
and creates a CSV file specifying which PDB files and chains to process
for embedding generation.

The SAbDab summary file contains comprehensive information about all antibody
structures in the PDB, including chain identifiers for heavy and light chains.
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s"
)
LOGGER = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Generate CSV from SAbDab summary TSV file",
        epilog="""
The SAbDab summary file can be downloaded from:
http://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/summary/all/

This script extracts PDB IDs and chain identifiers to create a CSV
file that can be used with create_embeddings.py --csv option.
        """,
    )
    parser.add_argument(
        "--tsv",
        type=str,
        required=True,
        help="Path to SAbDab summary TSV file (e.g., sabdab_summary_all.tsv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output CSV file path",
    )
    parser.add_argument(
        "--chain-type",
        type=str,
        choices=["heavy", "light", "both", "all"],
        default="both",
        help=(
            "Which chains to include: 'heavy' (H chains only), "
            "'light' (L chains only), 'both' (H and L chains), "
            "'all' (all antibody chains including nanobodies) (default: both)"
        ),
    )
    parser.add_argument(
        "--max-entries",
        type=int,
        default=0,
        help="Maximum number of entries to include (0 = all, default: 0)",
    )
    parser.add_argument(
        "--filter-resolution",
        type=float,
        default=0,
        help=(
            "Only include structures with resolution <= this value "
            "(0 = no filter, default: 0)"
        ),
    )
    parser.add_argument(
        "--filter-method",
        type=str,
        choices=["X-RAY DIFFRACTION", "ELECTRON MICROSCOPY", "any"],
        default="any",
        help="Filter by experimental method (default: any)",
    )
    parser.add_argument(
        "--no-scfv",
        action="store_true",
        help="Exclude scFv structures",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed output",
    )

    args = parser.parse_args()

    # Validate input file
    tsv_path = Path(args.tsv)
    if not tsv_path.exists():
        LOGGER.error(f"TSV file not found: {tsv_path}")
        sys.exit(1)

    # Open TSV file
    entries = []
    skipped_resolution = 0
    skipped_method = 0
    skipped_scfv = 0
    skipped_no_chain = 0

    LOGGER.info(f"Reading SAbDab summary file: {tsv_path}")

    with open(tsv_path, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            pdb_id = row["pdb"].lower()
            h_chain = row["Hchain"]
            l_chain = row["Lchain"]
            resolution = row["resolution"]
            method = row["method"]
            scfv = row["scfv"]

            # Apply filters
            # Resolution filter
            if args.filter_resolution > 0:
                try:
                    res_val = float(resolution)
                    if res_val > args.filter_resolution:
                        skipped_resolution += 1
                        continue
                except (ValueError, TypeError):
                    # Skip if resolution is not a valid number
                    skipped_resolution += 1
                    continue

            # Method filter
            if args.filter_method != "any":
                if method != args.filter_method:
                    skipped_method += 1
                    continue

            # scFv filter
            if args.no_scfv and scfv == "True":
                skipped_scfv += 1
                continue

            # Determine which chains to include based on chain_type
            chains_to_add = []

            if args.chain_type in ["heavy", "both", "all"]:
                if h_chain and h_chain != "NA":
                    chains_to_add.append(h_chain)

            if args.chain_type in ["light", "both"]:
                if l_chain and l_chain != "NA":
                    chains_to_add.append(l_chain)

            # For 'all', we also include any chains not in H/L
            # (e.g., nanobodies which might have other chain IDs)
            if args.chain_type == "all":
                if l_chain and l_chain != "NA":
                    chains_to_add.append(l_chain)

            # Skip if no chains to add
            if not chains_to_add:
                skipped_no_chain += 1
                continue

            # Add entries for each chain
            for chain in chains_to_add:
                entries.append(
                    {
                        "pdb_id": pdb_id,
                        "chain": chain,
                        "resolution": resolution,
                        "method": method,
                    }
                )

                if args.verbose:
                    LOGGER.info(
                        f"Added: {pdb_id} chain {chain} "
                        f"({resolution} Å, {method})"
                    )

            # Check max entries limit
            if args.max_entries > 0 and len(entries) >= args.max_entries:
                LOGGER.info(
                    f"Reached maximum entries limit: {args.max_entries}"
                )
                break

    # Write CSV file
    output_path = Path(args.output)
    LOGGER.info(f"Writing CSV to: {output_path}")

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pdb_id", "chain"])  # Header

        for entry in entries:
            writer.writerow([entry["pdb_id"], entry["chain"]])

    # Summary
    LOGGER.info("")
    LOGGER.info("=" * 70)
    LOGGER.info("SUMMARY")
    LOGGER.info("=" * 70)
    LOGGER.info(f"Total entries written: {len(entries)}")
    LOGGER.info(f"Skipped (resolution): {skipped_resolution}")
    LOGGER.info(f"Skipped (method): {skipped_method}")
    LOGGER.info(f"Skipped (scFv): {skipped_scfv}")
    LOGGER.info(f"Skipped (no chains): {skipped_no_chain}")
    LOGGER.info(f"Output file: {output_path}")


if __name__ == "__main__":
    main()
