#!/usr/bin/env python3
"""SAbDab Analysis Script.

This script evaluates SAbR numbering accuracy on IMGT-numbered SAbDab
structures. It reads a CSV file with PDB IDs and chains, optionally fetches
structures from SAbDab or uses local files, runs SAbR pipeline, and compares
output to expected IMGT numbering.

Usage:
    # Fetch from SAbDab
    python pre2021_analysis.py --csv pdb_chains.csv

    # Use local PDB files ({pdb-dir}/{chain_type}/{pdb}_{chain}.pdb)
    python pre2021_analysis.py --csv pdb_chains.csv --pdb-dir ./testset
"""

import argparse
import csv
import json
import sys
import tempfile
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import requests
from Bio.PDB import PDBIO, PDBParser, Select

from sabr.cli import renumber
from sabr.constants import IMGT_REGIONS
from sabr.embeddings.mpnn import from_pdb

# Suppress all warnings
warnings.filterwarnings("ignore")


def fetch_imgt_pdb(pdb_id: str, output_path: str, max_retries: int = 3) -> None:
    """Fetch IMGT-numbered PDB from SAbDab.

    Args:
        pdb_id: 4-letter PDB ID
        output_path: Path to save the PDB file
        max_retries: Maximum number of retry attempts

    Raises:
        RuntimeError: If the PDB cannot be fetched after all retries
    """
    base_url = "https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/pdb"
    url = f"{base_url}/{pdb_id}/?scheme=imgt"

    # Use browser-like headers to avoid being blocked by the server
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
    }

    last_error = None
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()

            content = response.text
            # Verify it's a valid PDB (not an error page)
            # Check for REMARK (SAbDab header) or ATOM records
            if (
                "REMARK" not in content[:500]
                and "ATOM" not in content[:1000]
                and "HEADER" not in content[:500]
            ):
                raise ValueError(
                    f"Invalid PDB content for {pdb_id}: "
                    f"response does not contain REMARK, ATOM or HEADER. "
                    f"First 200 chars: {content[:200]}"
                )

            with open(output_path, "w") as f:
                f.write(content)
            return

        except (requests.exceptions.RequestException, ValueError) as e:
            last_error = e
            print(
                f"Attempt {attempt + 1}/{max_retries} failed for {pdb_id}: {e}"
            )
            if attempt < max_retries - 1:
                time.sleep(2**attempt)  # Exponential backoff: 1s, 2s, 4s

    raise RuntimeError(
        f"Failed to fetch PDB {pdb_id} from {url} "
        f"after {max_retries} attempts. Last error: {last_error}"
    )


def check_sabdab_available() -> None:
    """Check if SAbDab server is available.

    Raises:
        SystemExit: If the server is unavailable after retries
    """
    base_url = "https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get(base_url, headers=headers, timeout=30)
            if response.status_code < 500:
                print(
                    f"SAbDab server is available (HTTP {response.status_code})"
                )
                return
        except requests.exceptions.RequestException as e:
            print(
                f"Server check attempt {attempt + 1}/{max_retries} failed: {e}"
            )
            if attempt < max_retries - 1:
                time.sleep(2**attempt)

    print("ERROR: SAbDab server is unavailable. Exiting.")
    sys.exit(1)


class ChainSelect(Select):
    """BioPython Select class to filter by chain and optionally by residue."""

    def __init__(self, chain_id: str, filter_to_imgt: bool = False):
        self.chain_id = chain_id
        self.filter_to_imgt = filter_to_imgt

    def accept_chain(self, chain):
        return chain.id == self.chain_id

    def accept_residue(self, residue):
        if not self.filter_to_imgt:
            return True
        resnum = residue.get_id()[1]
        return 1 <= resnum <= 128


def extract_chain_to_pdb(
    src_pdb: str, chain_id: str, dst_pdb: str, filter_to_imgt: bool = False
) -> bool:
    """Extract a specific chain from a PDB file."""
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("input", src_pdb)

        # Check if chain exists
        model = structure[0]
        if chain_id not in [c.id for c in model]:
            return False

        io = PDBIO()
        io.set_structure(structure)
        io.save(dst_pdb, ChainSelect(chain_id, filter_to_imgt))

        # Verify output has content
        with open(dst_pdb, "r") as f:
            content = f.read()
            if "ATOM" not in content:
                return False

        return True
    except Exception as e:
        print(f"Error extracting chain: {e}")
        return False


def run_sabr_pipeline(pdb_path: str, chain_id: str) -> Dict:
    """Run full SAbR pipeline on a PDB file.

    Returns:
        Dict with input_positions, output_positions, chain_type, sequence.

    Raises:
        RuntimeError: If SAbR pipeline fails
        ValueError: If structural gaps are detected in the PDB
    """
    # Extract embeddings (also provides input residue IDs)
    input_data = from_pdb(pdb_path, chain_id)

    # Skip structures with backbone gaps in IMGT range (positions 1-128)
    if input_data.gap_indices:
        gaps_in_imgt_range = []
        for gap_idx in input_data.gap_indices:
            residue_id = input_data.idxs[gap_idx]
            try:
                residue_num = int("".join(c for c in residue_id if c.isdigit()))
                if 1 <= residue_num <= 128:
                    gaps_in_imgt_range.append(residue_id)
            except ValueError:
                pass
        if gaps_in_imgt_range:
            raise ValueError(
                f"Structural gaps detected within IMGT range (1-128) "
                f"at positions: {sorted(gaps_in_imgt_range)}"
            )

    # Run the renumbering pipeline (handles alignment and ANARCI)
    anarci_out, chain_type, _ = renumber.run_renumbering_pipeline(
        input_data,
        numbering_scheme="imgt",
        chain_type="auto",
        deterministic_loop_renumbering=True,
    )

    # Parse output positions from ANARCI alignment
    output_positions = []
    for pos, _aa in anarci_out:
        resnum = pos[0]
        insertion = pos[1].strip() if pos[1].strip() else ""
        output_positions.append(f"{resnum}{insertion}")

    return {
        "input_positions": input_data.idxs,
        "output_positions": output_positions,
        "chain_type": chain_type,
        "sequence": input_data.sequence,
    }


def get_region_for_position(pos_num: int) -> str:
    """Get IMGT region name for a position number."""
    for reg_name, positions in IMGT_REGIONS.items():
        if pos_num in positions:
            return reg_name
    return "unknown"


def _position_in_imgt_range(pos: str) -> bool:
    """Check if a position string is within IMGT range 1-128."""
    try:
        pos_num = int("".join(c for c in pos if c.isdigit()))
        return 1 <= pos_num <= 128
    except ValueError:
        return False


def compare_positions(
    input_positions: List[str], output_positions: List[str]
) -> Dict:
    """Compare input and output IMGT positions.

    Only compares residues with positions 1-128 (IMGT variable region).

    Returns:
        Dict with deviations categorized by region.
    """
    # Filter to IMGT positions 1-128 only
    input_filtered = [p for p in input_positions if _position_in_imgt_range(p)]
    output_filtered = [
        p for p in output_positions if _position_in_imgt_range(p)
    ]

    deviations = defaultdict(list)
    perfect = True

    min_len = min(len(input_filtered), len(output_filtered))

    for i in range(min_len):
        inp = input_filtered[i]
        out = output_filtered[i]

        if inp != out:
            perfect = False
            # Determine region based on output position
            try:
                out_num = int("".join(c for c in out if c.isdigit()))
                region = get_region_for_position(out_num)
                deviations[region].append((i, inp, out))
            except ValueError:
                deviations["unknown"].append((i, inp, out))

    # Length mismatch
    if len(input_filtered) != len(output_filtered):
        perfect = False
        deviations["length_mismatch"].append(
            (
                f"input={len(input_filtered)}",
                f"output={len(output_filtered)}",
            )
        )

    return {
        "perfect": perfect,
        "deviations": dict(deviations),
        "n_deviations": sum(len(v) for v in deviations.values()),
    }


def load_csv(csv_path: str) -> List[Dict]:
    """Load PDB entries from CSV file."""
    entries = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append(
                {
                    "pdb_id": row["pdb_id"],
                    "chain": row["chain"],
                    "chain_type": row["chain_type"],
                }
            )
    return entries


def main():
    parser = argparse.ArgumentParser(description="SAbDab Analysis")
    parser.add_argument(
        "--csv", required=True, help="Path to CSV file with PDB IDs and chains"
    )
    parser.add_argument(
        "--output", default="analysis_results.json", help="Output JSON file"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of entries to process (0 = all)",
    )
    parser.add_argument(
        "--cache-dir", default=None, help="Directory to cache downloaded PDBs"
    )
    parser.add_argument(
        "--pdb-dir",
        default=None,
        help="Local PDB dir ({dir}/{chain_type}/{pdb}_{chain}.pdb)",
    )
    parser.add_argument(
        "--filter-to-imgt",
        action="store_true",
        help="Filter residues to IMGT positions 1-128 only",
    )
    args = parser.parse_args()

    # Load entries from CSV
    entries = load_csv(args.csv)
    print(f"Loaded {len(entries)} entries from {args.csv}")

    if args.limit > 0:
        entries = entries[: args.limit]
        print(f"Limited to {len(entries)} entries")

    # Group by chain type
    by_type = defaultdict(list)
    for entry in entries:
        by_type[entry["chain_type"]].append(entry)

    # Set up directories
    pdb_dir = Path(args.pdb_dir) if args.pdb_dir else None
    cache_dir = (
        Path(args.cache_dir) if args.cache_dir else Path(tempfile.mkdtemp())
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    if pdb_dir:
        print(f"Using local PDB directory: {pdb_dir}")
    else:
        print(f"Fetching from SAbDab, cache directory: {cache_dir}")
        # Check if SAbDab is available before starting
        check_sabdab_available()

    # Results
    results = {"heavy": [], "kappa": [], "lambda": []}

    for chain_type in ["heavy", "kappa", "lambda"]:
        type_entries = by_type.get(chain_type, [])
        if not type_entries:
            continue

        print(f"\nProcessing {len(type_entries)} {chain_type} chains...")

        for i, entry in enumerate(type_entries):
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{len(type_entries)}")

            pdb_id = entry["pdb_id"]
            chain_id = entry["chain"]

            # Try local PDB first, then fetch from SAbDab
            if pdb_dir:
                # Local files: {pdb_dir}/{chain_type}/{pdb}_{chain}.pdb
                chain_pdb = pdb_dir / chain_type / f"{pdb_id}_{chain_id}.pdb"
                if not chain_pdb.exists():
                    results[chain_type].append(
                        {
                            "pdb": f"{pdb_id}_{chain_id}",
                            "error": "Local PDB not found",
                            "perfect": False,
                        }
                    )
                    continue
            else:
                # Fetch from SAbDab
                full_pdb = cache_dir / f"{pdb_id}.pdb"
                if not full_pdb.exists():
                    try:
                        fetch_imgt_pdb(pdb_id, str(full_pdb))
                    except Exception as e:
                        print(f"Fetch failed for {pdb_id}: {e}")
                        results[chain_type].append(
                            {
                                "pdb": f"{pdb_id}_{chain_id}",
                                "error": f"Fetch failed: {e}",
                                "perfect": False,
                            }
                        )
                        continue

                # Extract chain
                chain_pdb = cache_dir / f"{pdb_id}_{chain_id}.pdb"
                if not extract_chain_to_pdb(
                    str(full_pdb), chain_id, str(chain_pdb), args.filter_to_imgt
                ):
                    results[chain_type].append(
                        {
                            "pdb": f"{pdb_id}_{chain_id}",
                            "error": "Failed to extract chain",
                            "perfect": False,
                        }
                    )
                    continue

            # Run SAbR (also extracts input positions from PDB)
            try:
                sabr_result = run_sabr_pipeline(str(chain_pdb), chain_id)
            except Exception as e:
                print(f"SAbR failed for {pdb_id}_{chain_id}: {e}")
                results[chain_type].append(
                    {
                        "pdb": f"{pdb_id}_{chain_id}",
                        "error": f"SAbR failed: {e}",
                        "perfect": False,
                    }
                )
                continue

            # Compare input positions (from IMGT-numbered PDB) with SAbR output
            input_positions = sabr_result["input_positions"]
            comparison = compare_positions(
                input_positions, sabr_result["output_positions"]
            )

            results[chain_type].append(
                {
                    "pdb": f"{pdb_id}_{chain_id}",
                    "chain_type_detected": sabr_result["chain_type"],
                    "n_residues": len(input_positions),
                    "perfect": comparison["perfect"],
                    "n_deviations": comparison["n_deviations"],
                    "deviations": comparison["deviations"],
                }
            )

    # Generate summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    total_perfect = 0
    total_count = 0
    total_failed = 0

    for chain_type in ["heavy", "kappa", "lambda"]:
        type_results = results.get(chain_type, [])
        if not type_results:
            continue

        # Only count successful cases (no error) in the totals
        successful = [r for r in type_results if "error" not in r]
        failed = [r for r in type_results if "error" in r]

        n_perfect = sum(1 for r in successful if r.get("perfect", False))
        n_successful = len(successful)
        n_failed = len(failed)
        total_perfect += n_perfect
        total_count += n_successful
        total_failed += n_failed

        if n_successful > 0:
            accuracy = round(100 * n_perfect / n_successful, 1)
            print(
                f"{chain_type.upper()}: {n_perfect}/{n_successful} perfect "
                f"({accuracy}%)"
                + (f" [{n_failed} failed]" if n_failed > 0 else "")
            )
        elif n_failed > 0:
            print(f"{chain_type.upper()}: 0/0 perfect [{n_failed} failed]")

    if total_count > 0:
        overall_accuracy = round(100 * total_perfect / total_count, 1)
        print(f"\nOVERALL: {total_perfect}/{total_count} perfect")
        print(f"  Accuracy: {overall_accuracy}%")
    if total_failed > 0:
        print(f"  Failed: {total_failed} entries")

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
