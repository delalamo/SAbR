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
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import requests
from Bio.PDB import PDBIO, PDBParser, Select

from sabr import mpnn_embeddings, renumber
from sabr.constants import IMGT_REGIONS


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
                raise RuntimeError(
                    f"Invalid PDB content for {pdb_id}: "
                    f"response does not contain REMARK, ATOM or HEADER. "
                    f"First 200 chars: {content[:200]}"
                )

            with open(output_path, "w") as f:
                f.write(content)
            return

        except requests.exceptions.RequestException as e:
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


class ChainResidueSelect(Select):
    """BioPython Select class to filter by chain and residue range."""

    def __init__(self, chain_id: str, min_res: int = 1, max_res: int = 128):
        self.chain_id = chain_id
        self.min_res = min_res
        self.max_res = max_res

    def accept_chain(self, chain):
        return chain.id == self.chain_id

    def accept_residue(self, residue):
        res_id = residue.get_id()
        resnum = res_id[1]
        return self.min_res <= resnum <= self.max_res


def extract_chain_to_pdb(src_pdb: str, chain_id: str, dst_pdb: str) -> bool:
    """Extract a specific chain (residues 1-128) from a PDB file."""
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("input", src_pdb)

        # Check if chain exists
        model = structure[0]
        if chain_id not in [c.id for c in model]:
            return False

        io = PDBIO()
        io.set_structure(structure)
        io.save(dst_pdb, ChainResidueSelect(chain_id, 1, 128))

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
    """
    # Extract embeddings (also provides input residue IDs)
    input_data = mpnn_embeddings.from_pdb(pdb_path, chain_id)

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


def compare_positions(
    input_positions: List[str], output_positions: List[str]
) -> Dict:
    """Compare input and output IMGT positions.

    Returns:
        Dict with deviations categorized by region.
    """
    deviations = defaultdict(list)
    perfect = True

    min_len = min(len(input_positions), len(output_positions))

    for i in range(min_len):
        inp = input_positions[i]
        out = output_positions[i]

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
    if len(input_positions) != len(output_positions):
        perfect = False
        deviations["length_mismatch"].append(
            (f"input={len(input_positions)}", f"output={len(output_positions)}")
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
                    fetch_imgt_pdb(pdb_id, str(full_pdb))

                # Extract chain
                chain_pdb = cache_dir / f"{pdb_id}_{chain_id}.pdb"
                if not extract_chain_to_pdb(
                    str(full_pdb), chain_id, str(chain_pdb)
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

    for chain_type in ["heavy", "kappa", "lambda"]:
        type_results = results.get(chain_type, [])
        if not type_results:
            continue

        n_perfect = sum(1 for r in type_results if r.get("perfect", False))
        n_total = len(type_results)
        total_perfect += n_perfect
        total_count += n_total

        accuracy = round(100 * n_perfect / n_total, 1) if n_total > 0 else 0
        print(
            f"{chain_type.upper()}: {n_perfect}/{n_total} perfect ({accuracy}%)"
        )

    if total_count > 0:
        overall_accuracy = round(100 * total_perfect / total_count, 1)
        print(f"\nOVERALL: {total_perfect}/{total_count} perfect")
        print(f"  Accuracy: {overall_accuracy}%")

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
