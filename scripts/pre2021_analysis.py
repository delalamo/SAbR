#!/usr/bin/env python3
"""Pre-2021 SAbDab Analysis Script.

This script evaluates SAbR numbering accuracy on pre-August 2021 SAbDab
structures. It reads a CSV file with PDB IDs and chains, fetches IMGT-numbered
structures from SAbDab, runs the SAbR pipeline, and compares output to
expected IMGT numbering.

Usage:
    python pre2021_analysis.py --csv pre2021_pdb_chains.csv
    python pre2021_analysis.py --csv pre2021_pdb_chains.csv --limit 10
"""

import argparse
import csv
import json
import tempfile
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from Bio.PDB import PDBIO, PDBParser, Select

from sabr import aln2hmm, mpnn_embeddings, softaligner
from sabr.constants import IMGT_LOOPS

# Build IMGT_REGIONS from sabr constants and IMGT framework definitions
# Framework positions are: FR1=1-26, FR2=39-55, FR3=66-104, FR4=118-128
IMGT_REGIONS = {
    "FR1": list(range(1, 27)),
    "CDR1": list(range(IMGT_LOOPS["CDR1"][0], IMGT_LOOPS["CDR1"][1] + 1)),
    "FR2": list(range(39, 56)),
    "CDR2": list(range(IMGT_LOOPS["CDR2"][0], IMGT_LOOPS["CDR2"][1] + 1)),
    "FR3": list(range(66, 105)),
    "CDR3": list(range(IMGT_LOOPS["CDR3"][0], IMGT_LOOPS["CDR3"][1] + 1)),
    "FR4": list(range(118, 129)),
}


def fetch_imgt_pdb(pdb_id: str, output_path: str) -> bool:
    """Fetch IMGT-numbered PDB from SAbDab.

    Args:
        pdb_id: 4-letter PDB ID
        output_path: Path to save the PDB file

    Returns:
        True if successful, False otherwise
    """
    base_url = "https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/pdb"
    url = f"{base_url}/{pdb_id}/?scheme=imgt"
    try:
        urllib.request.urlretrieve(url, output_path)
        # Verify it's a valid PDB (not an error page)
        with open(output_path, "r") as f:
            content = f.read(100)
            if "ATOM" not in content and "HEADER" not in content:
                return False
        return True
    except Exception as e:
        print(f"Failed to fetch {pdb_id}: {e}")
        return False


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


def parse_pdb_residue_ids(pdb_path: str, chain_id: str) -> List[str]:
    """Parse residue IDs from a PDB file for a specific chain using BioPython.

    Returns:
        List of residue ID strings (e.g., ['1', '2', '27A', '27B', ...])
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("input", pdb_path)

    residue_ids = []
    for chain in structure[0]:
        if chain.id != chain_id:
            continue

        for residue in chain.get_residues():
            # Skip heteroatoms
            hetflag = residue.get_id()[0]
            if hetflag.strip():
                continue

            res_id = residue.get_id()
            resnum = res_id[1]
            icode = res_id[2].strip()

            # Filter to residues 1-128
            if resnum < 1 or resnum > 128:
                continue

            if icode:
                residue_ids.append(f"{resnum}{icode}")
            else:
                residue_ids.append(str(resnum))

    return residue_ids


def run_sabr_pipeline(pdb_path: str, chain_id: str) -> Optional[Dict]:
    """Run full SAbR pipeline on a PDB file.

    Returns:
        Dict with 'output_positions', 'chain_type', 'sequence'
    """
    try:
        from anarci import number_sequence_from_alignment

        # Extract embeddings
        input_data = mpnn_embeddings.from_pdb(pdb_path, chain_id)

        # Align
        aligner = softaligner.SoftAligner()
        out = aligner(input_data, deterministic_loop_renumbering=True)

        # Convert to state vector
        sv, start, end, _ = aln2hmm.alignment_matrix_to_state_vector(
            out.alignment
        )

        # Build subsequence
        subsequence = "-" * start + input_data.sequence[: end - start]

        # Get ANARCI numbering
        anarci_out, _, _ = number_sequence_from_alignment(
            sv, subsequence, scheme="imgt", chain_type=out.chain_type
        )

        # Parse output positions
        output_positions = []
        for pos, aa in anarci_out:
            if aa != "-":
                resnum = pos[0]
                insertion = pos[1].strip() if pos[1].strip() else ""
                output_positions.append(f"{resnum}{insertion}")

        return {
            "output_positions": output_positions,
            "chain_type": out.chain_type,
            "sequence": input_data.sequence,
        }
    except Exception as e:
        print(f"Error running SAbR on {pdb_path}: {e}")
        return None


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
    parser = argparse.ArgumentParser(description="Pre-2021 SAbDab Analysis")
    parser.add_argument(
        "--csv", required=True, help="Path to CSV file with PDB IDs and chains"
    )
    parser.add_argument(
        "--output", default="pre2021_results.json", help="Output JSON file"
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

    # Set up cache directory
    cache_dir = (
        Path(args.cache_dir) if args.cache_dir else Path(tempfile.mkdtemp())
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using cache directory: {cache_dir}")

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

            # Fetch or use cached PDB
            full_pdb = cache_dir / f"{pdb_id}.pdb"
            if not full_pdb.exists():
                if not fetch_imgt_pdb(pdb_id, str(full_pdb)):
                    results[chain_type].append(
                        {
                            "pdb": f"{pdb_id}_{chain_id}",
                            "error": "Failed to fetch PDB",
                            "perfect": False,
                        }
                    )
                    continue

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

            # Parse input positions (from IMGT-numbered PDB)
            input_positions = parse_pdb_residue_ids(str(chain_pdb), chain_id)
            if not input_positions:
                results[chain_type].append(
                    {
                        "pdb": f"{pdb_id}_{chain_id}",
                        "error": "No residues found",
                        "perfect": False,
                    }
                )
                continue

            # Run SAbR
            sabr_result = run_sabr_pipeline(str(chain_pdb), chain_id)
            if sabr_result is None:
                results[chain_type].append(
                    {
                        "pdb": f"{pdb_id}_{chain_id}",
                        "error": "SAbR failed",
                        "perfect": False,
                    }
                )
                continue

            # Compare
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
