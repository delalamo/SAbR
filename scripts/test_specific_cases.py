#!/usr/bin/env python3
"""Test specific PDB cases to compare alignment implementations.

This script tests the current implementation against known test cases
that showed different behavior across implementation versions.
"""

import json
import sys
import tempfile
import warnings
from pathlib import Path

import requests
from Bio.PDB import PDBIO, PDBParser, Select

from sabr.cli import renumber
from sabr.constants import IMGT_REGIONS
from sabr.embeddings.mpnn import from_pdb

warnings.filterwarnings("ignore")


# Test cases that differ across implementations
TEST_CASES = [
    # (pdb_id, chain_id, chain_type, expected_behavior)
    # Cases fixed by nc_overhang (overhang penalty alone)
    ("7oh0", "C", "heavy", "nc_overhang_fixes"),  # CDR2 shift - 7 dev in main
    ("6vo0", "H", "heavy", "nc_overhang_fixes"),  # FR3 position 83->84
    ("5kem", "C", "kappa", "nc_overhang_fixes"),  # FR4 shift - 8 dev in main
    ("6xq2", "E", "kappa", "nc_overhang_fixes"),  # CDR2/FR3 - 9 dev in main
    ("7mdt", "L", "lambda", "nc_overhang_fixes"),  # FR1 shift - 5 dev in main
    # Cases only fixed by zero_gapopen_cdr (zero gap-open in CDRs)
    ("5hcg", "H", "heavy", "zero_gapopen_only"),  # length mismatch
    ("6xxo", "A", "heavy", "zero_gapopen_only"),  # CDR2/FR3 shift - 11 dev
]


class ChainSelect(Select):
    """BioPython Select class to filter by chain."""

    def __init__(self, chain_id: str):
        self.chain_id = chain_id

    def accept_chain(self, chain):
        return chain.id == self.chain_id


def fetch_imgt_pdb(pdb_id: str, output_path: str) -> None:
    """Fetch IMGT-numbered PDB from SAbDab."""
    base_url = "https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/pdb"
    url = f"{base_url}/{pdb_id}/?scheme=imgt"

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
    }

    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()

    with open(output_path, "w") as f:
        f.write(response.text)


def extract_chain_to_pdb(src_pdb: str, chain_id: str, dst_pdb: str) -> bool:
    """Extract a specific chain from a PDB file."""
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("input", src_pdb)

        model = structure[0]
        if chain_id not in [c.id for c in model]:
            return False

        io = PDBIO()
        io.set_structure(structure)
        io.save(dst_pdb, ChainSelect(chain_id))
        return True
    except Exception as e:
        print(f"Error extracting chain: {e}")
        return False


def get_region_for_position(pos_num: int) -> str:
    """Get IMGT region name for a position number."""
    for reg_name, positions in IMGT_REGIONS.items():
        if pos_num in positions:
            return reg_name
    return "unknown"


def run_sabr_and_compare(pdb_path: str, chain_id: str) -> dict:
    """Run SAbR pipeline and compare with expected IMGT numbering."""
    input_data = from_pdb(pdb_path, chain_id)

    anarci_out, chain_type, first_aligned_row = renumber.run_renumbering_pipeline(
        input_data,
        numbering_scheme="imgt",
        chain_type="auto",
        deterministic_loop_renumbering=True,
    )

    # Parse output positions
    output_positions = []
    for pos, _aa in anarci_out:
        resnum = pos[0]
        insertion = pos[1].strip() if pos[1].strip() else ""
        output_positions.append(f"{resnum}{insertion}")

    # Get input positions (expected IMGT from SAbDab)
    input_positions = input_data.idxs[first_aligned_row:]

    # Filter to IMGT range 1-128
    def in_imgt_range(pos):
        try:
            pos_num = int("".join(c for c in pos if c.isdigit()))
            return 1 <= pos_num <= 128
        except ValueError:
            return False

    input_filtered = [p for p in input_positions if in_imgt_range(p)]
    output_filtered = [p for p in output_positions if in_imgt_range(p)]

    # Compare
    deviations = []
    min_len = min(len(input_filtered), len(output_filtered))

    for i in range(min_len):
        inp = input_filtered[i]
        out = output_filtered[i]
        if inp != out:
            try:
                out_num = int("".join(c for c in out if c.isdigit()))
                region = get_region_for_position(out_num)
            except ValueError:
                region = "unknown"
            deviations.append((i, inp, out, region))

    # Length mismatch
    length_mismatch = None
    if len(input_filtered) != len(output_filtered):
        length_mismatch = {
            "imgt_input": len(input_filtered),
            "imgt_output": len(output_filtered),
            "diff": len(input_filtered) - len(output_filtered),
        }

    return {
        "chain_type": chain_type,
        "n_residues": len(input_positions),
        "perfect": len(deviations) == 0 and length_mismatch is None,
        "n_deviations": len(deviations),
        "deviations": deviations,
        "length_mismatch": length_mismatch,
    }


def main():
    cache_dir = Path(tempfile.mkdtemp())
    print(f"Cache directory: {cache_dir}\n")

    results = []

    for pdb_id, chain_id, chain_type, expected in TEST_CASES:
        print(f"{'='*60}")
        print(f"Testing {pdb_id}_{chain_id} ({chain_type})")
        print(f"Expected behavior: {expected}")
        print(f"{'='*60}")

        try:
            # Fetch PDB
            full_pdb = cache_dir / f"{pdb_id}.pdb"
            if not full_pdb.exists():
                print(f"Fetching {pdb_id} from SAbDab...")
                fetch_imgt_pdb(pdb_id, str(full_pdb))

            # Extract chain
            chain_pdb = cache_dir / f"{pdb_id}_{chain_id}.pdb"
            if not extract_chain_to_pdb(str(full_pdb), chain_id, str(chain_pdb)):
                print(f"ERROR: Failed to extract chain {chain_id}")
                continue

            # Run SAbR
            result = run_sabr_and_compare(str(chain_pdb), chain_id)

            print(f"\nResult:")
            print(f"  Chain type detected: {result['chain_type']}")
            print(f"  Perfect: {result['perfect']}")
            print(f"  N deviations: {result['n_deviations']}")

            if result["deviations"]:
                print(f"  Deviations:")
                for idx, inp, out, region in result["deviations"]:
                    print(f"    [{idx}] {inp} -> {out} ({region})")

            if result["length_mismatch"]:
                print(f"  Length mismatch: {result['length_mismatch']}")

            results.append({
                "pdb": f"{pdb_id}_{chain_id}",
                "chain_type": chain_type,
                "expected": expected,
                **result,
            })

        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                "pdb": f"{pdb_id}_{chain_id}",
                "chain_type": chain_type,
                "expected": expected,
                "error": str(e),
            })

        print()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    n_perfect = sum(1 for r in results if r.get("perfect", False))
    n_total = len([r for r in results if "error" not in r])

    print(f"Perfect: {n_perfect}/{n_total}")

    for r in results:
        status = "PERFECT" if r.get("perfect") else f"IMPERFECT ({r.get('n_deviations', '?')} dev)"
        if "error" in r:
            status = f"ERROR: {r['error']}"
        print(f"  {r['pdb']}: {status}")


if __name__ == "__main__":
    main()
