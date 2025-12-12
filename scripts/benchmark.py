#!/usr/bin/env python3
"""Benchmark renumbering accuracy from precomputed NPZ embeddings.

This script compares renumbering results from NPZ embedding files against
original PDB numbering to assess accuracy. It loads precomputed MPNN embeddings,
runs them through the SAbR renumbering pipeline (softaligner + ANARCI), and
compares the assigned numbering to the original PDB numbering.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from ANARCI import anarci
from Bio import PDB

from sabr import aln2hmm, constants, mpnn_embeddings, softaligner, util

logging.basicConfig(
    level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s"
)
LOGGER = logging.getLogger(__name__)


def extract_numbering_from_pdb(
    pdb_file: str, chain_id: str
) -> Dict[int, Tuple[int, str]]:
    """Extract residue numbering from a PDB file.

    Args:
        pdb_file: Path to PDB file
        chain_id: Chain identifier

    Returns:
        Dict mapping sequential index to (resnum, icode) tuple
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("structure", pdb_file)

    numbering = {}
    idx = 0
    for residue in structure[0][chain_id].get_residues():
        # Skip HETATM
        if residue.get_id()[0].strip() != "":
            continue
        resnum = residue.get_id()[1]
        icode = residue.get_id()[2].strip()
        numbering[idx] = (resnum, icode)
        idx += 1

    return numbering


def renumber_from_npz(
    npz_file: str,
    original_pdb: str,
    chain_id: str,
    chain_type_filter,
    numbering_scheme: str = "imgt",
) -> Tuple[List[Tuple[Tuple[int, str], str]], int, int, int]:
    """Run renumbering process using NPZ embeddings.

    Args:
        npz_file: Path to NPZ embedding file
        original_pdb: Path to original PDB file (for sequence)
        chain_id: Chain identifier
        chain_type_filter: Chain type filter for softaligner
        numbering_scheme: ANARCI numbering scheme (default: "imgt")

    Returns:
        Tuple of (anarci_out, start_res, end_res, alignment_start)
    """
    # Load embeddings from NPZ
    embeddings = mpnn_embeddings.from_npz(npz_file)

    # Get sequence from original PDB (for subsequence construction)
    sequence = util.fetch_sequence_from_pdb(original_pdb, chain_id)

    # Align embeddings against species references
    soft_aligner = softaligner.SoftAligner()
    out = soft_aligner(embeddings, chain_type=chain_type_filter)

    # Convert alignment to state vector
    sv, start, end = aln2hmm.alignment_matrix_to_state_vector(out.alignment)

    subsequence = "-" * start + sequence[start:end]

    # Run ANARCI numbering
    anarci_out, start_res, end_res = anarci.number_sequence_from_alignment(
        sv,
        subsequence,
        scheme=numbering_scheme,
        chain_type=out.species[-1],
    )

    # Filter out deletions
    anarci_out = [a for a in anarci_out if a[1] != "-"]

    return anarci_out, start_res, end_res, start


def compare_numbering(
    original: Dict[int, Tuple[int, str]],
    new_anarci: List[Tuple[Tuple[int, str], str]],
    alignment_start: int,
    file_name: str,
) -> Dict[str, any]:
    """Compare original PDB numbering to new ANARCI numbering.

    Insertion codes are normalized: empty strings and single spaces are
    treated as equivalent to avoid false positives.

    Args:
        original: Original PDB numbering (index -> (resnum, icode))
        new_anarci: New ANARCI numbering output
        alignment_start: Alignment start position
        file_name: Name of file being compared

    Returns:
        Dictionary with comparison statistics
    """
    differences = []
    matches = 0
    total_compared = 0

    # Build a mapping of position -> numbering for new ANARCI output
    # Account for alignment_start offset
    new_numbering = {}
    for i, ((resnum, icode), _aa) in enumerate(new_anarci):
        pdb_idx = alignment_start + i
        # Normalize insertion code: treat '' and ' ' as equivalent
        normalized_icode = icode.strip() if icode else ""
        new_numbering[pdb_idx] = (resnum, normalized_icode)

    # Compare
    for idx in sorted(original.keys()):
        if idx in new_numbering:
            total_compared += 1
            orig_num, orig_icode = original[idx]
            # Normalize original insertion code too
            normalized_orig_icode = orig_icode.strip() if orig_icode else ""
            new_num, new_icode = new_numbering[idx]

            if orig_num == new_num and normalized_orig_icode == new_icode:
                matches += 1
            else:
                differences.append(
                    {
                        "index": idx,
                        "original": (orig_num, normalized_orig_icode),
                        "new": (new_num, new_icode),
                    }
                )

    return {
        "file": file_name,
        "total_compared": total_compared,
        "matches": matches,
        "deviations": len(differences),
        "match_rate": matches / total_compared if total_compared > 0 else 0,
        "diff_details": differences[:5],  # First 5 differences for display
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark renumbering accuracy from precomputed NPZ files"
    )
    parser.add_argument(
        "--npz-dir",
        type=str,
        required=True,
        help="Directory containing NPZ embedding files",
    )
    parser.add_argument(
        "--pdb-dir",
        type=str,
        required=True,
        help="Directory containing original PDB files",
    )
    parser.add_argument(
        "--chain",
        type=str,
        default="A",
        help="Chain identifier (default: A)",
    )
    parser.add_argument(
        "--chain-type",
        type=str,
        choices=["heavy", "light", "auto"],
        default="heavy",
        help="Chain type filter for alignment (default: heavy)",
    )
    parser.add_argument(
        "--scheme",
        type=str,
        default="imgt",
        help="ANARCI numbering scheme (default: imgt)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed per-file output",
    )

    args = parser.parse_args()

    # Convert paths to Path objects
    npz_dir = Path(args.npz_dir)
    pdb_dir = Path(args.pdb_dir)

    if not npz_dir.exists():
        LOGGER.error(f"NPZ directory not found: {npz_dir}")
        sys.exit(1)

    if not pdb_dir.exists():
        LOGGER.error(f"PDB directory not found: {pdb_dir}")
        sys.exit(1)

    # Map chain type string to constant
    chain_type_map = {
        "heavy": constants.ChainType.HEAVY,
        "light": constants.ChainType.LIGHT,
        "auto": None,
    }
    chain_type_filter = chain_type_map[args.chain_type]

    # Process NPZ files
    npz_files = sorted(npz_dir.glob("*.npz"))
    LOGGER.info(f"Found {len(npz_files)} NPZ files to process\n")

    if len(npz_files) == 0:
        LOGGER.error(f"No NPZ files found in {npz_dir}")
        sys.exit(1)

    results = []
    total_deviations = 0
    total_residues_compared = 0
    failed_files = []

    for npz_file in npz_files:
        # Match NPZ file to PDB file (assumes same base name)
        base_name = npz_file.stem
        pdb_file = pdb_dir / f"{base_name}.pdb"

        if not pdb_file.exists():
            LOGGER.warning(f"PDB file not found for {npz_file.name}")
            continue

        if args.verbose:
            LOGGER.info(f"Processing: {npz_file.name}")

        try:
            # Extract original numbering from PDB
            original_numbering = extract_numbering_from_pdb(
                str(pdb_file), args.chain
            )

            # Run renumbering from NPZ
            anarci_out, start_res, end_res, alignment_start = renumber_from_npz(
                str(npz_file),
                str(pdb_file),
                args.chain,
                chain_type_filter,
                args.scheme,
            )

            # Compare
            comparison = compare_numbering(
                original_numbering, anarci_out, alignment_start, npz_file.name
            )

            results.append(comparison)
            total_deviations += comparison["deviations"]
            total_residues_compared += comparison["total_compared"]

            if args.verbose:
                LOGGER.info(
                    f"  Compared: {comparison['total_compared']}, "
                    f"Matches: {comparison['matches']}, "
                    f"Deviations: {comparison['deviations']} "
                    f"({comparison['match_rate']:.1%})"
                )

                if comparison["deviations"] > 0 and comparison["diff_details"]:
                    LOGGER.info("  First few deviations:")
                    for diff in comparison["diff_details"]:
                        orig = diff["original"]
                        new = diff["new"]
                        LOGGER.info(
                            f"    Index {diff['index']}: "
                            f"({orig[0]}, '{orig[1]}') -> "
                            f"({new[0]}, '{new[1]}')"
                        )
                LOGGER.info("")

        except Exception as e:
            LOGGER.error(f"  ERROR processing {npz_file.name}: {e}")
            failed_files.append((npz_file.name, str(e)))
            if args.verbose:
                import traceback

                traceback.print_exc()
            continue

    # Summary
    LOGGER.info("=" * 70)
    LOGGER.info("SUMMARY")
    LOGGER.info("=" * 70)
    LOGGER.info(f"Files processed: {len(results)}")
    LOGGER.info(f"Files failed: {len(failed_files)}")
    LOGGER.info(f"Total residues compared: {total_residues_compared}")
    LOGGER.info(f"Total deviations: {total_deviations}")

    if total_residues_compared > 0:
        overall_match_rate = (
            total_residues_compared - total_deviations
        ) / total_residues_compared
        LOGGER.info(f"Overall match rate: {overall_match_rate:.1%}")

    LOGGER.info("")

    # Detailed results
    if results:
        LOGGER.info("Per-file deviation counts:")
        for r in sorted(results, key=lambda x: x["deviations"], reverse=True):
            LOGGER.info(
                f"  {r['file']:<30} {r['deviations']:>4} deviations "
                f"({r['match_rate']:.1%} match)"
            )

    if failed_files:
        LOGGER.info("")
        LOGGER.info("Failed files:")
        for fname, error in failed_files:
            LOGGER.info(f"  {fname}: {error}")

    LOGGER.info("")
    LOGGER.info(
        f"FINAL RESULT: {total_deviations} total deviations "
        f"across all files"
    )

    # Return non-zero exit code if there were failures
    if failed_files:
        sys.exit(1)


if __name__ == "__main__":
    main()
