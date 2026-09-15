"""Report independent IMGT agreement, boundaries, and rejection separately.

Run from the repository root: python -m benchmarks.antibody.run
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from Bio.PDB import PDBParser

from sabr import renumber_structure
from sabr.alignment import align
from sabr.model import encode
from sabr.structure import extract_chain

DATA = Path(__file__).parent


def load_cases():
    return json.loads((DATA / "manifest.json").read_text())["cases"]


def load_case(case):
    structure = PDBParser(QUIET=True).get_structure(
        case["id"], DATA / case["fixture"]
    )
    residues = list(structure[0][case["chain"]])
    start, stop = case.get("row_slice", (0, len(residues)))
    for row, residue in enumerate(residues):
        if not start <= row < stop:
            structure[0][case["chain"]].detach_child(residue.id)
    labels = []
    if "labels" in case:
        with (DATA / case["labels"]).open(newline="") as handle:
            for entry in csv.DictReader(handle):
                row = int(entry["row"])
                if start <= row < stop:
                    labels.append(
                        {
                            **entry,
                            "row": row - start,
                            "domain": int(entry["domain"]),
                            "imgt_position": int(entry["imgt_position"]),
                        }
                    )
    return structure, labels


def evaluate_case(case, mode="sabr"):
    """Run the unmocked public API; inspect real alignment for boundaries.

    Final residue IDs include extrapolated terminal numbering, so they cannot
    identify alignment boundaries. A second real alignment supplies those
    bounds without changing or instrumenting the public API.
    """
    structure, labels = load_case(case)
    expected_domains = []
    for domain, chain_type in enumerate(case.get("expected_types", "")):
        rows = [x["row"] for x in labels if x["domain"] == domain]
        expected_domains.append(
            {"type": chain_type, "start": min(rows), "end": max(rows)}
        )
    result = {
        "id": case["id"],
        "category": case["category"],
        "positive": case["positive"],
        "mode": mode,
        "policy_control": "expected_rejection" in case,
        "accepted": False,
        "error": None,
        "labeled_residues": len(labels),
        "agreeing_residues": 0,
        "expected_domains": expected_domains,
        "predicted_domains": [],
        "exact_domains": 0,
        "boundary_errors": [],
        "mismatches": [],
    }
    try:
        numbered = renumber_structure(
            structure,
            case["chain"],
            scheme="imgt",
            mode=mode,
            **case["options"],
        )
    except ValueError as error:
        result["error"] = str(error)
        return result
    result["accepted"] = True
    residues = list(numbered[0][case["chain"]])
    for label in labels:
        # Multi-domain output uses a documented 1000-position stride.
        expected = (
            label["domain"] * 1000 + label["imgt_position"],
            label["insertion_code"],
        )
        residue = residues[label["row"]]
        actual = (residue.id[1], residue.id[2].strip())
        if actual == expected:
            result["agreeing_residues"] += 1
        else:
            result["mismatches"].append(
                {"row": label["row"], "expected": expected, "actual": actual}
            )
    data = extract_chain(structure, case["chain"], None)
    alignment, selected, _ = align(
        encode(data.coords, mode),
        data.gap_indices,
        "auto",
        0.0,
        mode=mode,
        scfv=case["options"].get("scfv", False),
    )
    result["selected_types"] = selected
    for domain, chain_type in enumerate(selected):
        block = alignment[:, domain * 128 : (domain + 1) * 128]
        rows = block.any(axis=1).nonzero()[0]
        result["predicted_domains"].append(
            {"type": chain_type, "start": int(rows[0]), "end": int(rows[-1])}
        )
    for expected, actual in zip(expected_domains, result["predicted_domains"]):
        if expected["type"] != actual["type"]:
            continue
        errors = [actual[key] - expected[key] for key in ("start", "end")]
        result["boundary_errors"].append(errors)
        result["exact_domains"] += int(errors == [0, 0])
    return result


def summarize(results):
    """Keep biological acceptance, negative controls and gap policy distinct."""
    positives = [
        r for r in results if r["positive"] and not r["policy_control"]
    ]
    accepted = [r for r in positives if r["accepted"]]
    negatives = [r for r in results if not r["positive"]]
    policy = [r for r in results if r["policy_control"]]
    errors = [
        abs(e) for r in positives for pair in r["boundary_errors"] for e in pair
    ]

    def ratio(numerator, denominator):
        return {
            "numerator": numerator,
            "denominator": denominator,
            "rate": numerator / denominator if denominator else None,
        }

    correct = sum(r["agreeing_residues"] for r in positives)
    return {
        "residue_agreement_all_positives": ratio(
            correct, sum(r["labeled_residues"] for r in positives)
        ),
        "residue_agreement_accepted_positives": ratio(
            correct, sum(r["labeled_residues"] for r in accepted)
        ),
        "exact_domain_boundaries": ratio(
            sum(r["exact_domains"] for r in positives),
            sum(len(r["expected_domains"]) for r in positives),
        ),
        "boundary_mean_absolute_error_matched_domains": (
            sum(errors) / len(errors) if errors else None
        ),
        "matched_boundary_count": len(errors),
        "positive_rejection": ratio(
            len(positives) - len(accepted), len(positives)
        ),
        "negative_rejection": ratio(
            sum(not r["accepted"] for r in negatives), len(negatives)
        ),
        "gap_policy_rejection": ratio(
            sum(not r["accepted"] for r in policy), len(policy)
        ),
        "unexpected_domain_count": sum(
            len(r["predicted_domains"]) != len(r["expected_domains"])
            for r in positives
        ),
    }


def report(mode="sabr"):
    results = [evaluate_case(case, mode) for case in load_cases()]
    return {
        "schema_version": 1,
        "manifest_sha256": hashlib.sha256(
            (DATA / "manifest.json").read_bytes()
        ).hexdigest(),
        "mode": mode,
        "scheme": "imgt",
        "noise_level": 0.0,
        "summary": summarize(results),
        "by_category": {
            category: summarize(
                [r for r in results if r["category"] == category]
            )
            for category in sorted({r["category"] for r in results})
        },
        "cases": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("sabr", "softalign"), default="sabr")
    parser.add_argument(
        "--output", type=Path, help="Write detailed JSON report"
    )
    args = parser.parse_args()
    result = report(args.mode)
    print("Case | Accepted | Residue agreement | Exact domain boundaries")
    print("--- | --- | --- | ---")
    for case in result["cases"]:
        print(
            f'{case["id"]} | {case["accepted"]} | '
            f'{case["agreeing_residues"]}/{case["labeled_residues"]} | '
            f'{case["exact_domains"]}/{len(case["expected_domains"])}'
        )
    print(json.dumps(result["summary"], indent=2))
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
