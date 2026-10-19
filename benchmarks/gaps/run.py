"""Paired gap experiment: PYTHONPATH=src:. python -m benchmarks.gaps.run.

Uses pinned independent SAbDab labels; never renumbers the shortened sequence
as ground truth. Baseline disables ONLY the new DP keyword, retaining gap-aware
correction skipping. Encodings are recomputed after every coordinate deletion.
"""

import json
import logging
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np

from benchmarks.antibody.run import load_case, load_cases
from sabr import alignment
from sabr.model import encode
from sabr.numbering import number_alignment
from sabr.structure import apply_numbering, extract_chain

ROOT = Path(__file__).parent
ORIGINAL = alignment._align_reference


def baseline(*args, **kwargs):
    kwargs.pop("query_gap_indices", None)
    return ORIGINAL(*args, **kwargs)


def evaluate(structure, chain, labels, scfv=False):
    data = extract_chain(structure, chain, None)
    embedding = encode(data.coords)
    result = {
        "n": len(labels),
        "gaps": sorted(data.gap_indices),
        "gap_residues": [
            [data.residue_ids[i], data.residue_ids[i + 1]]
            for i in sorted(data.gap_indices)
        ],
    }
    for name, function in [("baseline", baseline), ("gap_aware", ORIGINAL)]:
        try:
            with patch.object(alignment, "_align_reference", function):
                aln, selected, score = alignment.align(
                    embedding, data.gap_indices, "auto", 0.0, scfv=scfv
                )
            numbered = number_alignment(aln, data.sequence, "imgt", selected)
            output = apply_numbering(structure, chain, data, numbered)
            residues = list(output[0][chain])
            errors = []
            for entry in labels:
                expected = (
                    entry["domain"] * 1000 + entry["imgt_position"],
                    entry["insertion_code"],
                )
                residue = residues[entry["row"]]
                actual = (residue.id[1], residue.id[2].strip())
                if actual != expected:
                    errors.append(
                        {
                            "row": entry["row"],
                            "expected": expected,
                            "actual": actual,
                            "alignment_columns": (
                                np.flatnonzero(aln[entry["row"]]) + 1
                            ).tolist(),
                        }
                    )
            result[name] = {
                "errors": len(errors),
                "mismatches": errors,
                "selected": selected,
                "score": score,
                "failure": None,
            }
        except ValueError as exc:
            result[name] = {"errors": len(labels), "failure": str(exc)}
    return result


def run():
    records = []
    for case in load_cases():
        if case["id"] not in (
            "12e8_H",
            "12e8_L",
            "1mel_A",
            "1lmk_A",
            "1moe_A",
            "6nou_A",
            "5kve_L",
        ):
            continue
        structure, labels = load_case(case)
        natural = case["id"] in ("6nou_A", "5kve_L")
        records.append(
            {
                "id": case["id"],
                "kind": "natural" if natural else "intact",
                **evaluate(
                    structure,
                    case["chain"],
                    labels,
                    len(case["expected_types"]) > 1,
                ),
            }
        )
        if natural:
            continue
        for domain in range(len(case["expected_types"])):
            domain_labels = [e for e in labels if e["domain"] == domain]
            # Crop each complete domain before deletion; retain its labels.
            domain_rows = {e["row"] for e in domain_labels}
            for region, position in [
                ("FR1", 10),
                ("CDR1", 29),
                ("FR2", 45),
                ("CDR2", 58),
                ("DE", 81),
                ("FR3", 91),
                ("CDR3", 108),
            ]:
                starts = [
                    i
                    for i, e in enumerate(domain_labels)
                    if e["imgt_position"] >= position
                ]
                if not starts:
                    continue
                start = starts[0]
                for count in (1, 3, 5):
                    deleted = {
                        e["row"] for e in domain_labels[start : start + count]
                    }
                    kept = domain_rows - deleted
                    mutant = structure.copy()
                    target = mutant[0][case["chain"]]
                    for row, residue in enumerate(list(target)):
                        if row not in kept:
                            target.detach_child(residue.id)
                    rowmap = {old: new for new, old in enumerate(sorted(kept))}
                    truth = [
                        {**e, "row": rowmap[e["row"]], "domain": 0}
                        for e in domain_labels
                        if e["row"] in kept
                    ]
                    records.append(
                        {
                            "id": f"{case['id']}_d{domain}_{region}_{count}",
                            "kind": "deletion",
                            "parent": case["id"],
                            "domain": domain,
                            "region": region,
                            "deleted_count": count,
                            **evaluate(mutant, case["chain"], truth),
                        }
                    )
        print(case["id"], len(records), flush=True)
    (ROOT / "results.json").write_text(json.dumps(records, indent=2) + "\n")
    for kind in ("intact", "natural", "deletion"):
        group = [r for r in records if r["kind"] == kind]
        print(kind, "cases", len(group), "residues", sum(r["n"] for r in group))
        for method in ("baseline", "gap_aware"):
            print(
                method,
                "errors",
                sum(r[method]["errors"] for r in group),
                "exact",
                sum(r[method]["errors"] == 0 for r in group),
                "failures",
                sum(r[method]["failure"] is not None for r in group),
            )
        print(
            "better/equal/worse",
            [
                sum(
                    np.sign(r["gap_aware"]["errors"] - r["baseline"]["errors"])
                    == s
                    for r in group
                )
                for s in (-1, 0, 1)
            ],
        )


if __name__ == "__main__":
    logging.disable(logging.CRITICAL)
    warnings.simplefilter("ignore", UserWarning)
    run()
