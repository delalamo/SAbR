"""Independent biological agreement and unmocked pipeline invariance."""

import hashlib
from functools import lru_cache

import numpy as np
import pytest
from Bio.SeqUtils import seq1

from benchmarks.antibody.run import (
    DATA,
    evaluate_case,
    load_case,
    load_cases,
    summarize,
)
from sabr import renumber_structure
from sabr.structure import extract_chain

CASES = load_cases()
BY_ID = {case["id"]: case for case in CASES}
MODES = ("sabr", "softalign")


@lru_cache(maxsize=None)
def _result(case_id, mode):
    return evaluate_case(BY_ID[case_id], mode)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_fixture_provenance_and_independent_labels(case):
    assert (
        hashlib.sha256((DATA / case["fixture"]).read_bytes()).hexdigest()
        == case["fixture_sha256"]
    )
    if "labels" in case:
        assert (
            hashlib.sha256((DATA / case["labels"]).read_bytes()).hexdigest()
            == case["labels_sha256"]
        )
    structure, labels = load_case(case)
    data = extract_chain(structure, case["chain"], None)
    assert len(data.sequence) == len(structure[0][case["chain"]])
    assert [label["row"] for label in labels] == sorted(
        {label["row"] for label in labels}
    )
    for label in labels:
        assert data.sequence[label["row"]] == label["amino_acid"]
        assert data.residue_ids[label["row"]] == (
            int(label["source_residue_number"]),
            label["source_insertion_code"],
        )
        assert case["expected_types"][label["domain"]] == label["chain_type"]
        assert 1 <= label["imgt_position"] <= 128
    if case["positive"]:
        assert labels
        assert case["annotation_url"].startswith(
            "https://sabdab.opig.stats.ox.ac.uk/"
        )
        assert case["residue_mapping_url"].endswith(".cif")
    # Linker gaps remain in the fixtures rather than being filled or removed.
    assert bool(data.gap_indices) == (case["pdb_id"] in ("6NOU", "5KVE"))


def test_cohort_contains_both_scfv_orders_and_distinct_controls():
    assert {case["category"] for case in CASES} == {
        "conventional",
        "vhh_long_cdr3",
        "diabody_vh_vl",
        "diabody_vl_vh",
        "scfv_vh_vl",
        "scfv_vl_vh",
        "negative",
        "truncated",
        "gap_policy",
    }
    assert BY_ID["6nou_A"]["expected_types"] == "HK"
    assert BY_ID["5kve_L"]["expected_types"] == "KH"
    _, vhh = load_case(BY_ID["1mel_A"])
    assert sum(105 <= row["imgt_position"] <= 117 for row in vhh) == 26


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    "case",
    [case for case in CASES if case["positive"]],
    ids=lambda case: case["id"],
)
def test_curated_biological_agreement_does_not_regress(case, mode):
    result = _result(case["id"], mode)
    if "expected_rejection" in case:
        assert not result["accepted"]
        assert case["expected_rejection"] in result["error"]
        return
    assert result["accepted"], result["error"]
    assert result["selected_types"] == case["expected_types"]
    # Reviewed limits describe current discrepancies, not a generated oracle.
    # Improvements pass; widening a limit requires scientific review.
    allowed_mismatches = 8 if case["id"] == "12e8_H_c_truncated" else 0
    assert (
        result["labeled_residues"] - result["agreeing_residues"]
        <= allowed_mismatches
    )
    assert len(result["predicted_domains"]) == len(result["expected_domains"])
    allowed_boundary_error = 1 if case["id"] == "6nou_A" else 0
    assert (
        sum(abs(value) for pair in result["boundary_errors"] for value in pair)
        <= allowed_boundary_error
    )
    assert len(result["boundary_errors"]) == len(result["expected_domains"])


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="Known limitation: alignment currently accepts non-antibody folds; "
    "a calibrated biological rejection rule is not implemented.",
)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case_id", ("1ubq_A", "1lyz_A"))
def test_non_antibody_controls_should_be_rejected(case_id, mode):
    assert not _result(case_id, mode)["accepted"]


def _signature(structure, case, mode):
    try:
        numbered = renumber_structure(
            structure, case["chain"], mode=mode, **case["options"]
        )
    except ValueError as error:
        return ("rejected", str(error))
    return (
        "accepted",
        tuple(
            (r.id[1], r.id[2].strip(), seq1(r.resname))
            for r in numbered[0][case["chain"]]
        ),
    )


@lru_cache(maxsize=None)
def _original_signature(case_id, mode):
    case = BY_ID[case_id]
    structure, _ = load_case(case)
    return _signature(structure, case, mode)


def _atoms(structure):
    # Transform every alternate conformer, including atoms not selected by
    # BioPython's convenience iterator; SAbR may use any complete conformer.
    return [
        atom
        for residue in structure.get_residues()
        for atom in residue.get_unpacked_list()
    ]


def _transform(structure, chain, variant):
    if variant in ("rotation", "combined"):
        # A general proper rotation about (1, 2, 3), not an axis permutation.
        axis = np.array([1.0, 2.0, 3.0]) / np.sqrt(14)
        x, y, z = axis
        cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        angle = 0.731
        rotation = (
            np.eye(3)
            + np.sin(angle) * cross
            + (1 - np.cos(angle)) * (cross @ cross)
        )
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-15)
        assert np.linalg.det(rotation) == pytest.approx(1)
        for atom in _atoms(structure):
            atom.coord = atom.coord @ rotation
    if variant in ("translation", "combined"):
        for atom in _atoms(structure):
            atom.coord = atom.coord + np.array([137.25, -92.5, 48.125])
    if variant in ("numbering", "combined"):
        target = structure[0][chain]
        residues = list(target)
        numbers = np.random.default_rng(2026).permutation(
            (len(residues) + 2) // 3
        )
        for residue in residues:
            target.detach_child(residue.id)
        for row, residue in enumerate(residues):
            # Nonmonotonic, negative/gapped IDs, and reused numbers with
            # distinct insertion codes must not change physical chain order.
            residue.id = (
                " ",
                int(numbers[row // 3]) * 17 - 900,
                " AB"[row % 3],
            )
            target.add(residue)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    "variant", ("rotation", "translation", "numbering", "combined")
)
@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_complete_pipeline_is_invariant(case, variant, mode):
    structure, _ = load_case(case)
    _transform(structure, case["chain"], variant)
    original_ids = [r.id for r in structure[0][case["chain"]]]
    original_coords = np.array([a.coord.copy() for a in _atoms(structure)])
    assert _signature(structure, case, mode) == _original_signature(
        case["id"], mode
    )
    assert [r.id for r in structure[0][case["chain"]]] == original_ids
    np.testing.assert_array_equal(
        [a.coord for a in _atoms(structure)], original_coords
    )


def test_summary_does_not_hide_rejections_or_missing_domains():
    # A tiny hand-worked example: 3/6 residues overall vs 3/4 accepted;
    # one of four domains has exact boundaries; four endpoints are matched.
    common = {
        "positive": True,
        "policy_control": False,
        "accepted": True,
        "labeled_residues": 4,
        "agreeing_residues": 3,
        "expected_domains": [{}, {}, {}],
        "predicted_domains": [{}, {}],
        "exact_domains": 1,
        "boundary_errors": [[0, 0], [0, 2]],
    }
    rejected = {
        **common,
        "accepted": False,
        "labeled_residues": 2,
        "agreeing_residues": 0,
        "expected_domains": [{}],
        "predicted_domains": [],
        "exact_domains": 0,
        "boundary_errors": [],
    }
    negative = {**rejected, "positive": False, "labeled_residues": 0}
    policy = {**rejected, "policy_control": True}
    false_acceptance = {**negative, "accepted": True}
    result = summarize([common, rejected, negative, false_acceptance, policy])
    assert result["residue_agreement_all_positives"]["rate"] == 0.5
    assert result["residue_agreement_accepted_positives"]["rate"] == 0.75
    assert result["exact_domain_boundaries"]["rate"] == pytest.approx(1 / 4)
    assert result["boundary_mean_absolute_error_matched_domains"] == 0.5
    assert result["matched_boundary_count"] == 4
    assert result["positive_rejection"]["rate"] == 0.5
    assert result["negative_rejection"]["rate"] == 0.5
    assert result["gap_policy_rejection"]["rate"] == 1
    assert result["unexpected_domain_count"] == 2
    assert summarize([])["residue_agreement_all_positives"]["rate"] is None
    assert (
        summarize([rejected])["residue_agreement_accepted_positives"]["rate"]
        is None
    )
    assert summarize([rejected])["residue_agreement_all_positives"]["rate"] == 0
