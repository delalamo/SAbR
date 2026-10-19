import json
from pathlib import Path

import numpy as np
import pytest
from Bio.PDB import PDBParser

import sabr.numbering as numbering
from sabr import constants, renumber_structure
from sabr.numbering import alignment_to_states, number_alignment

DATA = Path(__file__).parent / "data"


def test_auto_selection_and_all_numbering_schemes_match_goldens(aligned_cases):
    expected = json.loads((DATA / "numbering_baseline.json").read_text())
    for chain_type, result in aligned_cases.items():
        assert result["selected"] == chain_type
        for scheme in constants.NUMBERING_SCHEMES:
            numbered = number_alignment(
                result["alignment"],
                result["data"].sequence,
                scheme,
                chain_type,
            )
            golden = expected[chain_type]["schemes"][scheme]
            actual = [
                [number, insertion_code, amino_acid]
                for _, number, insertion_code, amino_acid in numbered
            ]
            assert numbered[0][0] == golden["first_row"]
            assert actual == golden["numbered"]


def test_alignment_state_conversion_preserves_orphan_insertions():
    alignment = np.zeros((5, 8), dtype=int)
    alignment[0, 1] = 1
    alignment[1, 2] = 1
    alignment[4, 5] = 1
    states, start, first_row = alignment_to_states(alignment)
    assert start == 1
    assert first_row == 0
    assert ((3, "i"), 3) in states
    assert ((3, "i"), 4) in states
    assert ((4, "d"), None) in states


def test_reduced_reference_positions_control_states_and_padding(monkeypatch):
    alignment = np.zeros((4, 4), dtype=int)
    alignment[1, 1] = alignment[2, 2] = alignment[3, 3] = 1
    ref_positions = (29, 30, 36, 37)

    def fake_scheme(states, sequence, scheme, chain_type):
        assert states == [
            ((30, "m"), 29),
            ((36, "m"), 30),
            ((37, "m"), 31),
        ]
        assert sequence == "-" * 29 + "CDE"
        return [((30, ""), "C"), ((36, ""), "D"), ((37, ""), "E")], 29, 31

    monkeypatch.setattr(numbering, "_apply_scheme", fake_scheme)
    assert number_alignment(
        alignment,
        "ACDE",
        "imgt",
        "H",
        ref_positions=ref_positions,
    ) == [
        (1, 30, "", "C"),
        (2, 36, "", "D"),
        (3, 37, "", "E"),
    ]


def test_k_reduced_reference_expands_to_every_imgt_state(monkeypatch):
    missing = numbering._load_missing_imgt_positions()["K"]
    ref_positions = tuple(
        position
        for position in range(1, constants.IMGT_MAX_POSITION + 1)
        if position not in missing
    )
    alignment = np.eye(len(ref_positions), dtype=int)
    sequence = "A" * len(ref_positions)
    captured = {}

    def fake_scheme(states, padded_sequence, scheme, chain_type):
        captured["states"] = states
        matches = [state for state in states if state[0][1] == "m"]
        return (
            [
                ((position, ""), padded_sequence[sequence_index])
                for ((position, _), sequence_index) in matches
            ],
            0,
            len(sequence) - 1,
        )

    monkeypatch.setattr(numbering, "_apply_scheme", fake_scheme)
    records = number_alignment(
        alignment,
        sequence,
        "imgt",
        "A",
        ref_type="K",
        ref_positions=ref_positions,
    )

    states = captured["states"]
    assert [state[0][0] for state in states] == list(
        range(1, constants.IMGT_MAX_POSITION + 1)
    )
    assert {
        position for ((position, state_type), _) in states if state_type == "d"
    } == missing
    assert len(records) == len(ref_positions)


@pytest.mark.parametrize("chain_type", ("A", "B", "G", "D"))
@pytest.mark.parametrize(
    ("scheme", "last_number"), (("imgt", 127), ("aho", 148))
)
def test_reduced_k_reference_runs_real_tcr_numbering(
    chain_type, scheme, last_number
):
    missing = numbering._load_missing_imgt_positions()["K"]
    ref_positions = tuple(
        position
        for position in range(1, constants.IMGT_MAX_POSITION + 1)
        if position not in missing
    )
    records = number_alignment(
        np.eye(len(ref_positions), dtype=int),
        "A" * len(ref_positions),
        scheme,
        chain_type,
        ref_type="K",
        ref_positions=ref_positions,
    )

    assert len(records) == len(ref_positions)
    assert records[0][1] == 1
    assert records[-1][1] == last_number


def test_missing_deletion_insertion_is_idempotent_and_preserves_insertions():
    states = [
        ((9, "m"), 0),
        ((9, "i"), 1),
        ((11, "m"), 2),
    ]
    completed = numbering._insert_missing_deletions(states, "H")

    assert numbering._insert_missing_deletions(completed, "H") is completed
    assert [state for state in completed if state[0][0] == 9] == states[:2]
    assert ((10, "d"), None) in completed

    missing = numbering._load_missing_imgt_positions()["K"]
    expanded = [
        (
            (
                position,
                ("d" if position in missing else "m"),
            ),
            None,
        )
        for position in range(1, constants.IMGT_MAX_POSITION + 1)
    ]
    assert numbering._insert_missing_deletions(expanded, "K") is expanded


def test_reference_type_choices_come_from_reference_asset(monkeypatch):
    monkeypatch.setattr(
        numbering,
        "_load_missing_imgt_positions",
        lambda: {"X": frozenset({2})},
    )
    states = [((1, "m"), 0)]
    assert numbering._insert_missing_deletions(states, "X") == [
        ((1, "m"), 0),
        ((2, "d"), None),
    ]
    with pytest.raises(ValueError, match="one of X"):
        numbering._insert_missing_deletions(states, "H")


@pytest.mark.parametrize(
    ("ref_positions", "message"),
    [
        ([1, 2], "tuple"),
        ((1,), "column count"),
        ((1, True), "integers"),
        ((0, 2), "between"),
        ((1, 129), "between"),
        ((2, 1), "increasing"),
        ((1, 1), "increasing"),
    ],
)
def test_reduced_reference_positions_are_validated(ref_positions, message):
    with pytest.raises(ValueError, match=message):
        alignment_to_states(np.eye(2, dtype=int), ref_positions=ref_positions)


@pytest.mark.parametrize("chain_type", ("A", "B", "G", "D"))
def test_low_level_tcr_aho_uses_the_actual_chain_type(monkeypatch, chain_type):
    alignment = np.zeros((1, constants.IMGT_MAX_POSITION), dtype=int)
    alignment[0, 0] = 1

    def fake_aho(states, sequence, actual_chain_type):
        assert actual_chain_type == chain_type
        assert {
            position
            for ((position, state_type), _) in states
            if state_type == "d"
        } == numbering._load_missing_imgt_positions()["K"]
        return [((1, ""), "A")], 0, 0

    monkeypatch.setattr(numbering, "number_aho", fake_aho)
    assert number_alignment(
        alignment,
        "A",
        "aho",
        chain_type,
        ref_type="K",
    ) == [(0, 1, "", "A")]


@pytest.mark.parametrize("chain_type", ("A", "B", "G", "D"))
@pytest.mark.parametrize("scheme", ("chothia", "kabat", "martin", "wolfguy"))
def test_tcr_chain_types_reject_antibody_only_schemes(chain_type, scheme):
    with pytest.raises(ValueError, match="only IMGT or AHo"):
        numbering._apply_scheme([], "", scheme, chain_type)


def test_reference_metadata_rejects_invalid_or_ambiguous_combinations():
    with pytest.raises(ValueError, match="ref_type"):
        number_alignment(
            np.eye(2, dtype=int),
            "AC",
            "imgt",
            "H",
            ref_type="A",
        )
    with pytest.raises(ValueError, match="do not match"):
        number_alignment(
            np.eye(2, dtype=int),
            "AC",
            "imgt",
            "H",
            ref_type="K",
            ref_positions=(1, 2),
        )
    with pytest.raises(ValueError, match="single-domain"):
        number_alignment(
            np.eye(2, dtype=int),
            "AC",
            "imgt",
            "HK",
            ref_type="K",
        )


def test_default_numbering_skips_reduced_reference_completion(monkeypatch):
    monkeypatch.setattr(
        numbering,
        "_load_missing_imgt_positions",
        lambda: pytest.fail("default path loaded reference metadata"),
    )
    monkeypatch.setattr(
        numbering,
        "_insert_missing_deletions",
        lambda *args: pytest.fail("default path completed reference states"),
    )
    monkeypatch.setattr(
        numbering,
        "_apply_scheme",
        lambda states, sequence, scheme, actual_type: (
            [((1, ""), "A"), ((2, ""), "C")],
            0,
            1,
        ),
    )
    assert number_alignment(np.eye(2, dtype=int), "AC", "imgt", "H") == [
        (0, 1, "", "A"),
        (1, 2, "", "C"),
    ]


def test_8sve_huge_cdr1_matches_the_accepted_full_mapping():
    golden = json.loads((DATA / "8sve_cdr1_imgt.json").read_text())
    structure = PDBParser(QUIET=True).get_structure("long", DATA / "8sve_L.pdb")
    source_residues = [
        residue for residue in structure[0]["M"] if not residue.id[0].strip()
    ]
    with pytest.warns(UserWarning) as warnings:
        result = renumber_structure(
            structure,
            "M",
            scheme="imgt",
            chain_type="K",
        )
    assert [str(item.message) for item in warnings] == golden["warning"]
    target_residues = [
        residue for residue in result[0]["M"] if not residue.id[0].strip()
    ]
    actual = [
        [
            source.id[1],
            source.id[2].strip(),
            target.id[1],
            target.id[2].strip(),
            source.resname,
        ]
        for source, target in zip(source_residues, target_residues, strict=True)
    ]
    assert actual == golden["mapping"]
    assert sum(item[2] == 32 for item in actual) == 69
    assert sum(item[2] == 33 for item in actual) == 70


def test_number_alignment_retains_original_query_rows(monkeypatch):
    alignment = np.zeros((5, 8), dtype=int)
    alignment[2, 2] = alignment[3, 3] = alignment[4, 4] = 1

    def fake_scheme(states, sequence, scheme, chain_type):
        assert sequence == "--DEF"
        return [((3, ""), "D"), ((4, ""), "E"), ((5, ""), "F")], 2, 4

    monkeypatch.setattr(numbering, "_apply_scheme", fake_scheme)
    assert number_alignment(alignment, "ACDEF", "imgt", "H") == [
        (2, 3, "", "D"),
        (3, 4, "", "E"),
        (4, 5, "", "F"),
    ]


def test_scfv_numbering_uses_1000_block_and_sequential_linker(monkeypatch):
    alignment = np.zeros((5, 256), dtype=int)
    alignment[0, 0] = 1
    alignment[1, 1] = 1
    alignment[3, 128] = 1
    alignment[4, 129] = 1

    def fake_scheme(states, sequence, scheme, chain_type):
        if chain_type == "H":
            return [((1, ""), "A"), ((2, ""), "C")], 0, 1
        return [((1, ""), "E"), ((2, ""), "F")], 0, 1

    monkeypatch.setattr(numbering, "_apply_scheme", fake_scheme)
    assert number_alignment(alignment, "ACDEF", "imgt", "HK") == [
        (0, 1, "", "A"),
        (1, 2, "", "C"),
        (2, 3, "", "D"),
        (3, 1001, "", "E"),
        (4, 1002, "", "F"),
    ]


def test_higher_order_numbering_offsets_every_domain_and_linker(monkeypatch):
    alignment = np.zeros((8, 384), dtype=int)
    alignment[0, 0] = 1
    alignment[1, 1] = 1
    alignment[3, 128] = 1
    alignment[4, 129] = 1
    alignment[6, 256] = 1
    alignment[7, 257] = 1

    def fake_scheme(states, sequence, scheme, chain_type):
        return [((1, ""), sequence[0]), ((2, ""), sequence[1])], 0, 1

    monkeypatch.setattr(numbering, "_apply_scheme", fake_scheme)
    assert number_alignment(alignment, "ACDEFGHI", "imgt", "HHK") == [
        (0, 1, "", "A"),
        (1, 2, "", "C"),
        (2, 3, "", "D"),
        (3, 1001, "", "E"),
        (4, 1002, "", "F"),
        (5, 1003, "", "G"),
        (6, 2001, "", "H"),
        (7, 2002, "", "I"),
    ]


@pytest.mark.parametrize(
    ("records", "start", "end", "message"),
    [
        ([((1, ""), "A"), ((2, ""), "C")], -1, 0, "bounds"),
        ([((1, ""), "A")], 0, 1, "length"),
        ([((1, ""), "X"), ((2, ""), "C")], 0, 1, "expected"),
    ],
)
def test_number_alignment_rejects_invalid_anarci_output(
    monkeypatch, records, start, end, message
):
    monkeypatch.setattr(
        numbering,
        "_apply_scheme",
        lambda states, sequence, scheme, chain_type: (records, start, end),
    )
    with pytest.raises(ValueError, match=message):
        number_alignment(np.eye(2, dtype=int), "AC", "imgt", "H")
