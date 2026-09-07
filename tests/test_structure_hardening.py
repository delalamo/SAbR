import json
from pathlib import Path

import numpy as np
import pytest
from Bio.PDB import PDBParser

from sabr import constants, renumber_structure
from sabr.structure import (
    _detect_gaps,
    _new_residue_ids,
    _parent_residue_name,
    _select_backbone,
    extract_chain,
)

DATA = Path(__file__).parent / "data"
ASSET = (
    Path(__file__).parents[1]
    / "src"
    / "sabr"
    / "assets"
    / "modified_residues.json"
)


def _atom(name, altloc, occupancy, coordinate):
    return name, altloc, occupancy, np.asarray(coordinate, dtype=float)


def test_altloc_selection_prefers_blank_then_occupancy_then_name():
    blank = [
        _atom(name, "", 0.1, (index, 0, 0))
        for index, name in enumerate(("N", "CA", "C"))
    ]
    named = [
        _atom(name, "A", 1.0, (index, 1, 0))
        for index, name in enumerate(("N", "CA", "C"))
    ]
    assert all(coord[1] == 0 for coord in _select_backbone(blank + named, "x"))

    a = [
        _atom(name, "A", 0.4, (index, 1, 0))
        for index, name in enumerate(("N", "CA", "C"))
    ]
    b = [
        _atom(name, "B", 0.8, (index, 2, 0))
        for index, name in enumerate(("N", "CA", "C"))
    ]
    assert all(coord[1] == 2 for coord in _select_backbone(a + b, "x"))
    tied_b = [
        _atom(name, "B", 0.4, (index, 2, 0))
        for index, name in enumerate(("N", "CA", "C"))
    ]
    for atoms in (a + tied_b, tied_b + a):
        assert all(coord[1] == 1 for coord in _select_backbone(atoms, "x"))
    shared = [_atom("N", "", 0.1, (0, 3, 0))]
    selected = _select_backbone(shared + a[1:] + b[1:], "x")
    assert [coord[1] for coord in selected] == [3, 2, 2]


def test_altloc_selection_rejects_incomplete_mixed_conformers():
    atoms = [
        _atom("N", "A", 1.0, (0, 0, 0)),
        _atom("CA", "B", 1.0, (1, 0, 0)),
        _atom("C", "A", 1.0, (2, 0, 0)),
    ]
    with pytest.raises(ValueError, match="incompatible altlocs"):
        _select_backbone(atoms, "test residue")


def test_modified_residue_snapshot_provenance_and_representative_mappings():
    document = json.loads(ASSET.read_text())
    assert document["source"].startswith("https://files.wwpdb.org/")
    assert document["snapshot_date"] == "2026-07-11"
    assert document["source_sha256"] == (
        "0b3323123ec10b997afe1c530b4cad30306e60b451b2b062c59bc9bb5cbe0679"
    )
    assert len(document["mapping"]) == 1107
    assert {
        name: _parent_residue_name(name)
        for name in ("MSE", "SEP", "TPO", "PTR", "CSO")
    } == {
        "MSE": "MET",
        "SEP": "SER",
        "TPO": "THR",
        "PTR": "TYR",
        "CSO": "CYS",
    }
    assert _parent_residue_name("DAL") is None


def test_modified_residue_uses_parent_for_sequence_without_mutation():
    structure = PDBParser(QUIET=True).get_structure(
        "heavy", DATA / "test_heavy_chain.pdb"
    )
    residue = list(structure[0]["F"])[0]
    residue.resname = "MSE"
    data = extract_chain(structure, "F", (2, 2))
    assert data.sequence == "M"
    assert residue.resname == "MSE"


def test_unknown_atom_polymer_and_oversized_selection_fail(monkeypatch):
    structure = PDBParser(QUIET=True).get_structure(
        "heavy", DATA / "test_heavy_chain.pdb"
    )
    list(structure[0]["F"])[0].resname = "ZZZ"
    with pytest.raises(ValueError, match="Unsupported polymer residue ZZZ"):
        extract_chain(structure, "F", (2, 2))

    structure = PDBParser(QUIET=True).get_structure(
        "heavy", DATA / "test_heavy_chain.pdb"
    )
    monkeypatch.setattr(constants, "MAX_SELECTED_RESIDUES", 1)
    with pytest.raises(ValueError, match="safety limit.*residue_range"):
        extract_chain(structure, "F", None)


def test_ambiguous_and_unknown_peptide_chemistry_fails_explicitly():
    structure = PDBParser(QUIET=True).get_structure(
        "heavy", DATA / "test_heavy_chain.pdb"
    )
    list(structure[0]["F"])[0].resname = "DAL"
    with pytest.raises(ValueError, match="Unsupported polymer residue DAL"):
        extract_chain(structure, "F", (2, 2))

    structure = PDBParser(QUIET=True).get_structure(
        "heavy", DATA / "test_heavy_chain.pdb"
    )
    residue = list(structure[0]["F"])[0]
    residue.resname = "ZZZ"
    residue.id = ("H_ZZZ", residue.id[1], residue.id[2])
    with pytest.raises(ValueError, match="Unsupported peptide-linked residue"):
        extract_chain(structure, "F", None)


def test_unrelated_unknown_hetero_is_preserved_but_not_sequenced():
    structure = PDBParser(QUIET=True).get_structure(
        "heavy", DATA / "test_heavy_chain.pdb"
    )
    residue = list(structure[0]["F"])[0]
    residue.resname = "ZZZ"
    residue.id = ("H_ZZZ", residue.id[1], residue.id[2])
    for atom in residue:
        atom.coord += 1000
    data = extract_chain(structure, "F", None)
    assert len(data.sequence) == len(list(structure[0]["F"])) - 1
    assert residue in list(structure[0]["F"])


def test_query_row_mapping_validates_order_span_and_sequence():
    structure = PDBParser(QUIET=True).get_structure(
        "heavy", DATA / "test_heavy_chain.pdb"
    )
    data = extract_chain(structure, "F", (2, 6))
    records = [
        (1, 2, "", data.sequence[1]),
        (2, 3, "", data.sequence[2]),
        (3, 4, "", data.sequence[3]),
    ]
    mapping = _new_residue_ids(data, records)
    assert [mapping[index] for index in data.residue_indices] == [
        (1, ""),
        (2, ""),
        (3, ""),
        (4, ""),
        (5, ""),
    ]

    non_positive_mapping = _new_residue_ids(
        data, [(2, 1, "", data.sequence[2])]
    )
    assert [non_positive_mapping[index] for index in data.residue_indices] == [
        (-1, ""),
        (0, ""),
        (1, ""),
        (2, ""),
        (3, ""),
    ]

    for malformed, message in (
        ([records[1], records[0]], "unordered"),
        ([records[0], records[2]], "internal"),
        ([(99, 1, "", "A")], "outside"),
        ([(1, 2, "", "X")], "mismatch"),
    ):
        with pytest.raises(ValueError, match=message):
            _new_residue_ids(data, malformed)


@pytest.mark.parametrize(
    "distance,expected", [(2.659999, set()), (2.66, set()), (2.660001, {0})]
)
def test_gap_detection_threshold_is_strict(distance, expected):
    coords = np.zeros((2, 4, 3))
    coords[1, 0, 0] = distance
    assert _detect_gaps(coords) == expected


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_coordinates_raise_validation_error_without_arithmetic_errors(
    value,
):
    structure = PDBParser(QUIET=True).get_structure(
        "heavy", DATA / "test_heavy_chain.pdb"
    )
    next(structure.get_atoms()).coord[0] = value
    with (
        np.errstate(invalid="raise", divide="raise"),
        pytest.raises(ValueError, match="non-finite coordinates"),
    ):
        renumber_structure(structure, "F")
