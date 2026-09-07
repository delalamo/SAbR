import copy
from pathlib import Path

import numpy as np
import pytest
from Bio.PDB import Atom, Chain, PDBParser, Residue
from Bio.PDB.Structure import Structure

import sabr.api as api
from sabr import constants, renumber_structure
from sabr.structure import extract_chain

DATA = Path(__file__).parent / "data"


def _bio_ids(structure, chain):
    return [
        (residue.id[1], residue.id[2].strip())
        for residue in structure[0][chain]
        if not residue.id[0].strip()
    ]


def _stub_pipeline(monkeypatch, captured_modes=None, captured_options=None):
    def fake_encode(coords, mode):
        if captured_modes is not None:
            captured_modes["encoder"] = mode
        return np.zeros((len(coords), 64))

    def fake_align(embeddings, gaps, chain_type, noise, mode, scfv):
        if captured_modes is not None:
            captured_modes["alignment"] = mode
        if captured_options is not None:
            captured_options.update({"chain_type": chain_type, "scfv": scfv})
        return (
            np.zeros((len(embeddings), 128), dtype=int),
            "H" if chain_type == "auto" else chain_type,
            0.0,
        )

    monkeypatch.setattr(
        api,
        "encode",
        fake_encode,
    )
    monkeypatch.setattr(
        api,
        "align",
        fake_align,
    )
    monkeypatch.setattr(
        api,
        "number_alignment",
        lambda alignment, sequence, scheme, chain_type: (
            [
                (index, index + 1, "", amino_acid)
                for index, amino_acid in enumerate(sequence)
            ]
        ),
    )


def test_biopython_structure_is_copied_before_numbering(monkeypatch):
    _stub_pipeline(monkeypatch)
    path = DATA / "test_heavy_chain.pdb"
    structure = PDBParser(QUIET=True).get_structure("structure", path)
    original_ids = _bio_ids(structure, "F")
    structure.header["nested"] = ["original"]
    original_coords = np.array([atom.coord for atom in structure.get_atoms()])

    result = renumber_structure(structure, "F", chain_type="H")

    assert isinstance(result, Structure)
    assert result is not structure
    assert _bio_ids(structure, "F") == original_ids
    assert _bio_ids(result, "F") != original_ids
    np.testing.assert_array_equal(
        [atom.coord for atom in result.get_atoms()], original_coords
    )
    assert [r.resname for r in result.get_residues()] == [
        r.resname for r in structure.get_residues()
    ]
    next(result.get_atoms()).coord[0] += 100
    result.header["nested"].append("changed")
    np.testing.assert_array_equal(
        [atom.coord for atom in structure.get_atoms()], original_coords
    )
    assert structure.header["nested"] == ["original"]


def test_non_target_content_and_out_of_range_residues_are_preserved(
    monkeypatch,
):
    _stub_pipeline(monkeypatch)
    structure = PDBParser(QUIET=True).get_structure(
        "heavy", DATA / "test_heavy_chain.pdb"
    )
    other = Chain.Chain("Z")
    other_residue = Residue.Residue(("W", 1, " "), "HOH", " ")
    other_residue.add(
        Atom.Atom(
            "O",
            np.asarray((1.0, 2.0, 3.0)),
            1.0,
            1.0,
            " ",
            " O  ",
            1,
            element="O",
        )
    )
    other.add(other_residue)
    structure[0].add(other)
    original_other = copy.deepcopy(structure[0]["Z"])
    last_id = list(structure[0]["F"])[-1].id

    result = renumber_structure(
        structure,
        "F",
        chain_type="H",
        residue_range=(2, 127),
    )

    assert list(result[0]["F"])[-1].id == last_id
    assert list(structure[0]["F"])[-1].id == last_id
    assert result[0]["Z"][other_residue.id]["O"].coord.tolist() == (
        original_other[other_residue.id]["O"].coord.tolist()
    )


def test_numeric_residue_range_includes_insertion_codes():
    structure = PDBParser(QUIET=True).get_structure(
        "insertions", DATA / "test_insertion_codes.pdb"
    )
    data = extract_chain(structure, "A", (52, 52))
    assert data.residue_ids == ((52, ""), (52, "A"), (52, "B"))


def test_range_that_would_collide_with_unchanged_ids_is_rejected(
    monkeypatch,
):
    _stub_pipeline(monkeypatch)
    structure = PDBParser(QUIET=True).get_structure(
        "heavy", DATA / "test_heavy_chain.pdb"
    )
    unchanged = copy.deepcopy(list(structure[0]["F"])[0])
    unchanged.detach_parent()
    unchanged.id = (" ", 1, " ")
    structure[0]["F"].add(unchanged)
    with pytest.raises(ValueError, match="collide"):
        renumber_structure(
            structure,
            "F",
            chain_type="H",
            residue_range=(2, 128),
        )


def test_arbitrary_biopython_chain_ids_are_supported(monkeypatch):
    _stub_pipeline(monkeypatch)
    structure = PDBParser(QUIET=True).get_structure(
        "heavy", DATA / "test_heavy_chain.pdb"
    )
    target = structure[0]["F"]
    structure[0].detach_child("F")
    target.id = "heavy_chain"
    structure[0].add(target)
    result = renumber_structure(structure, "heavy_chain", chain_type="H")
    assert result[0]["heavy_chain"].id == "heavy_chain"


def test_softalign_mode_is_forwarded_to_encoder_and_alignment(monkeypatch):
    captured_modes = {}
    _stub_pipeline(monkeypatch, captured_modes)
    structure = PDBParser(QUIET=True).get_structure(
        "heavy", DATA / "test_heavy_chain.pdb"
    )
    renumber_structure(structure, "F", mode="softalign")
    assert captured_modes == {
        "encoder": "softalign",
        "alignment": "softalign",
    }


def test_structural_gap_stops_before_model_execution(monkeypatch):
    structure = PDBParser(QUIET=True).get_structure("gap", DATA / "8sve_L.pdb")

    def unexpected_encode(coords, mode):
        pytest.fail("encoder ran despite a structural gap")

    monkeypatch.setattr(api, "encode", unexpected_encode)
    with pytest.raises(
        ValueError,
        match="SAbR will not run.*--dangerously-allow-structural-gaps",
    ):
        renumber_structure(structure, "M")


def test_structural_gap_can_be_explicitly_allowed(monkeypatch):
    captured_modes = {}
    _stub_pipeline(monkeypatch, captured_modes)
    structure = PDBParser(QUIET=True).get_structure("gap", DATA / "8sve_L.pdb")

    result = renumber_structure(
        structure,
        "M",
        dangerously_allow_structural_gaps=True,
    )

    assert isinstance(result, Structure)
    assert captured_modes == {"encoder": "sabr", "alignment": "sabr"}


@pytest.mark.parametrize(
    ("chain_type", "normalized"),
    [
        ("k, h", "H,K"),
        ("HK,HL", "HK,HL"),
        ("hhk, hhl", "HHK,HHL"),
        ("H,H,K", "H,K"),
    ],
)
def test_chain_type_accepts_domain_candidate_sets(
    monkeypatch, chain_type, normalized
):
    captured = {}
    _stub_pipeline(monkeypatch, captured_options=captured)
    structure = PDBParser(QUIET=True).get_structure(
        "heavy", DATA / "test_heavy_chain.pdb"
    )
    renumber_structure(structure, "F", chain_type=chain_type)
    assert captured == {"chain_type": normalized, "scfv": False}


def test_scfv_is_the_documented_chain_type_alias(monkeypatch):
    captured = {}
    _stub_pipeline(monkeypatch, captured_options=captured)
    structure = PDBParser(QUIET=True).get_structure(
        "heavy", DATA / "test_heavy_chain.pdb"
    )
    renumber_structure(structure, "F", scfv=True)
    assert captured == {
        "chain_type": "HK,HL,KH,LH",
        "scfv": True,
    }


def test_chain_type_domain_choices_come_from_reference_asset(monkeypatch):
    monkeypatch.setattr(
        api,
        "load_references",
        lambda noise_level, mode: {"X": (None, None)},
    )
    options = api._RenumberOptions(
        scheme="imgt",
        chain_type="x",
        noise_level=0.0,
        residue_range=None,
        mode="sabr",
        scfv=False,
    )
    assert options.chain_type == "X"

    with pytest.raises(ValueError, match="auto, X"):
        api._RenumberOptions(
            scheme="imgt",
            chain_type="H",
            noise_level=0.0,
            residue_range=None,
            mode="sabr",
            scfv=False,
        )


@pytest.mark.parametrize("chain_type", constants.TCR_CHAIN_TYPES)
def test_tcr_chain_uses_k_reference_and_actual_numbering_type(
    monkeypatch, chain_type
):
    structure = PDBParser(QUIET=True).get_structure(
        "heavy", DATA / "test_heavy_chain.pdb"
    )
    captured = {}
    monkeypatch.setattr(
        api,
        "encode",
        lambda coords, mode: np.zeros((len(coords), 64)),
    )

    def fake_align(embeddings, gaps, requested_type, noise, mode, scfv):
        captured["alignment_type"] = requested_type
        return np.zeros((len(embeddings), 128), dtype=int), "K", 0.0

    def fake_number(alignment, sequence, scheme, numbering_type):
        captured["numbering_type"] = numbering_type
        return [
            (index, index + 1, "", amino_acid)
            for index, amino_acid in enumerate(sequence)
        ]

    monkeypatch.setattr(api, "align", fake_align)
    monkeypatch.setattr(api, "number_alignment", fake_number)

    renumber_structure(structure, "F", chain_type=chain_type.lower())

    assert captured == {
        "alignment_type": chain_type,
        "numbering_type": chain_type,
    }


def test_missing_backbone_is_reported_with_the_residue():
    structure = PDBParser(QUIET=True).get_structure(
        "missing", DATA / "test_no_backbone.pdb"
    )
    with pytest.raises(ValueError, match="missing required backbone"):
        renumber_structure(structure, "A")


@pytest.mark.parametrize(
    ("chain", "message"),
    [("", "non-empty"), ("missing", "not found")],
)
def test_invalid_or_missing_chain_is_reported(chain, message):
    structure = PDBParser(QUIET=True).get_structure(
        "heavy", DATA / "test_heavy_chain.pdb"
    )
    with pytest.raises(ValueError, match=message):
        renumber_structure(structure, chain)


def test_missing_backbone_outside_selected_range_is_ignored():
    structure = PDBParser(QUIET=True).get_structure(
        "heavy", DATA / "test_heavy_chain.pdb"
    )
    list(structure[0]["F"])[-1].detach_child("N")
    data = extract_chain(structure, "F", (2, 127))
    assert data.residue_ids[0] == (2, "")
    assert data.residue_ids[-1] == (127, "")


def test_multiple_models_are_rejected():
    structure = PDBParser(QUIET=True).get_structure(
        "heavy", DATA / "test_heavy_chain.pdb"
    )
    second = copy.deepcopy(structure[0])
    second.id = 1
    structure.add(second)
    with pytest.raises(ValueError, match="exactly one model"):
        renumber_structure(structure, "F")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"scheme": "unknown"}, "scheme"),
        ({"chain_type": "heavy"}, "chain_type"),
        ({"chain_type": "H,,K"}, "chain_type"),
        ({"chain_type": "auto,H"}, "chain_type"),
        ({"noise_level": 0.3}, "noise_level"),
        ({"residue_range": [1, 2]}, "residue_range"),
        ({"residue_range": (2, 1)}, "start"),
        ({"noise_level": "invalid"}, "noise_level"),
        ({"noise_level": False}, "noise_level"),
        ({"mode": "other"}, "mode"),
        ({"mode": None}, "mode"),
        ({"residue_range": (False, 10)}, "residue_range"),
        ({"scfv": "yes"}, "scfv"),
        ({"scfv": True, "chain_type": "H"}, "auto"),
        (
            {"dangerously_allow_structural_gaps": "yes"},
            "dangerously_allow_structural_gaps",
        ),
    ],
)
def test_invalid_public_options_fail_before_model_execution(kwargs, message):
    structure = PDBParser(QUIET=True).get_structure(
        "heavy", DATA / "test_heavy_chain.pdb"
    )
    with pytest.raises(ValueError, match=message):
        renumber_structure(structure, "F", **kwargs)


def test_unsupported_structure_type_raises_type_error():
    with pytest.raises(TypeError, match="Bio.PDB"):
        renumber_structure(object(), "A")


@pytest.mark.parametrize("chain_type", constants.TCR_CHAIN_TYPES)
@pytest.mark.parametrize("scheme", ("chothia", "kabat", "martin", "wolfguy"))
def test_invalid_tcr_scheme_fails_before_encoding(
    monkeypatch, chain_type, scheme
):
    monkeypatch.setattr(
        api,
        "encode",
        lambda *args: pytest.fail("unsupported TCR scheme reached encoder"),
    )
    structure = PDBParser(QUIET=True).get_structure(
        "heavy", DATA / "test_heavy_chain.pdb"
    )
    with pytest.raises(ValueError, match="only IMGT or AHo"):
        renumber_structure(structure, "F", chain_type=chain_type, scheme=scheme)
