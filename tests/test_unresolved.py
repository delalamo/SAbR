"""Metadata-backed loops keep numbering from their complete sequence."""

import io

import numpy as np
import pytest
from Bio.PDB import MMCIFIO, PDBParser

from sabr.corrections import apply_corrections
from sabr.io import AnisouMMCIFParser
from sabr.numbering import number_alignment
from sabr.unresolved import expand_loop_alignment, missing_between


@pytest.mark.parametrize(
    "scheme", ["imgt", "aho", "kabat", "chothia", "martin"]
)
@pytest.mark.parametrize(
    "deleted",
    [
        (23, 24),
        (38,),
        (27, 28),
        (56, 60),
        (80, 81),
        (105, 110),
        tuple(range(26, 38)),
    ],
)
def test_numbering_matches_complete_loop(scheme, deleted):
    full = apply_corrections(np.eye(128, dtype=int))
    sequence = "A" * 128
    expected = number_alignment(full, sequence, scheme, "H")
    keep = [row for row in range(128) if row not in deleted]
    alignment = full[keep]
    missing = {}
    for row, (left, right) in enumerate(zip(keep, keep[1:])):
        if right - left > 1:
            missing[row] = "A" * (right - left - 1)
    expanded, letters, rows = expand_loop_alignment(
        alignment, "A" * len(keep), frozenset(missing), missing
    )
    actual = number_alignment(expanded, letters, scheme, "H")
    assert [
        (rows[r], n, i, a) for r, n, i, a in actual if rows[r] is not None
    ] == [(keep.index(r), n, i, a) for r, n, i, a in expected if r in keep]


@pytest.mark.parametrize("insert_at", [111, 82])
def test_missing_insertion_keeps_insertion_codes(insert_at):
    full = np.insert(np.eye(128, dtype=int), [insert_at] * 4, 0, axis=0)
    full = apply_corrections(full)
    expected = number_alignment(full, "A" * len(full), "imgt", "H")
    deleted = insert_at + 1
    keep = [r for r in range(len(full)) if r != deleted]
    expanded, sequence, rows = expand_loop_alignment(
        full[keep],
        "A" * len(keep),
        frozenset({deleted - 1}),
        {deleted - 1: "A"},
    )
    actual = number_alignment(expanded, sequence, "imgt", "H")
    assert [(n, i) for r, n, i, a in actual if rows[r] is not None] == [
        (n, i) for r, n, i, a in expected if r != deleted
    ]


def test_unexplained_gap_or_missing_anchor_preserves_alignment():
    alignment = np.eye(128, dtype=int)
    for gaps, missing in [
        (frozenset({26, 28}), {26: "A"}),
        (frozenset(), {1: "A"}),
    ]:
        result, sequence, rows = expand_loop_alignment(
            alignment, "A" * 128, gaps, missing
        )
        np.testing.assert_array_equal(result, alignment)
    alignment[:, 22] = 0
    result, _, _ = expand_loop_alignment(
        alignment, "A" * 128, frozenset({26}), {26: "A"}
    )
    np.testing.assert_array_equal(result, alignment)


def test_pdb_metadata_chain_model_and_insertion_codes():
    from pathlib import Path

    path = Path(__file__).parent / "data" / "test_heavy_chain.pdb"
    structure = PDBParser(QUIET=True).get_structure("test", path)
    record = {
        "chain": "F",
        "model": None,
        "ssseq": 27,
        "insertion": "A",
        "res_name": "GLY",
    }
    structure.header["missing_residues"] = [
        record,
        {**record, "chain": "X", "ssseq": 28},
        {**record, "model": 999, "ssseq": 29},
    ]
    assert missing_between(structure, "F", ((27, ""), (27, "B"), (30, ""))) == {
        0: "G"
    }
    assert missing_between(structure, "F", ((1, ""), (10, ""))) == {}


def test_cif_metadata_is_retained(tmp_path):
    from pathlib import Path

    path = Path(__file__).parent / "data" / "test_heavy_chain.pdb"
    structure = PDBParser(QUIET=True).get_structure("test", path)
    writer = MMCIFIO()
    writer.set_structure(structure)
    buffer = io.StringIO()
    writer.save(buffer)
    category = "_pdbx_unobs_or_zero_occ_residues."
    names = [
        "auth_asym_id",
        "auth_seq_id",
        "auth_comp_id",
        "PDB_ins_code",
        "PDB_model_num",
        "polymer_flag",
        "occupancy_flag",
    ]
    text = (
        buffer.getvalue()
        + "loop_\n"
        + "\n".join(category + name for name in names)
        + "\nF 27 GLY A 1 Y 1\nF 28 ALA ? 1 Y 0\nF 29 ALA ? 1 N 1\n#\n"
    )
    parsed = AnisouMMCIFParser(QUIET=True).get_structure(
        "test", io.StringIO(text)
    )
    assert parsed.header["missing_residues"] == [
        {
            "chain": "F",
            "ssseq": 27,
            "res_name": "GLY",
            "insertion": "A",
            "model": 1,
        }
    ]
    assert missing_between(parsed, "F", ((27, ""), (30, ""))) == {0: "G"}


@pytest.mark.parametrize(
    "positions", [(28, 29), (58, 59), (81, 82), (108, 109)]
)
@pytest.mark.parametrize("format", ["pdb", "cif"])
def test_public_api_with_unresolved_metadata(positions, format, tmp_path):
    from antibody_cases import load_case, load_cases
    from Bio.PDB import PDBIO

    from sabr import renumber_structure
    from sabr.cli import _read_structure

    case = next(c for c in load_cases() if c["id"] == "1mel_A")
    structure, labels = load_case(case)
    chain = structure[0][case["chain"]]
    residues = list(chain)
    removed = [e["row"] for e in labels if e["imgt_position"] in positions]
    assert removed
    remarks = []
    for row in removed:
        residue = residues[row]
        remarks.append(
            f"REMARK 465     {residue.resname:3s} {chain.id} "
            f"{residue.id[1]:5d}{residue.id[2]}\n"
        )
        chain.detach_child(residue.id)
    writer = PDBIO()
    writer.set_structure(structure)
    buffer = io.StringIO()
    writer.save(buffer)
    path = tmp_path / "missing.pdb"
    path.write_text("".join(remarks) + buffer.getvalue())
    parsed = _read_structure(path)
    assert len(parsed.header["missing_residues"]) == len(removed)
    if format == "cif":
        writer = MMCIFIO()
        writer.set_structure(parsed)
        buffer = io.StringIO()
        writer.save(buffer)
        names = ("auth_asym_id", "auth_seq_id", "auth_comp_id", "PDB_ins_code")
        category = "_pdbx_unobs_or_zero_occ_residues."
        text = (
            buffer.getvalue()
            + "loop_\n"
            + "\n".join(category + name for name in names)
            + "\n"
        )
        for record in parsed.header["missing_residues"]:
            text += (
                f"{record['chain']} {record['ssseq']} {record['res_name']} "
                f"{record['insertion'] or '?'}\n"
            )
        path = tmp_path / "missing.cif"
        path.write_text(text + "#\n")
        parsed = _read_structure(path)
    with pytest.warns(UserWarning, match="Skipping"):
        output = renumber_structure(parsed, case["chain"])
    assert [(r.id[1], r.id[2].strip()) for r in output[0][case["chain"]]] == [
        (e["imgt_position"], e["insertion_code"])
        for e in labels
        if e["row"] not in removed
    ]
    assert len(list(output.get_atoms())) == len(list(parsed.get_atoms()))
    assert output.header["missing_residues"] == []
    assert len(parsed.header["missing_residues"]) == len(removed)


def test_two_domains_keep_absolute_observed_rows():
    full = np.eye(256, dtype=int)
    apply_corrections(full[:, :128])
    apply_corrections(full[:, 128:])
    keep = [row for row in range(256) if row not in (27, 209)]
    expected = number_alignment(full, "A" * 256, "imgt", "HK")
    missing = {26: "A", 207: "A"}
    expanded, sequence, rows = expand_loop_alignment(
        full[keep], "A" * len(keep), frozenset(missing), missing
    )
    actual = number_alignment(expanded, sequence, "imgt", "HK")
    assert [
        (rows[r], n, i, a) for r, n, i, a in actual if rows[r] is not None
    ] == [(keep.index(r), n, i, a) for r, n, i, a in expected if r in keep]


def test_ambiguous_metadata_does_not_expand_a_gap():
    from pathlib import Path

    structure = PDBParser(QUIET=True).get_structure(
        "test", Path(__file__).parent / "data" / "test_heavy_chain.pdb"
    )
    record = {"chain": "F", "model": None, "ssseq": 27, "res_name": "GLY"}
    for records in [
        [record, {**record, "res_name": "ALA"}],
        [{**record, "ssseq": "?"}],
        [{**record, "ssseq": 26}],
    ]:
        structure.header["missing_residues"] = records
        assert missing_between(structure, "F", ((26, ""), (28, ""))) == {}
    structure.header["missing_residues"] = [record, record]
    assert missing_between(structure, "F", ((26, ""), (28, ""))) == {0: "G"}
    assert missing_between(structure, "F", ((28, ""), (26, ""))) == {}
