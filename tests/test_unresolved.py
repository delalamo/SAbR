"""Metadata-backed loops keep numbering from their complete sequence."""

import io

import numpy as np
import pytest
from Bio.PDB import MMCIFIO, MMCIFParser, PDBParser
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

from sabr.unresolved import (
    insert_ghost_embeddings,
    missing_between,
    missing_residues_from_mmcif,
)


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
    parsed = MMCIFParser(QUIET=True).get_structure("test", io.StringIO(text))
    metadata = MMCIF2Dict(io.StringIO(text))
    assert "missing_residues" not in parsed.header
    assert missing_residues_from_mmcif(metadata) == [
        {
            "chain": "F",
            "ssseq": 27,
            "res_name": "GLY",
            "insertion": "A",
            "model": 1,
        }
    ]
    assert missing_between(parsed, "F", ((27, ""), (30, "")), metadata) == {
        0: "G"
    }


@pytest.mark.parametrize(
    "positions", [(28, 29), (58, 59), (81, 82), (108, 109)]
)
@pytest.mark.parametrize("format", ["pdb", "cif", "cif-api"])
def test_public_api_with_unresolved_metadata(
    positions, format, tmp_path, monkeypatch
):
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
    if format.startswith("cif"):
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
    options = {}
    if format == "cif-api":
        parsed = MMCIFParser(QUIET=True).get_structure("test", path)
        options["mmcif_dict"] = MMCIF2Dict(path)
        assert "missing_residues" not in parsed.header
    import sabr.api as api

    real_encode, real_align = api.encode, api.align
    observed_count = len(residues) - len(removed)
    calls = []

    def check_encode(coords, mode):
        assert len(coords) == observed_count
        calls.append("encode")
        return real_encode(coords, mode)

    def check_align(embeddings, gaps, *args, **kwargs):
        assert calls == ["encode"]
        assert len(embeddings) == len(residues)
        assert np.count_nonzero(~np.any(embeddings, axis=1)) == len(removed)
        assert gaps == frozenset()
        calls.append("align")
        return real_align(embeddings, gaps, *args, **kwargs)

    monkeypatch.setattr(api, "encode", check_encode)
    monkeypatch.setattr(api, "align", check_align)
    output = renumber_structure(parsed, case["chain"], **options)
    assert calls == ["encode", "align"]
    assert [(r.id[1], r.id[2].strip()) for r in output[0][case["chain"]]] == [
        (e["imgt_position"], e["insertion_code"])
        for e in labels
        if e["row"] not in removed
    ]
    assert len(list(output.get_atoms())) == len(list(parsed.get_atoms()))
    assert not output.header.get("missing_residues")
    if format == "cif-api":
        assert "missing_residues" not in parsed.header
        assert len(missing_residues_from_mmcif(options["mmcif_dict"])) == len(
            removed
        )
    else:
        assert len(parsed.header["missing_residues"]) == len(removed)


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
    # Explicit mmCIF data takes precedence, even with no missing category.
    assert missing_between(structure, "F", ((26, ""), (28, "")), {}) == {}


def test_ghost_vectors_and_unexplained_gap_indices():
    embeddings = np.arange(5 * 64, dtype=np.float32).reshape(5, 64)
    original = embeddings.copy()
    expanded, sequence, gaps, rows = insert_ghost_embeddings(
        embeddings, "ACDEF", frozenset({0, 2, 3}), {0: "GH", 3: "I"}
    )
    assert sequence == "AGHCDEIF"
    assert rows == (0, None, None, 1, 2, 3, None, 4)
    assert gaps == frozenset({4})
    assert expanded.dtype == embeddings.dtype
    np.testing.assert_array_equal(expanded[[1, 2, 6]], 0)
    np.testing.assert_array_equal(expanded[[0, 3, 4, 5, 7]], original)
    np.testing.assert_array_equal(embeddings, original)
    unchanged, sequence, gaps, rows = insert_ghost_embeddings(
        embeddings, "ACDEF", frozenset({2}), {}
    )
    assert unchanged is embeddings
    assert sequence == "ACDEF"
    assert gaps == frozenset({2})
    assert rows == tuple(range(5))
