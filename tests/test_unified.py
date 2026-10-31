"""Unified structure-consistent Ig-fold numbering."""

from pathlib import Path

import numpy as np
import pytest
from Bio.PDB import PDBParser

from sabr import constants, unified
from sabr.api import renumber_structure
from sabr.model import encode
from sabr.structure import extract_chain

DATA = Path(__file__).parent / "data"


@pytest.mark.parametrize(
    ("length", "anchors", "expected"),
    [
        (5, {0: "1", 4: "5"}, ["1", "2", "3", "4", "5"]),
        # short loop: numbers come from both ends, the gap sits in the middle
        (4, {0: "105", 3: "117"}, ["105", "106", "116", "117"]),
        # termini count outward through integers, including zero
        (6, {2: "3", 3: "4"}, ["1", "2", "3", "4", "5", "6"]),
        (3, {1: "1"}, ["0", "1", "2"]),
        # letter sites are never loop slots; insertions attach N-terminally
        (5, {0: "8", 2: "8A", 4: "9"}, ["8", "8.1", "8A", "8A.1", "9"]),
    ],
)
def test_fill_loops_rules(length, anchors, expected):
    assert unified.fill_loops(length, anchors) == expected


def test_long_loop_insertions_sit_at_the_apex():
    labels = unified.fill_loops(17, {0: "107", 16: "115"})
    assert labels[:5] == ["107", "108", "109", "110", "111"]
    assert labels[5:13] == [f"111.{k}" for k in range(1, 9)]
    assert labels[13:] == ["112", "113", "114", "115"]


def test_labels_sort_in_chain_order():
    labels = [
        "-1",
        "0",
        "8",
        "8.1",
        "8.2",
        "8A",
        "8A.1",
        "8B",
        "8Z",
        "8AA",
        "9",
    ]
    shuffled = labels[5:] + labels[:5]
    assert sorted(shuffled, key=unified.label_key) == labels


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("23", (23, "")),
        ("8A", (8, "A")),
        ("8A.1", (8, "A1")),
        ("111.12", (111, "12")),
        ("-2", (-2, "")),
    ],
)
def test_split_label(label, expected):
    assert unified.split_label(label) == expected
    assert unified.join_label(*expected) == label


def test_reference_is_one_ordered_set():
    embeddings, labels = unified.load_reference()
    assert embeddings.shape == (len(labels), constants.EMBED_DIM)
    assert np.isfinite(embeddings).all()
    assert list(labels) == sorted(labels, key=unified.label_key)
    assert len(set(labels)) == len(labels)
    for conserved in ("23", "41", "89", "104", "118"):
        assert conserved in labels


def _numbered(path, chain):
    structure = PDBParser(QUIET=True).get_structure("x", path)
    data = extract_chain(structure, chain, None)
    records = unified.number_embeddings(encode(data.coords), data.gap_indices)
    return data, records


@pytest.mark.parametrize(
    ("path", "chain", "domains"),
    [
        ("test_heavy_chain.pdb", "F", 1),
        ("12e8_L.pdb", "L", 2),
        ("1bjm_A.pdb", "A", 2),
    ],
)
def test_real_chains_number_conserved_core(path, chain, domains):
    data, records = _numbered(DATA / path, chain)
    assert [row for row, _, _ in records] == list(range(len(data.sequence)))
    keys = [unified.label_key(unified.join_label(n, c)) for _, n, c in records]
    assert keys == sorted(keys)
    by_number = {
        (number, code): data.sequence[row] for row, number, code in records
    }
    assert by_number[(23, "")] == "C"
    assert by_number[(41, "")] == "W"
    assert by_number[(104, "")] == "C"
    offsets = {
        number // constants.DOMAIN_NUMBERING_STRIDE
        for _, number, _ in records
        if number > 0
    }
    assert len(offsets) == domains
    if domains == 2:
        stride = constants.DOMAIN_NUMBERING_STRIDE
        assert by_number[(stride + 23, "")] == "C"
        assert by_number[(stride + 104, "")] == "C"


def test_public_api_unified_scheme():
    structure = PDBParser(QUIET=True).get_structure("x", DATA / "12e8_L.pdb")
    result = renumber_structure(structure, "L", scheme="unified")
    chain = next(iter(result))["L"]
    residues = {residue.id[1:]: residue.get_resname() for residue in chain}
    assert residues[(23, " ")] == "CYS"
    assert residues[(104, " ")] == "CYS"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"chain_type": "K"},
        {"mode": "softalign"},
        {"noise_level": 0.5},
        {"scfv": True},
    ],
)
def test_unified_scheme_rejects_reference_options(kwargs):
    structure = PDBParser(QUIET=True).get_structure("x", DATA / "12e8_L.pdb")
    with pytest.raises(ValueError):
        renumber_structure(structure, "L", scheme="unified", **kwargs)


def test_cli_writes_unified_numbering(tmp_path):
    from click.testing import CliRunner

    import sabr.cli as cli

    output = tmp_path / "unified.pdb"
    result = CliRunner().invoke(
        cli.main,
        [
            "-i",
            str(DATA / "test_heavy_chain.pdb"),
            "-c",
            "F",
            "-o",
            str(output),
            "-n",
            "unified",
        ],
    )
    assert result.exit_code == 0, result.output
    written = output if output.exists() else output.with_suffix(".cif")
    assert written.exists()


def test_no_domain_raises():
    rng = np.random.default_rng(0)
    embeddings = rng.normal(0, 0.01, (50, constants.EMBED_DIM))
    with pytest.raises(ValueError, match="No immunoglobulin-fold domain"):
        unified.number_embeddings(embeddings.astype(np.float32))


def test_invalid_labels_raise():
    with pytest.raises(ValueError):
        unified.label_key("A12")
    with pytest.raises(ValueError):
        unified.split_label("12-")
    with pytest.raises(ValueError):
        unified.fill_loops(3, {})


def test_too_short_query_has_no_anchors():
    anchors, score = unified.assign_anchors(np.zeros((1, constants.EMBED_DIM)))
    assert anchors == {} and score == float("-inf")


@pytest.mark.parametrize("case_id", ("1ubq_A", "1lyz_A"))
def test_unified_scheme_rejects_non_ig_controls(case_id):
    path = DATA / "antibody" / "fixtures" / f"{case_id}.pdb"
    structure = PDBParser(QUIET=True).get_structure(case_id, path)
    with pytest.raises(ValueError, match="No immunoglobulin-fold domain"):
        renumber_structure(structure, "A", scheme="unified")
