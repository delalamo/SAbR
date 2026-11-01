"""Structures assembled in code (without a ``header``) must be renumbered.

Regression tests: ``Bio.PDB.Structure.Structure`` objects built
programmatically have no ``header`` attribute (only parsers set it).
``renumber_structure`` previously raised ``AttributeError`` for them, both
without ``mmcif_dict`` (missing-residue lookup) and with it (stale metadata
cleanup on the returned copy).
"""

import copy
from pathlib import Path

import pytest
from Bio.PDB import PDBParser
from Bio.PDB.Model import Model
from Bio.PDB.Structure import Structure

from sabr import renumber_structure

DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def parsed():
    return PDBParser(QUIET=True).get_structure(
        "heavy", DATA / "test_heavy_chain.pdb"
    )


def _assembled(parsed, chain):
    structure = Structure("assembled")
    model = Model(0)
    structure.add(model)
    model.add(copy.deepcopy(parsed[0][chain]))
    return structure


def _ids(structure, chain):
    return [residue.id for residue in structure[0][chain]]


@pytest.mark.parametrize("mmcif_dict", [None, {}])
def test_structure_without_header_is_renumbered(parsed, mmcif_dict):
    assembled = _assembled(parsed, "F")
    assert not hasattr(assembled, "header")
    expected = renumber_structure(parsed, "F", chain_type="H")
    result = renumber_structure(
        assembled, "F", chain_type="H", mmcif_dict=mmcif_dict
    )
    assert _ids(result, "F") == _ids(expected, "F")


def test_structure_without_header_is_not_mutated(parsed):
    assembled = _assembled(parsed, "F")
    before = _ids(assembled, "F")
    renumber_structure(assembled, "F", chain_type="H")
    assert _ids(assembled, "F") == before
    assert not hasattr(assembled, "header")


def test_structure_with_none_header_is_renumbered(parsed):
    assembled = _assembled(parsed, "F")
    assembled.header = None
    result = renumber_structure(assembled, "F", chain_type="H")
    expected = renumber_structure(parsed, "F", chain_type="H")
    assert _ids(result, "F") == _ids(expected, "F")
