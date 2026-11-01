"""Selections too short to align or number must fail with clear errors.

Regression tests: one-residue selections previously raised a NumPy
zero-size reduction error inside the aligner, and two- to four-residue
selections raised ``TypeError`` when the numbering scheme returned no
sequence bounds. Both arise in deposited structures whose antibody chains
lack modelled variable domains (for example constant-domain-only chains).
"""

import copy
from pathlib import Path

import numpy as np
import pytest
from Bio.PDB import PDBParser

from sabr import renumber_structure
from sabr.alignment import _affine_value_and_grad, align

DATA = Path(__file__).parent / "data"
SHORT_SELECTION = r"at least two residues|No residues could be numbered"


@pytest.fixture(scope="module")
def heavy_chain():
    return PDBParser(QUIET=True).get_structure(
        "heavy", DATA / "test_heavy_chain.pdb"
    )


def _truncated(structure, chain, count):
    truncated = copy.deepcopy(structure)
    for residue in list(truncated[0][chain])[count:]:
        truncated[0][chain].detach_child(residue.id)
    return truncated


@pytest.mark.parametrize("count", [1, 2, 3, 4])
def test_short_chain_raises_clear_value_error(heavy_chain, count):
    structure = _truncated(heavy_chain, "F", count)
    with pytest.raises(ValueError, match=SHORT_SELECTION):
        renumber_structure(structure, "F")


@pytest.mark.parametrize("count", [1, 2, 3, 4])
def test_short_residue_range_raises_clear_value_error(heavy_chain, count):
    ids = [residue.id[1] for residue in heavy_chain[0]["F"]]
    with pytest.raises(ValueError, match=SHORT_SELECTION):
        renumber_structure(
            heavy_chain, "F", residue_range=(ids[0], ids[count - 1])
        )


@pytest.mark.parametrize("count", [1, 2])
def test_short_chain_with_explicit_chain_type(heavy_chain, count):
    structure = _truncated(heavy_chain, "F", count)
    with pytest.raises(ValueError, match=SHORT_SELECTION):
        renumber_structure(structure, "F", chain_type="H")


def test_align_rejects_single_residue_query():
    with pytest.raises(ValueError, match="at least two residues"):
        align(np.zeros((1, 64), dtype=np.float32), frozenset(), "auto", 0.0)


def test_affine_alignment_enforces_documented_minimum_size():
    similarities = np.zeros((1, 5), dtype=np.float32)
    with pytest.raises(ValueError, match="at least two residues"):
        _affine_value_and_grad(similarities, (1, 5), 1.0, 0.0, 0.0)


def test_short_selection_does_not_mutate_input(heavy_chain):
    structure = _truncated(heavy_chain, "F", 3)
    before = [residue.id for residue in structure[0]["F"]]
    with pytest.raises(ValueError):
        renumber_structure(structure, "F")
    assert [residue.id for residue in structure[0]["F"]] == before
