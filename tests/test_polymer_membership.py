"""HETATM chemistry alone does not establish peptide membership."""

import copy
from pathlib import Path

import numpy as np
import pytest
from Bio.PDB import PDBParser

from sabr.structure import apply_numbering, extract_chain

DATA = Path(__file__).parent / "data"


@pytest.fixture
def structure():
    return PDBParser(QUIET=True).get_structure(
        "heavy", DATA / "test_heavy_chain.pdb"
    )


@pytest.mark.parametrize("resname", ["LEU", "MSE", "SEP"])
def test_disconnected_amino_acid_ligands_are_preserved(structure, resname):
    chain = structure[0]["F"]
    before = extract_chain(structure, "F", None)
    ligand = copy.deepcopy(chain.child_list[0])
    ligand.detach_parent()
    ligand.resname = resname
    ligand.id = ("H_" + resname, 900, " ")
    for atom in ligand:
        atom.coord += 1000
    chain.add(ligand)
    original_ids = [residue.id for residue in chain]
    original_coords = np.array([atom.coord.copy() for atom in ligand])

    after = extract_chain(structure, "F", None)
    assert after.sequence == before.sequence
    assert after.residue_ids == before.residue_ids
    assert after.residue_indices == before.residue_indices
    assert after.gap_indices == before.gap_indices
    np.testing.assert_array_equal(after.coords, before.coords)

    numbered = [(i, i + 100, "", aa) for i, aa in enumerate(after.sequence)]
    result = apply_numbering(structure, "F", after, numbered)
    output_ligand = result[0]["F"][ligand.id]
    assert output_ligand.resname == resname
    np.testing.assert_array_equal(
        [atom.coord for atom in output_ligand], original_coords
    )
    assert [residue.id for residue in chain] == original_ids
    np.testing.assert_array_equal(
        [atom.coord for atom in ligand], original_coords
    )
    assert result[0]["F"].child_list[0].id[1] == 100


@pytest.mark.parametrize("resname,letter", [("LEU", "L"), ("MSE", "M")])
@pytest.mark.parametrize("index", [0, 10, -1])
def test_linked_hetero_residue_is_included_in_single_residue_range(
    structure, resname, letter, index
):
    residue = structure[0]["F"].child_list[index]
    residue.resname = resname
    residue.id = ("H_" + resname, residue.id[1], residue.id[2])
    data = extract_chain(structure, "F", (residue.id[1], residue.id[1]))
    assert data.sequence == letter
    assert data.residue_ids == ((residue.id[1], residue.id[2].strip()),)


def test_connected_modified_residue_with_missing_ca_still_fails(structure):
    residue = structure[0]["F"].child_list[10]
    residue.resname = "MSE"
    residue.id = ("H_MSE", residue.id[1], residue.id[2])
    residue.detach_child("CA")
    with pytest.raises(ValueError, match="missing required backbone atom.*CA"):
        extract_chain(structure, "F", None)


def test_atom_polymer_fragments_are_kept_across_chain_breaks(structure):
    before = extract_chain(structure, "F", None)
    for residue in structure[0]["F"].child_list[10:]:
        for atom in residue:
            atom.coord += 1000
    after = extract_chain(structure, "F", None)
    assert after.sequence == before.sequence
    assert after.residue_indices == before.residue_indices
    assert 9 in after.gap_indices


def test_unknown_linked_hetero_is_rejected_even_in_isolated_range(structure):
    residue = structure[0]["F"].child_list[10]
    residue.resname = "ZZZ"
    residue.id = ("H_ZZZ", residue.id[1], residue.id[2])
    with pytest.raises(ValueError, match="Unsupported peptide-linked residue"):
        extract_chain(structure, "F", (residue.id[1], residue.id[1]))


@pytest.mark.parametrize(
    "neighbor_altloc,selected", [("B", False), (" ", True)]
)
def test_peptide_membership_requires_compatible_altlocs(
    structure, neighbor_altloc, selected
):
    chain = structure[0]["F"]
    residue = chain.child_list[10]
    residue.id = ("H_" + residue.resname, residue.id[1], residue.id[2])
    for atom in chain.get_atoms():
        atom.set_altloc("A" if atom.parent is residue else neighbor_altloc)
    data = extract_chain(structure, "F", None)
    assert (10 in data.residue_indices) == selected


def test_hetero_only_peptide_is_still_selected(structure):
    chain = structure[0]["F"]
    before = extract_chain(structure, "F", None)
    for residue in chain:
        residue.id = ("H_" + residue.resname, residue.id[1], residue.id[2])
    after = extract_chain(structure, "F", None)
    assert after.sequence == before.sequence
    np.testing.assert_array_equal(after.coords, before.coords)
