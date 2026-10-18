"""Atomic displacement data must survive numbering and format conversion."""

import copy
from pathlib import Path

import numpy as np
import pytest
from Bio.PDB import MMCIFIO, PDBIO, PDBParser
from Bio.PDB.Atom import DisorderedAtom
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from click.testing import CliRunner

from sabr.cli import _read_structure, _write_structure, main

DATA = Path(__file__).parent / "data"
COMPONENTS = ("U[1][1]", "U[2][2]", "U[3][3]", "U[1][2]", "U[1][3]", "U[2][3]")


def _atoms(structure):
    return [
        atom
        for model in structure
        for chain in model
        for residue in chain.get_unpacked_list()
        for atom in residue.get_unpacked_list()
    ]


def _structure():
    structure = PDBParser(QUIET=True).get_structure(
        "anisotropic", DATA / "test_heavy_chain.pdb"
    )
    other = structure[0]["F"].copy()
    other.id = "Z"
    structure[0].add(other)
    residue = next(iter(structure[0]["F"]))
    original = residue["CB"]
    residue.detach_child("CB")
    disordered = DisorderedAtom("CB")
    for altloc, occupancy in (("A", 0.6), ("B", 0.4)):
        atom = original.copy()
        atom.altloc = altloc
        atom.occupancy = occupancy
        disordered.disordered_add(atom)
    residue.add(disordered)
    for residue in structure[0]["F"]:
        residue.id = (residue.id[0], residue.id[1] + 500, residue.id[2])
    atoms = _atoms(structure)
    # Sparse tensors, including both conformers and a non-target chain.
    selected = [atoms[1], *disordered.disordered_get_list(), atoms[-2]]
    for index, atom in enumerate(selected, start=1):
        atom.set_anisou(
            np.array([0.11, 0.22, 0.33, 0.012, -0.013, 0.023], dtype="f")
            * index
        )
    return structure


def _write_independent_input(structure, path):
    """Build fixtures with upstream writers and explicit standard records."""
    writer = PDBIO() if path.suffix == ".pdb" else MMCIFIO()
    writer.set_structure(structure)
    writer.save(str(path))
    atoms = _atoms(structure)
    if path.suffix == ".pdb":
        output = []
        remaining = iter(atoms)
        for line in path.read_text().splitlines(keepends=True):
            output.append(line)
            if line.startswith(("ATOM  ", "HETATM")):
                tensor = next(remaining).get_anisou()
                if tensor is not None:
                    fields = "".join(
                        f"{round(float(v) * 10000):7d}" for v in tensor
                    )
                    output.append("ANISOU" + line[6:28] + fields + line[70:])
        path.write_text("".join(output))
    else:
        data = MMCIF2Dict(str(path))
        rows = [
            i for i, atom in enumerate(atoms) if atom.get_anisou() is not None
        ]
        rows.reverse()  # Association must use atom IDs, not row positions.
        data["_atom_site_anisotrop.id"] = [
            data["_atom_site.id"][i] for i in rows
        ]
        for component, name in enumerate(COMPONENTS):
            data["_atom_site_anisotrop." + name] = [
                str(float(atoms[i].get_anisou()[component])) for i in rows
            ]
        writer.set_dict(data)
        writer.save(str(path))


def _assert_atoms_preserved(before, after, tolerance):
    left, right = _atoms(before), _atoms(after)
    assert len(left) == len(right)
    for original, result in zip(left, right, strict=True):
        assert (
            original.name,
            original.altloc,
            original.occupancy,
            original.bfactor,
        ) == (result.name, result.altloc, result.occupancy, result.bfactor)
        np.testing.assert_allclose(
            original.coord, result.coord, rtol=0, atol=0.00051
        )
        if original.get_anisou() is None:
            assert result.get_anisou() is None
        else:
            assert result.get_anisou() is not None
            np.testing.assert_allclose(
                original.get_anisou(),
                result.get_anisou(),
                rtol=0,
                atol=tolerance,
            )


@pytest.mark.parametrize("suffix", (".pdb", ".cif"))
def test_writer_preserves_sparse_tensors_and_altlocs(tmp_path, suffix):
    structure = _structure()
    original = copy.deepcopy(structure)
    output = tmp_path / ("output" + suffix)
    _write_structure(structure, output)
    _assert_atoms_preserved(original, _read_structure(output), 0.000051)
    _assert_atoms_preserved(original, structure, 0)


def test_mmcif_reader_maps_reordered_tensor_rows_by_atom_id(tmp_path):
    structure = _structure()
    path = tmp_path / "sparse.cif"
    _write_independent_input(structure, path)
    _assert_atoms_preserved(structure, _read_structure(path), 1e-7)


@pytest.mark.parametrize("input_suffix", (".pdb", ".cif"))
@pytest.mark.parametrize("output_suffix", (".pdb", ".cif"))
def test_real_cli_preserves_tensors_across_formats(
    tmp_path, input_suffix, output_suffix
):
    original = _structure()
    source = tmp_path / ("source" + input_suffix)
    output = tmp_path / ("numbered" + output_suffix)
    _write_independent_input(original, source)
    result = CliRunner().invoke(
        main, ["-i", str(source), "-c", "F", "-t", "H", "-o", str(output)]
    )
    assert result.exit_code == 0, result.output
    numbered = _read_structure(output)
    _assert_atoms_preserved(original, numbered, 0.000051)
    assert [r.id for r in numbered[0]["F"]] != [r.id for r in original[0]["F"]]
    assert [r.id for r in numbered[0]["Z"]] == [r.id for r in original[0]["Z"]]


@pytest.mark.parametrize(
    "malformation",
    ("missing", "duplicate", "unknown", "nonfinite", "ambiguous"),
)
def test_malformed_mmcif_tensors_fail_explicitly(tmp_path, malformation):
    path = tmp_path / "invalid.cif"
    _write_independent_input(_structure(), path)
    data = MMCIF2Dict(str(path))
    ids = data["_atom_site_anisotrop.id"]
    if malformation == "missing":
        del data["_atom_site_anisotrop.U[1][1]"]
    elif malformation == "duplicate":
        ids[1] = ids[0]
    elif malformation == "unknown":
        ids[0] = "absent"
    elif malformation == "ambiguous":
        data["_atom_site.id"][1] = data["_atom_site.id"][0]
    else:
        data["_atom_site_anisotrop.U[1][1]"][0] = "nan"
    writer = MMCIFIO()
    writer.set_dict(data)
    writer.save(str(path))
    with pytest.raises(ValueError):
        _read_structure(path)


def test_mmcif_tensors_support_nonnumeric_atom_ids_and_extended_insertions(
    tmp_path,
):
    structure = _structure()
    residue = next(iter(structure[0]["F"]))
    residue.id = (residue.id[0], residue.id[1], "AA")
    path = tmp_path / "extended.cif"
    _write_independent_input(structure, path)
    data = MMCIF2Dict(str(path))
    old_id = data["_atom_site_anisotrop.id"][0]
    data["_atom_site_anisotrop.id"][0] = "custom_atom"
    data["_atom_site.id"][data["_atom_site.id"].index(old_id)] = "custom_atom"
    writer = MMCIFIO()
    writer.set_dict(data)
    writer.save(str(path))
    parsed = _read_structure(path)
    assert next(iter(parsed[0]["F"])).id[2] == "AA"
    _assert_atoms_preserved(structure, parsed, 1e-7)
    _write_structure(parsed, path)
    _assert_atoms_preserved(structure, _read_structure(path), 1e-7)


def test_unrepresentable_pdb_tensor_fails_without_publishing_output(tmp_path):
    structure = _structure()
    _atoms(structure)[1].set_anisou(np.array([1000, 1, 1, 0, 0, 0], dtype="f"))
    source, output = tmp_path / "large.cif", tmp_path / "output.pdb"
    _write_independent_input(structure, source)
    result = CliRunner().invoke(
        main, ["-i", str(source), "-c", "F", "-t", "H", "-o", str(output)]
    )
    assert result.exit_code != 0
    assert "use mmCIF output" in result.output
    assert list(tmp_path.iterdir()) == [source]
