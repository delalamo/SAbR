"""Biopython structure handling."""

import copy
import functools
import json
from dataclasses import dataclass
from importlib.resources import files

import numpy as np
from Bio.PDB.Polypeptide import is_aa
from Bio.PDB.Residue import DisorderedResidue
from Bio.PDB.Structure import Structure

from sabr import constants


@dataclass(frozen=True, slots=True)
class _ChainData:
    """Private, normalized view of the selected polymer residues."""

    coords: np.ndarray
    sequence: str
    residue_ids: tuple[tuple[int, str], ...]
    residue_indices: tuple[int, ...]
    gap_indices: frozenset[int]


def _compute_cb(n_coord: np.ndarray, ca_coord: np.ndarray, c_coord: np.ndarray):
    epsilon = 1e-8

    def normalize(vector):
        length = np.sqrt(
            np.square(vector).sum(axis=-1, keepdims=True) + epsilon
        )
        return vector / length

    direction = normalize(n_coord - ca_coord)
    normal = normalize(np.cross(n_coord - c_coord, direction))
    return ca_coord + (
        constants.CB_BOND_LENGTH * np.cos(constants.CB_BOND_ANGLE) * direction
        + constants.CB_BOND_LENGTH
        * np.sin(constants.CB_BOND_ANGLE)
        * np.cos(constants.CB_DIHEDRAL)
        * np.cross(normal, direction)
        + constants.CB_BOND_LENGTH
        * np.sin(constants.CB_BOND_ANGLE)
        * np.sin(constants.CB_DIHEDRAL)
        * (-normal)
    )


def _validate_structure(structure, chain: str):
    if not isinstance(structure, Structure):
        raise TypeError(
            "structure must be a Bio.PDB.Structure.Structure object."
        )
    if not isinstance(chain, str) or not chain:
        raise ValueError("chain must be a non-empty string.")
    if len(structure) != 1:
        raise ValueError(
            "SAbR requires a structure containing exactly one model."
        )


def _find_chain(structure, chain: str):
    _validate_structure(structure, chain)
    model = next(iter(structure))
    for candidate in model:
        if candidate.id == chain:
            return candidate
    names = [candidate.id for candidate in model]
    raise ValueError(f"Chain '{chain}' not found. Available chains: {names}.")


@functools.cache
def _modified_residue_mapping() -> dict:
    path = files("sabr.assets") / "modified_residues.json"
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)["mapping"]


def _parent_residue_name(residue_name: str) -> str | None:
    if residue_name in constants.AA_3TO1:
        return residue_name
    return _modified_residue_mapping().get(residue_name)


def _altloc(value) -> str:
    return str(value or "").strip(" \x00")


def _select_backbone(atoms: list, label: str) -> tuple:
    """Select one internally consistent N/CA/C conformer."""
    names = ("N", "CA", "C")

    def select_atom(name, altloc):
        exact = [
            atom for atom in atoms if atom[0] == name and atom[1] == altloc
        ]
        shared = [atom for atom in atoms if atom[0] == name and atom[1] == ""]
        choices = exact or shared
        if not choices:
            return None
        return max(choices, key=lambda atom: atom[2])

    blank = [select_atom(name, "") for name in names]
    if all(blank):
        return tuple(atom[3] for atom in blank)

    candidates = []
    for altloc in sorted({atom[1] for atom in atoms if atom[1]}):
        selected = [select_atom(name, altloc) for name in names]
        if all(selected):
            candidates.append(
                (sum(atom[2] for atom in selected), altloc, selected)
            )
    if not candidates:
        present = {atom[0] for atom in atoms}
        missing = [name for name in names if name not in present]
        if missing:
            raise ValueError(
                f"Residue {label} is missing required backbone atom(s): "
                f"{', '.join(missing)}."
            )
        raise ValueError(
            f"Residue {label} has no complete backbone conformer because "
            "its N, CA, and C atoms use incompatible altlocs."
        )
    _, _, selected = sorted(candidates, key=lambda item: (-item[0], item[1]))[0]
    return tuple(atom[3] for atom in selected)


def _residue_data(residue) -> tuple:
    label = f"{residue.resname} {residue.id[1]}{residue.id[2].strip()}"
    atoms = [
        (
            atom.name,
            _altloc(atom.altloc),
            float(atom.occupancy or 0.0),
            np.asarray(atom.coord, dtype=np.float64),
        )
        for atom in residue.get_unpacked_list()
        if atom.name in ("N", "CA", "C")
    ]
    return _select_backbone(atoms, label)


def _detect_gaps(coords: np.ndarray) -> frozenset:
    if len(coords) < 2:
        return frozenset()
    distances = np.linalg.norm(coords[:-1, 2] - coords[1:, 0], axis=1)
    return frozenset(
        int(index)
        for index in np.where(distances > constants.PEPTIDE_BOND_MAX_DISTANCE)[
            0
        ]
    )


def extract_chain(
    structure, chain: str, residue_range: tuple | None
) -> _ChainData:
    """Normalize selected polymer residues without modifying the structure.

    Besides producing model coordinates and a one-letter sequence, extraction
    retains each residue's index in the original chain so numbering can later
    be applied to a same-type clone containing all original hetero residues.

    Unknown HETATM residues need a deliberate second pass. Most are ligands or
    solvent and should be preserved but ignored; an unknown residue with a
    complete backbone that is peptide-bonded to a recognized amino acid is
    instead unsupported polymer chemistry and must fail explicitly. Its
    neighbors are not known until the first pass has classified the chain.
    """
    target = _find_chain(structure, chain)
    coords = []
    sequence = []
    residue_ids = []
    residue_indices = []
    normalized = []
    unknown_hetero = []
    polymer_ids = set()

    # First pass: read Biopython metadata and separate recognized
    # polymer residues from unknown HETATM records that may merely be ligands.
    for index, residue in enumerate(target):
        number = residue.id[1]
        insertion_code = residue.id[2].strip()
        residue_name = residue.resname
        is_atom_record = not residue.id[0].strip()
        if residue_range is not None and not (
            residue_range[0] <= number <= residue_range[1]
        ):
            continue
        if isinstance(residue, DisorderedResidue):
            raise ValueError(
                f"Residue {number}{insertion_code} has ambiguous "
                "microheterogeneous residue names."
            )

        parent_name = _parent_residue_name(residue_name)
        if parent_name is None and is_atom_record:
            raise ValueError(
                f"Unsupported polymer residue {residue_name} "
                f"{number}{insertion_code}."
            )
        if parent_name is None and is_aa(residue):
            raise ValueError(
                f"Unsupported polymer residue {residue_name} "
                f"{number}{insertion_code}; the pinned CCD snapshot has no "
                "single canonical parent."
            )
        if parent_name is None:
            try:
                backbone = _residue_data(residue)
            except ValueError:
                continue
            unknown_hetero.append(
                (index, residue_name, number, insertion_code, backbone)
            )
            continue

        residue_key = (number, insertion_code)
        if residue_key in polymer_ids:
            raise ValueError(
                f"Residue {number}{insertion_code} has ambiguous "
                "microheterogeneous residue names."
            )
        polymer_ids.add(residue_key)
        backbone = _residue_data(residue)
        normalized.append(
            (
                index,
                number,
                insertion_code,
                parent_name,
                backbone,
            )
        )

    # A backbone-shaped HETATM is safe to ignore only when it is not connected
    # to the recognized polymer on either side. This catches unsupported
    # peptide residues without rejecting unrelated ligands or solvent.
    for index, residue_name, number, insertion_code, backbone in unknown_hetero:
        previous = next(
            (entry for entry in reversed(normalized) if entry[0] < index),
            None,
        )
        following = next(
            (entry for entry in normalized if entry[0] > index),
            None,
        )
        connected = (
            previous is not None
            and np.linalg.norm(previous[4][2] - backbone[0])
            <= constants.PEPTIDE_BOND_MAX_DISTANCE
        ) or (
            following is not None
            and np.linalg.norm(backbone[2] - following[4][0])
            <= constants.PEPTIDE_BOND_MAX_DISTANCE
        )
        if connected:
            raise ValueError(
                f"Unsupported peptide-linked residue {residue_name} "
                f"{number}{insertion_code}."
            )

    if len(normalized) > constants.MAX_SELECTED_RESIDUES:
        raise ValueError(
            f"Selected chain contains {len(normalized)} polymer residues; "
            f"the safety limit is {constants.MAX_SELECTED_RESIDUES}. Use "
            "residue_range to select one antibody domain."
        )

    # Second pass: build the dense model inputs after residue classification is
    # complete. Modified residues use their canonical parent only for sequence
    # generation; the returned numbering is applied to unchanged source atoms.
    for index, number, insertion_code, parent_name, backbone in normalized:
        if not np.isfinite(backbone).all():
            raise ValueError(
                f"Residue {number}{insertion_code} has non-finite coordinates."
            )
        n_coord, ca_coord, c_coord = backbone
        residue_coords = np.stack(
            (
                n_coord,
                ca_coord,
                c_coord,
                _compute_cb(n_coord, ca_coord, c_coord),
            )
        )
        if not np.isfinite(residue_coords).all():
            raise ValueError(
                f"Residue {number}{insertion_code} has non-finite coordinates."
            )
        coords.append(residue_coords)
        sequence.append(constants.AA_3TO1[parent_name])
        residue_ids.append((number, insertion_code))
        residue_indices.append(index)

    if not coords:
        description = "the requested range" if residue_range else "the chain"
        raise ValueError(
            "No polymer residues with complete backbone found in "
            f"{description}."
        )
    stacked_coords = np.stack(coords)
    return _ChainData(
        stacked_coords,
        "".join(sequence),
        tuple(residue_ids),
        tuple(residue_indices),
        _detect_gaps(stacked_coords),
    )


def _new_residue_ids(data: _ChainData, numbered: list) -> dict:
    if not numbered:
        raise ValueError("ANARCI returned no numbered residues.")
    query_rows = [record[0] for record in numbered]
    if query_rows != sorted(set(query_rows)):
        raise ValueError("ANARCI returned duplicate or unordered query rows.")
    if query_rows[0] < 0 or query_rows[-1] >= len(data.sequence):
        raise ValueError("ANARCI returned a query row outside the structure.")
    if query_rows != list(range(query_rows[0], query_rows[-1] + 1)):
        raise ValueError("ANARCI left an unnumbered internal query row.")

    mapping = {}
    first_row, first_number = numbered[0][:2]
    first_assigned_number = first_number - first_row
    for query_index in range(first_row):
        mapping[data.residue_indices[query_index]] = (
            first_assigned_number + query_index,
            "",
        )

    for query_index, number, insertion_code, expected in numbered:
        actual = data.sequence[query_index]
        if expected != actual:
            original = data.residue_ids[query_index]
            raise ValueError(
                f"Residue mismatch at {original[0]}{original[1]}: "
                f"ANARCI expected {expected}, structure contains {actual}."
            )
        mapping[data.residue_indices[query_index]] = (number, insertion_code)

    last_row, last_number = numbered[-1][:2]
    for query_index in range(last_row + 1, len(data.sequence)):
        last_number += 1
        mapping[data.residue_indices[query_index]] = (last_number, "")
    return mapping


def _check_collisions(structure, chain: str, mapping: dict) -> None:
    target = _find_chain(structure, chain)
    final_ids = []
    for index, residue in enumerate(target):
        if index in mapping:
            number, insertion_code = mapping[index]
            final_id = (residue.id[0], number, insertion_code or " ")
        else:
            final_id = residue.id
        final_ids.append(final_id)
    if len(final_ids) != len(set(final_ids)):
        raise ValueError(
            "Renumbering would collide with an unchanged residue ID outside "
            "the selected range. Renumber the whole chain or choose a "
            "non-overlapping range."
        )


def apply_numbering(
    structure,
    chain: str,
    data: _ChainData,
    numbered: list,
):
    """Return a copy with numbering applied to selected residues."""
    mapping = _new_residue_ids(data, numbered)
    _check_collisions(structure, chain, mapping)

    result = copy.deepcopy(structure)
    target = _find_chain(result, chain)
    residues = list(target)
    for residue in residues:
        residue.detach_parent()
    for index, (number, insertion_code) in mapping.items():
        residue = residues[index]
        residue.id = (residue.id[0], number, insertion_code or " ")
    target.child_list = residues
    target.child_dict = {}
    for residue in residues:
        residue.set_parent(target)
        target.child_dict[residue.id] = residue
    return result
