"""Cross-scheme invariants for synthetic domains and real scFv structures.

Expensive encodings/alignments are cached by fixture and parameter mode.
Numbering and structure application still run for every scheme. The real
fixtures and their independent labels/provenance live in tests/data/antibody.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
from antibody_cases import load_case, load_cases
from Bio.PDB import PDBParser

from sabr import constants, renumber_structure
from sabr.alignment import align
from sabr.model import encode
from sabr.numbering import number_alignment
from sabr.structure import apply_numbering, extract_chain

DATA = Path(__file__).parent / "data"
CASES = {case["id"]: case for case in load_cases()}


@lru_cache(maxsize=None)
def _single_domain(chain_type, mode):
    filename, chain = {
        "H": ("test_heavy_chain.pdb", "F"),
        "K": ("12e8_L.pdb", "L"),
        "L": ("1bjm_A.pdb", "A"),
    }[chain_type]
    structure = PDBParser(QUIET=True).get_structure("domain", DATA / filename)
    data = extract_chain(structure, chain, None)
    matrix, selected, _ = align(
        encode(data.coords, mode), data.gap_indices, chain_type, 0.0, mode=mode
    )
    assert selected == chain_type
    rows = np.flatnonzero(matrix.any(axis=1))
    start, end = int(rows[0]), int(rows[-1]) + 1
    matrix = matrix[start:end].copy()
    matrix.flags.writeable = False
    return matrix, data.sequence[start:end]


def _assert_record_invariants(records, sequence):
    rows = [record[0] for record in records]
    assert rows == list(range(rows[0], rows[-1] + 1))
    assert len(
        {(number, code.strip()) for _, number, code, _ in records}
    ) == len(records)
    assert (
        "".join(record[3] for record in records)
        == sequence[rows[0] : rows[-1] + 1]
    )


@pytest.mark.parametrize("mode", constants.MODES)
@pytest.mark.parametrize("scheme", constants.NUMBERING_SCHEMES)
@pytest.mark.parametrize("order", ["HK", "HL", "KH", "LH"])
@pytest.mark.parametrize("linker_length", [0, 8])
def test_synthetic_domain_order_scheme_and_linker_invariants(
    mode, scheme, order, linker_length
):
    first, first_sequence = _single_domain(order[0], mode)
    second, second_sequence = _single_domain(order[1], mode)
    second_start = len(first) + linker_length
    sequence = first_sequence + "G" * linker_length + second_sequence
    matrix = np.zeros((len(sequence), 256), dtype=int)
    matrix[: len(first), :128] = first
    matrix[second_start:, 128:] = second
    original = matrix.copy()

    records = number_alignment(matrix, sequence, scheme, order)
    _assert_record_invariants(records, sequence)
    assert [record[0] for record in records] == list(range(len(sequence)))
    first_records = number_alignment(first, first_sequence, scheme, order[0])
    second_records = number_alignment(second, second_sequence, scheme, order[1])
    assert records[: len(first)] == first_records
    # The documented output stride is independent of the 128-column blocks.
    assert records[second_start:] == [
        (row + second_start, number + 1000, code, amino_acid)
        for row, number, code, amino_acid in second_records
    ]
    assert records[len(first) : second_start] == [
        (len(first) + index, first_records[-1][1] + index + 1, "", "G")
        for index in range(linker_length)
    ]
    assert max(record[1] for record in records[:second_start]) < min(
        record[1] for record in records[second_start:]
    )
    np.testing.assert_array_equal(matrix, original)


@lru_cache(maxsize=None)
def _real_scfv(case_id, mode):
    case = CASES[case_id]
    structure, labels = load_case(case)
    data = extract_chain(structure, case["chain"], None)
    matrix, selected, _ = align(
        encode(data.coords, mode),
        data.gap_indices,
        "auto",
        0.0,
        mode=mode,
        scfv=True,
    )
    assert selected == case["expected_types"]
    matrix.flags.writeable = False
    return structure, labels, data, matrix, selected


def _snapshot(structure):
    residues = list(structure.get_residues())
    atoms = [
        atom for residue in residues for atom in residue.get_unpacked_list()
    ]
    fields = [
        (
            atom.name,
            atom.altloc,
            atom.occupancy,
            atom.bfactor,
            atom.element,
            atom.serial_number,
        )
        for atom in atoms
    ]
    return (
        [r.id for r in residues],
        fields,
        np.array([a.coord.copy() for a in atoms]),
    )


@pytest.mark.parametrize("mode", constants.MODES)
@pytest.mark.parametrize("scheme", constants.NUMBERING_SCHEMES)
@pytest.mark.parametrize("case_id", ["6nou_A", "5kve_L"])
def test_real_scfv_all_schemes_preserve_atoms_and_domain_order(
    case_id, mode, scheme
):
    structure, labels, data, matrix, selected = _real_scfv(case_id, mode)
    original_ids, original_fields, original_coords = _snapshot(structure)
    records = number_alignment(matrix, data.sequence, scheme, selected)
    _assert_record_invariants(records, data.sequence)
    result = apply_numbering(structure, CASES[case_id]["chain"], data, records)
    ids, fields, coords = _snapshot(result)
    assert len(set(ids)) == len(ids)
    assert fields == original_fields
    np.testing.assert_array_equal(coords, original_coords)
    unchanged_ids, unchanged_fields, unchanged_coords = _snapshot(structure)
    assert unchanged_ids == original_ids
    assert unchanged_fields == original_fields
    np.testing.assert_array_equal(unchanged_coords, original_coords)
    for row, number, code, _ in records:
        assert ids[data.residue_indices[row]][1:] == (number, code or " ")

    # Independent labels locate the two domains even for non-IMGT output.
    first_rows = [label["row"] for label in labels if label["domain"] == 0]
    second_rows = [label["row"] for label in labels if label["domain"] == 1]
    assert max(first_rows) < min(second_rows)
    assert max(ids[row][1] for row in first_rows) < min(
        ids[row][1] for row in second_rows
    )
    first_matches = np.flatnonzero(matrix[:, :128].any(axis=1))
    second_matches = np.flatnonzero(matrix[:, 128:].any(axis=1))
    assert first_matches[-1] < second_matches[0]
    assert np.all(matrix.sum(axis=0) <= 1)
    assert np.all(matrix.sum(axis=1) <= 1)


@pytest.mark.parametrize("mode", constants.MODES)
@pytest.mark.parametrize("case_id", ["6nou_A", "5kve_L"])
def test_real_scfv_public_api_matches_unmocked_stages(case_id, mode):
    structure, _, data, matrix, selected = _real_scfv(case_id, mode)
    case = CASES[case_id]
    records = number_alignment(matrix, data.sequence, "wolfguy", selected)
    expected = apply_numbering(structure, case["chain"], data, records)
    result = renumber_structure(
        structure, case["chain"], scheme="wolfguy", mode=mode, **case["options"]
    )
    expected_ids, expected_fields, expected_coords = _snapshot(expected)
    ids, fields, coords = _snapshot(result)
    assert ids == expected_ids
    assert fields == expected_fields
    np.testing.assert_array_equal(coords, expected_coords)
