"""Cross-scheme invariants for synthetic domains built from test fixtures."""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
from Bio.PDB import PDBParser

from sabr import constants
from sabr.alignment import align
from sabr.model import encode
from sabr.numbering import number_alignment
from sabr.structure import extract_chain

DATA = Path(__file__).parent / "data"


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
