"""Regression tests for SAbR's local changes to the vendored ANARCI code.

These run even though the upstream module is omitted from coverage metrics.
Synthetic alignments exercise numbering independently of model recognition.
"""

import string

import numpy as np
import pytest

from sabr._anarci import schemes
from sabr.numbering import number_alignment

# IMGT input spans for each scheme's renumbered region, and SAbR's supported
# output lengths. These spans are not interchangeable CDR definitions.
REGIONS = [
    ("imgt", "H", "CDR1", 27, 38, 10012),
    ("imgt", "H", "CDR2", 56, 65, 10010),
    ("imgt", "H", "CDR3", 105, 117, 10013),
    ("aho", "H", "CDR1", 25, 40, 5018),
    ("aho", "H", "CDR2", 56, 75, 5020),
    ("aho", "H", "FW3", 76, 91, 5016),
    ("aho", "H", "CDR3", 105, 117, 5032),
    *[
        (scheme, "H", loop, start, end, limit)
        for scheme in ("chothia", "kabat", "martin")
        for loop, start, end, limit in [
            (
                "CDR1",
                24,
                40 if scheme == "kabat" else 38,
                5013 if scheme == "kabat" else 5011,
            ),
            ("CDR2", 55, 65, 5008),
            ("CDR3", 105, 117, 5010),
        ]
    ],
    *[
        (scheme, chain, loop, start, end, limit)
        for scheme in ("chothia", "kabat", "martin")
        for chain in ("K", "L")
        for loop, start, end, limit in [
            ("CDR1", 24, 40, 5011),
            ("CDR2", 57, 67, 5004),
            ("CDR3", 105, 117, 5009),
        ]
    ],
    ("wolfguy", "H", "CDR1", 27, 40, 49),
    ("wolfguy", "H", "CDR2", 55, 74, 49),
    ("wolfguy", "H", "CDR3", 105, 117, 51),
    *[
        ("wolfguy", chain, loop, start, end, 49)
        for chain in ("K", "L")
        for loop, start, end in [
            ("CDR1", 24, 40),
            ("CDR2", 56, 69),
            ("CDR3", 105, 117),
        ]
    ],
]


def _alignment(start, end, length, *, orphan=False):
    """Make a full domain with a specified region length and central inserts."""
    columns = list(range(128))
    centre = (start + end) // 2
    extra = length - (end - start + 1)
    assert extra >= 0
    columns[centre:centre] = [-1 if orphan else centre - 1] * extra
    matrix = np.zeros((len(columns), 128), dtype=bool)
    for row, column in enumerate(columns):
        if column >= 0:
            matrix[row, column] = True
    residues = "ACDEFGHIKLMNPQRSTVWY"
    sequence = "".join(
        residues[row % len(residues)] for row in range(len(columns))
    )
    return matrix, sequence


def _assert_complete(records, sequence):
    assert [row for row, *_ in records] == list(range(len(sequence)))
    assert "".join(aa for *_, aa in records) == sequence
    identifiers = [(number, code) for _, number, code, _ in records]
    assert len(set(identifiers)) == len(sequence)


@pytest.mark.parametrize(
    "scheme,chain,loop,start,end,limit",
    REGIONS,
    ids=["-".join(region[:3]) for region in REGIONS],
)
@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_scheme_region_length_boundaries(
    scheme, chain, loop, start, end, limit, offset
):
    length = limit + offset
    alignment, sequence = _alignment(start, end, length)
    if offset > 0:
        label = {"imgt": "IMGT", "aho": "AHo"}.get(scheme, scheme.title())
        with pytest.raises(ValueError) as error:
            number_alignment(alignment, sequence, scheme, chain)
        assert str(error.value) == (
            f"{label} {loop} length {length} exceeds the supported limit "
            f"of {limit} residues."
        )
    else:
        records = number_alignment(alignment, sequence, scheme, chain)
        _assert_complete(records, sequence)


@pytest.mark.parametrize(
    "scheme", ["imgt", "kabat", "chothia", "martin", "aho"]
)
@pytest.mark.parametrize("chain", ["H", "K", "L"])
@pytest.mark.parametrize("length", [35, 36, 37, 58, 59, 117, 118])
@pytest.mark.parametrize("orphan", [False, True])
def test_cdr3_crosses_inherited_limits(scheme, chain, length, orphan):
    alignment, sequence = _alignment(105, 117, length, orphan=orphan)
    _assert_complete(
        number_alignment(alignment, sequence, scheme, chain), sequence
    )


@pytest.mark.parametrize("scheme", ["imgt", "aho"])
@pytest.mark.parametrize("chain", ["A", "B", "G", "D"])
def test_extended_cdr3_supports_tcr_chains(scheme, chain):
    alignment, sequence = _alignment(105, 117, 118)
    _assert_complete(
        number_alignment(alignment, sequence, scheme, chain), sequence
    )


@pytest.mark.parametrize("count", [0, 1, 25, 26, 27, 702, 703, 5000])
def test_extended_alphabet_has_exact_capacity_and_one_blank(count):
    alphabet = schemes._generate_extended_alphabet(count)
    assert len(alphabet) == count + 1
    assert len(set(alphabet)) == count + 1
    assert alphabet[-1] == " "
    assert all(code.isalpha() and code.isupper() for code in alphabet[:-1])
    assert alphabet[: min(count, 26)] == list(string.ascii_uppercase[:count])
    if count >= 27:
        assert alphabet[25:27] == ["Z", "AA"]
    if count >= 703:
        assert alphabet[701:703] == ["ZZ", "AAA"]


def test_extended_alphabet_rejects_negative_capacity():
    with pytest.raises(ValueError, match="non-negative"):
        schemes._generate_extended_alphabet(-1)


@pytest.mark.parametrize("scheme", ["kabat", "chothia", "martin"])
@pytest.mark.parametrize(
    "chain,base,anchor", [("heavy", 10, 100), ("light", 9, 95)]
)
@pytest.mark.parametrize("insertions", [26, 27, 702, 703, 5000])
def test_linear_cdr3_insertion_order(scheme, chain, base, anchor, insertions):
    annotations = schemes.get_cdr3_annotations(base + insertions, scheme, chain)
    codes = [code for number, code in annotations if number == anchor]
    assert codes == [" ", *schemes.alphabet[:insertions]]
    assert annotations[-2:] == [(anchor + 1, " "), (anchor + 2, " ")]


@pytest.mark.parametrize(
    "base,start,end,left,right",
    [(12, 27, 39, 32, 33), (10, 56, 66, 60, 61), (13, 105, 118, 111, 112)],
)
@pytest.mark.parametrize("insertions", [52, 53, 54, 104, 105, 1405, 10000])
def test_imgt_insertions_keep_symmetric_sequence_order(
    base, start, end, left, right, insertions
):
    annotations = schemes.get_imgt_cdr(base + insertions, base, start, end)
    assert [code for number, code in annotations if number == left] == [
        " ",
        *schemes.alphabet[: insertions // 2],
    ]
    assert [code for number, code in annotations if number == right] == [
        *reversed(schemes.alphabet[: (insertions + 1) // 2]),
        " ",
    ]
    assert len(set(annotations)) == base + insertions


@pytest.mark.parametrize("length", [0, 1, 5, 13, 14, 65, 118, 10013])
def test_legacy_imgt_helper_uses_the_same_numbering(length):
    assert schemes.get_cdr3_annotations(length) == schemes.get_imgt_cdr(
        length, 13, 105, 118
    )


def test_legacy_imgt_helper_reports_limit():
    with pytest.raises(ValueError, match="IMGT CDR3 length 10014.*10013"):
        schemes.get_cdr3_annotations(10014)


@pytest.mark.parametrize("insertions", [5000, 5001])
def test_framework_insertions_do_not_consume_the_blank_sentinel(insertions):
    alignment, sequence = _alignment(43, 43, 1 + insertions)
    if insertions == 5001:
        with pytest.raises(
            ValueError,
            match="IMGT insertions at IMGT position 43 length 5001.*5000",
        ):
            number_alignment(alignment, sequence, "imgt", "H")
    else:
        records = number_alignment(alignment, sequence, "imgt", "H")
        _assert_complete(records, sequence)
        assert [code for _, number, code, _ in records if number == 43] == [
            " ",
            *schemes.alphabet[:-1],
        ]
