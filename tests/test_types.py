import numpy as np
import pytest

from sabr.alignment.soft_aligner import AlignmentResult
from sabr.embeddings.mpnn import QueryEmbeddings
from sabr.options import RenumberOptions
from sabr.structure.residues import ResidueId, ResidueRange
from sabr.types import ChainType, NumberingScheme, parse_chain_type


def test_mpnnembeddings_shape_mismatch_raises():
    # embeddings has 2 rows, idxs has 3 items -> mismatch
    embedding = np.zeros((2, 5), dtype=float)
    idx = ["a", "b", "c"]

    with pytest.raises(ValueError) as excinfo:
        QueryEmbeddings(name="test_case", embeddings=embedding, idxs=idx)

    # Check key parts of the error message
    msg = str(excinfo.value)
    assert "embeddings.shape[0] (2) must match len(idxs) (3)" in msg
    assert "Error raised for test_case" in msg


def test_alignment_result_holds_passed_values():
    alignment = np.ones((2, 2), dtype=int)
    output = AlignmentResult(
        alignment=alignment,
        score=1.5,
        sim_matrix=None,
        selected_chain_type=ChainType.HEAVY,
    )

    assert output.alignment.shape == (2, 2)
    assert output.score == pytest.approx(1.5)
    assert output.selected_chain_type is ChainType.HEAVY


def test_residue_id_parses_insertion_codes():
    assert ResidueId.parse("100A") == ResidueId(100, "A")
    assert ResidueId.parse("-3B") == ResidueId(-3, "B")


def test_residue_id_rejects_biopython_tuple_publicly():
    with pytest.raises(ValueError, match="Invalid residue id"):
        ResidueId.parse((" ", 10, "B"))


def test_residue_range_includes_insertion_codes_by_number():
    residue_range = ResidueRange(10, 20)

    assert residue_range.contains("10")
    assert residue_range.contains("10A")
    assert residue_range.contains("20B")
    assert not residue_range.contains("21")


def test_renumber_options_parse_enums_and_auto():
    options = RenumberOptions.from_values(
        numbering_scheme="imgt",
        chain_type="H",
        residue_range=ResidueRange(1, 128),
    )

    assert options.numbering_scheme is NumberingScheme.IMGT
    assert options.chain_type is ChainType.HEAVY
    assert options.residue_range == ResidueRange(1, 128)


def test_renumber_options_auto_chain_type_is_none():
    options = RenumberOptions.from_values(chain_type="auto")

    assert options.chain_type is None


def test_parse_chain_type_rejects_aliases_and_suffixes():
    for value in ["heavy", "kappa", "lambda", "mouse_H"]:
        with pytest.raises(ValueError, match="H, K, L, or auto"):
            parse_chain_type(value)


def test_negative_residue_ranges_are_valid():
    assert ResidueRange(-5, 10).contains("-1A")


def test_tuple_residue_range_is_not_normalized():
    with pytest.raises(ValueError, match="ResidueRange"):
        RenumberOptions.from_values(residue_range=(1, 128))
