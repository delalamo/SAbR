"""Unit tests for the public renumber API."""

import numpy as np
import pytest
from Bio import PDB

import sabr.renumber as renumber
from sabr.options import RenumberOptions
from sabr.types import ChainType
from tests.conftest import (
    FIXTURES,
    DummyEmbeddings,
    create_dummy_aligner,
    extract_residue_ids_from_structure,
    load_alignment_fixture,
)


def _fake_numbering_backend(_states, subsequence, _scheme, _chain_type):
    sequence = subsequence.replace("-", "")
    return [((idx + 1, " "), aa) for idx, aa in enumerate(sequence)]


def _dummy_from_chain(chain, **_kwargs):
    from sabr.constants import AA_3TO1

    sequence = ""
    for residue in chain.get_residues():
        if residue.get_id()[0].strip():
            continue
        one_letter = AA_3TO1.get(residue.get_resname())
        if one_letter:
            sequence += one_letter

    return DummyEmbeddings(
        name="test_chain",
        embeddings=np.zeros((len(sequence), 64)),
        idxs=[str(i) for i in range(len(sequence))],
        sequence=sequence,
    )


class TestRenumberStructure:
    """Test suite for in-memory renumbering."""

    def test_renumber_structure_basic(self, monkeypatch):
        data = FIXTURES["8_21"]
        if not data["pdb"].exists():
            pytest.skip(f"Missing structure fixture at {data['pdb']}")

        alignment, chain_type = load_alignment_fixture(data["alignment"])
        DummyAligner = create_dummy_aligner(alignment, chain_type)

        monkeypatch.setattr(renumber, "from_chain", _dummy_from_chain)
        monkeypatch.setattr(renumber, "SoftAligner", lambda **_kwargs: DummyAligner())

        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure("test", str(data["pdb"]))
        renumberer = renumber.Renumberer(numbering_backend=_fake_numbering_backend)

        result = renumberer.renumber_structure(
            structure,
            chain_id=data["chain"],
            options=RenumberOptions(),
        )

        assert isinstance(result, PDB.Structure.Structure)
        assert data["chain"] in [chain.id for chain in result[0]]

    def test_renumber_structure_preserves_residue_count(self, monkeypatch):
        data = FIXTURES["8_21"]
        if not data["pdb"].exists():
            pytest.skip(f"Missing structure fixture at {data['pdb']}")

        alignment, chain_type = load_alignment_fixture(data["alignment"])
        DummyAligner = create_dummy_aligner(alignment, chain_type)

        monkeypatch.setattr(renumber, "from_chain", _dummy_from_chain)
        monkeypatch.setattr(renumber, "SoftAligner", lambda **_kwargs: DummyAligner())

        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure("test", str(data["pdb"]))
        renumberer = renumber.Renumberer(numbering_backend=_fake_numbering_backend)

        result = renumberer.renumber_structure(
            structure,
            chain_id=data["chain"],
            options=RenumberOptions(),
        )

        original_ids = extract_residue_ids_from_structure(structure, data["chain"])
        renumbered_ids = extract_residue_ids_from_structure(result, data["chain"])

        assert len(original_ids) == len(renumbered_ids)

    def test_options_control_custom_gap_penalties(self, monkeypatch):
        data = FIXTURES["8_21"]
        if not data["pdb"].exists():
            pytest.skip(f"Missing structure fixture at {data['pdb']}")

        alignment, chain_type = load_alignment_fixture(data["alignment"])
        captured_kwargs = {}
        DummyAligner = create_dummy_aligner(alignment, chain_type, captured_kwargs)

        monkeypatch.setattr(renumber, "from_chain", _dummy_from_chain)
        monkeypatch.setattr(renumber, "SoftAligner", lambda **_kwargs: DummyAligner())

        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure("test", str(data["pdb"]))
        renumberer = renumber.Renumberer(numbering_backend=_fake_numbering_backend)

        captured_kwargs.clear()
        renumberer.renumber_structure(
            structure,
            chain_id=data["chain"],
            options=RenumberOptions(custom_gap_penalties=False),
        )
        assert captured_kwargs["use_custom_gap_penalties"] is False

    def test_options_control_chain_type_embedding_label(self, monkeypatch):
        data = FIXTURES["8_21"]
        if not data["pdb"].exists():
            pytest.skip(f"Missing structure fixture at {data['pdb']}")

        alignment, chain_type = load_alignment_fixture(data["alignment"])
        captured_kwargs = {}
        DummyAligner = create_dummy_aligner(alignment, chain_type, captured_kwargs)

        monkeypatch.setattr(renumber, "from_chain", _dummy_from_chain)
        monkeypatch.setattr(renumber, "SoftAligner", lambda **_kwargs: DummyAligner())

        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure("test", str(data["pdb"]))
        renumberer = renumber.Renumberer(numbering_backend=_fake_numbering_backend)

        renumberer.renumber_structure(
            structure,
            chain_id=data["chain"],
            options=RenumberOptions(chain_type=ChainType.HEAVY),
        )

        assert captured_kwargs["chain_type"] is ChainType.HEAVY


class TestNumberingPlan:
    """Test suite for the alignment-to-numbering boundary."""

    def test_create_numbering_plan_returns_dataclass(self, monkeypatch):
        data = FIXTURES["8_21"]
        if not data["alignment"].exists():
            pytest.skip(f"Missing alignment fixture at {data['alignment']}")

        alignment, chain_type = load_alignment_fixture(data["alignment"])
        DummyAligner = create_dummy_aligner(alignment, chain_type)
        monkeypatch.setattr(renumber, "SoftAligner", lambda **_kwargs: DummyAligner())

        dummy_embeddings = DummyEmbeddings(
            name="test",
            embeddings=np.zeros((alignment.shape[0], 64)),
            idxs=[str(i) for i in range(alignment.shape[0])],
            sequence="A" * alignment.shape[0],
        )
        renumberer = renumber.Renumberer(numbering_backend=_fake_numbering_backend)

        plan = renumberer.create_numbering_plan(
            dummy_embeddings,
            RenumberOptions(),
        )

        assert plan.detected_chain_type.value in ("H", "K", "L")
        assert plan.selected_reference in ("H", "K", "L")
        assert isinstance(plan.first_aligned_row, int)

    def test_numbering_plan_passes_embedding_label_to_anarci(self, monkeypatch):
        alignment = np.eye(3, 128, dtype=int)
        DummyAligner = create_dummy_aligner(alignment, "K")
        monkeypatch.setattr(renumber, "SoftAligner", lambda **_kwargs: DummyAligner())

        captured = {}

        def numbering_backend(states, subsequence, scheme, chain_type):
            captured["chain_type"] = chain_type
            return _fake_numbering_backend(states, subsequence, scheme, chain_type)

        dummy_embeddings = DummyEmbeddings(
            name="test",
            embeddings=np.zeros((alignment.shape[0], 64)),
            idxs=[str(i) for i in range(alignment.shape[0])],
            sequence="AAA",
        )
        renumberer = renumber.Renumberer(numbering_backend=numbering_backend)

        plan = renumberer.create_numbering_plan(
            dummy_embeddings,
            RenumberOptions(),
        )

        assert plan.detected_chain_type is ChainType.KAPPA
        assert captured["chain_type"] is ChainType.KAPPA
