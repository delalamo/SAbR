"""Unit tests for the renumber module."""

import numpy as np
import pytest
from Bio import PDB

import sabr.alignment.soft_aligner as soft_aligner_module
from sabr.cli import renumber
from sabr.embeddings import mpnn as mpnn_embeddings_module
from tests.conftest import (
    FIXTURES,
    DummyEmbeddings,
    create_dummy_aligner,
    create_dummy_from_pdb,
    extract_residue_ids_from_structure,
    load_alignment_fixture,
)


class TestRenumberStructure:
    """Test suite for renumber_structure function."""

    def test_renumber_structure_basic(self, monkeypatch):
        """Core test: renumber_structure returns a valid Structure."""
        data = FIXTURES["8_21"]
        if not data["pdb"].exists():
            pytest.skip(f"Missing structure fixture at {data['pdb']}")

        alignment, chain_type = load_alignment_fixture(data["alignment"])

        DummyAligner = create_dummy_aligner(alignment, chain_type)
        dummy_from_chain = self._create_dummy_from_chain()

        monkeypatch.setattr(
            mpnn_embeddings_module, "from_chain", dummy_from_chain
        )
        monkeypatch.setattr(
            soft_aligner_module, "SoftAligner", lambda: DummyAligner()
        )

        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure("test", str(data["pdb"]))

        result = renumber.renumber_structure(
            structure,
            chain=data["chain"],
            numbering_scheme="imgt",
        )

        assert isinstance(result, PDB.Structure.Structure)
        assert data["chain"] in [ch.id for ch in result[0]]

    def test_renumber_structure_preserves_residue_count(self, monkeypatch):
        """Core test: renumber_structure preserves residue count."""
        data = FIXTURES["8_21"]
        if not data["pdb"].exists():
            pytest.skip(f"Missing structure fixture at {data['pdb']}")

        alignment, chain_type = load_alignment_fixture(data["alignment"])

        DummyAligner = create_dummy_aligner(alignment, chain_type)
        dummy_from_pdb = create_dummy_from_pdb()
        dummy_from_chain = self._create_dummy_from_chain()

        monkeypatch.setattr(mpnn_embeddings_module, "from_pdb", dummy_from_pdb)
        monkeypatch.setattr(
            mpnn_embeddings_module, "from_chain", dummy_from_chain
        )
        monkeypatch.setattr(
            soft_aligner_module, "SoftAligner", lambda: DummyAligner()
        )

        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure("test", str(data["pdb"]))

        result = renumber.renumber_structure(
            structure,
            chain=data["chain"],
            numbering_scheme="imgt",
        )

        original_ids = extract_residue_ids_from_structure(
            structure, data["chain"]
        )
        renumbered_ids = extract_residue_ids_from_structure(
            result, data["chain"]
        )

        assert len(original_ids) == len(renumbered_ids)

    def test_renumber_structure_passes_use_custom_gap_penalties(
        self, monkeypatch
    ):
        """Test use_custom_gap_penalties is passed via renumber_structure."""
        data = FIXTURES["8_21"]
        if not data["pdb"].exists():
            pytest.skip(f"Missing structure fixture at {data['pdb']}")

        alignment, chain_type = load_alignment_fixture(data["alignment"])

        # Capture kwargs passed to the aligner
        captured_kwargs = {}
        DummyAligner = create_dummy_aligner(
            alignment, chain_type, captured_kwargs
        )
        dummy_from_chain = self._create_dummy_from_chain()

        monkeypatch.setattr(
            mpnn_embeddings_module, "from_chain", dummy_from_chain
        )
        # Patch SoftAligner in the renumber module where it's imported
        monkeypatch.setattr(renumber, "SoftAligner", lambda: DummyAligner())

        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure("test", str(data["pdb"]))

        # Test with use_custom_gap_penalties=False
        captured_kwargs.clear()
        renumber.renumber_structure(
            structure,
            chain=data["chain"],
            numbering_scheme="imgt",
            use_custom_gap_penalties=False,
        )
        assert captured_kwargs.get("use_custom_gap_penalties") is False

        # Test with use_custom_gap_penalties=True (explicit)
        captured_kwargs.clear()
        renumber.renumber_structure(
            structure,
            chain=data["chain"],
            numbering_scheme="imgt",
            use_custom_gap_penalties=True,
        )
        assert captured_kwargs.get("use_custom_gap_penalties") is True

    def _create_dummy_from_chain(self):
        """Create a mock from_chain function."""

        def dummy_from_chain(chain, **kwargs):
            from sabr.constants import AA_3TO1

            sequence = ""
            for res in chain.get_residues():
                if res.get_id()[0].strip():
                    continue
                resname = res.get_resname()
                if resname in AA_3TO1:
                    sequence += AA_3TO1[resname]

            actual_n_residues = len(sequence)
            return DummyEmbeddings(
                name="test_chain",
                embeddings=np.zeros((actual_n_residues, 64)),
                idxs=[str(i) for i in range(actual_n_residues)],
                sequence=sequence,
            )

        return dummy_from_chain


class TestRunRenumberingPipeline:
    """Test suite for run_renumbering_pipeline function."""

    def test_run_renumbering_pipeline_returns_tuple(self, monkeypatch):
        """Core test: run_renumbering_pipeline returns expected tuple."""
        data = FIXTURES["8_21"]
        if not data["pdb"].exists():
            pytest.skip(f"Missing structure fixture at {data['pdb']}")

        alignment, chain_type = load_alignment_fixture(data["alignment"])

        DummyAligner = create_dummy_aligner(alignment, chain_type)
        monkeypatch.setattr(renumber, "SoftAligner", lambda: DummyAligner())

        n_residues = alignment.shape[0]
        dummy_seq = "EVQLVESGGGLVQPGGSLRLSCAASGFTFS" * 4
        dummy_seq = dummy_seq[:n_residues]

        dummy_embeddings = DummyEmbeddings(
            name="test",
            embeddings=np.zeros((n_residues, 64)),
            idxs=[str(i) for i in range(n_residues)],
            sequence=dummy_seq,
        )

        result = renumber.run_renumbering_pipeline(
            dummy_embeddings,
            numbering_scheme="imgt",
            chain_type="auto",
        )

        assert isinstance(result, tuple)
        assert len(result) == 3

        anarci_out, detected_chain_type, first_aligned_row = result
        assert isinstance(anarci_out, list)
        assert detected_chain_type in ("H", "K", "L")
        assert isinstance(first_aligned_row, int)

    def test_run_renumbering_pipeline_passes_use_custom_gap_penalties(
        self, monkeypatch
    ):
        """Test that use_custom_gap_penalties is passed through pipeline."""
        data = FIXTURES["8_21"]
        if not data["pdb"].exists():
            pytest.skip(f"Missing structure fixture at {data['pdb']}")

        alignment, chain_type = load_alignment_fixture(data["alignment"])

        # Capture kwargs passed to the aligner
        captured_kwargs = {}
        DummyAligner = create_dummy_aligner(
            alignment, chain_type, captured_kwargs
        )
        monkeypatch.setattr(renumber, "SoftAligner", lambda: DummyAligner())

        n_residues = alignment.shape[0]
        dummy_seq = "EVQLVESGGGLVQPGGSLRLSCAASGFTFS" * 4
        dummy_seq = dummy_seq[:n_residues]

        dummy_embeddings = DummyEmbeddings(
            name="test",
            embeddings=np.zeros((n_residues, 64)),
            idxs=[str(i) for i in range(n_residues)],
            sequence=dummy_seq,
        )

        # Test with use_custom_gap_penalties=False
        captured_kwargs.clear()
        renumber.run_renumbering_pipeline(
            dummy_embeddings,
            numbering_scheme="imgt",
            use_custom_gap_penalties=False,
        )
        assert captured_kwargs.get("use_custom_gap_penalties") is False

        # Test with use_custom_gap_penalties=True (explicit)
        captured_kwargs.clear()
        renumber.run_renumbering_pipeline(
            dummy_embeddings,
            numbering_scheme="imgt",
            use_custom_gap_penalties=True,
        )
        assert captured_kwargs.get("use_custom_gap_penalties") is True
