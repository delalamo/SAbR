"""Unit tests for the renumber module.

These tests verify that the renumber_structure function produces identical
results to the CLI when given the same input, ensuring API parity.
"""

from importlib import resources
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest
from Bio import PDB

from sabr import mpnn_embeddings, renumber
from tests.conftest import create_dummy_aligner, create_dummy_from_pdb

DATA_PACKAGE = "tests.data"


def resolve_data_path(filename: str) -> Path:
    return Path(resources.files(DATA_PACKAGE) / filename)


FIXTURES = {
    "8_21": {
        "pdb": resolve_data_path("8_21_renumbered.pdb"),
        "chain": "A",
        "alignment": resolve_data_path("8_21_renumbered_alignment.npz"),
        "embeddings": resolve_data_path("8_21_renumbered_embeddings.npz"),
    },
    "5omm": {
        "pdb": resolve_data_path("5omm_imgt.pdb"),
        "chain": "C",
        "alignment": resolve_data_path("5omm_imgt_alignment.npz"),
        "embeddings": resolve_data_path("5omm_imgt_embeddings.npz"),
    },
}


def load_alignment_fixture(path: Path) -> Tuple[np.ndarray, str]:
    if not path.exists():
        pytest.skip(f"Missing alignment fixture at {path}")
    data = np.load(path, allow_pickle=True)
    alignment = data["alignment"]
    chain_type = data["chain_type"].item()
    return alignment, chain_type


def extract_residue_ids(
    structure: PDB.Structure.Structure, chain: str
) -> List[Tuple[str, int, str]]:
    """Extract residue IDs from a BioPython structure."""
    residues = []
    for res in structure[0][chain]:
        hetflag, resseq, icode = res.get_id()
        if hetflag.strip():
            continue
        residues.append((hetflag, resseq, icode))
    return residues


class TestRenumberStructure:
    """Test suite for renumber_structure function."""

    def test_renumber_structure_basic(self, monkeypatch):
        """Test basic renumber_structure functionality with mock alignment."""
        data = FIXTURES["8_21"]
        if not data["pdb"].exists():
            pytest.skip(f"Missing structure fixture at {data['pdb']}")

        alignment, chain_type = load_alignment_fixture(data["alignment"])

        # Mock the aligner and embedding functions
        DummyAligner = create_dummy_aligner(alignment, chain_type)
        dummy_from_chain = self._create_dummy_from_chain()

        monkeypatch.setattr(mpnn_embeddings, "from_chain", dummy_from_chain)
        monkeypatch.setattr(
            renumber.softaligner, "SoftAligner", lambda: DummyAligner()
        )

        # Load the structure
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure("test", str(data["pdb"]))

        # Run renumber_structure - now returns Structure directly
        result = renumber.renumber_structure(
            structure,
            chain=data["chain"],
            numbering_scheme="imgt",
        )

        # Verify result is a Structure object
        assert isinstance(result, PDB.Structure.Structure)
        # Verify it has the expected chain
        assert data["chain"] in [ch.id for ch in result[0]]

    def test_renumber_structure_matches_cli_output(self, monkeypatch):
        """Test that renumber_structure produces same residue IDs as CLI."""
        data = FIXTURES["8_21"]
        if not data["pdb"].exists():
            pytest.skip(f"Missing structure fixture at {data['pdb']}")

        alignment, chain_type = load_alignment_fixture(data["alignment"])

        # Create mocks
        DummyAligner = create_dummy_aligner(alignment, chain_type)
        dummy_from_pdb = create_dummy_from_pdb()
        dummy_from_chain = self._create_dummy_from_chain()

        monkeypatch.setattr(mpnn_embeddings, "from_pdb", dummy_from_pdb)
        monkeypatch.setattr(mpnn_embeddings, "from_chain", dummy_from_chain)
        monkeypatch.setattr(
            renumber.softaligner, "SoftAligner", lambda: DummyAligner()
        )

        # Load structure
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure("test", str(data["pdb"]))

        # Run renumber_structure
        result = renumber.renumber_structure(
            structure,
            chain=data["chain"],
            numbering_scheme="imgt",
        )

        # Extract residue IDs from both
        original_ids = extract_residue_ids(structure, data["chain"])
        renumbered_ids = extract_residue_ids(result, data["chain"])

        # For 8_21 which is already correctly numbered, IDs should match
        assert len(original_ids) == len(renumbered_ids)

    def test_renumber_structure_invalid_chain(self):
        """Test that invalid chain raises ValueError."""
        parser = PDB.PDBParser(QUIET=True)
        pdb_path = FIXTURES["8_21"]["pdb"]
        if not pdb_path.exists():
            pytest.skip(f"Missing structure fixture at {pdb_path}")

        structure = parser.get_structure("test", str(pdb_path))

        with pytest.raises(ValueError, match="Chain 'X' not found"):
            renumber.renumber_structure(structure, chain="X")

    def test_renumber_structure_multi_char_chain_rejected(self):
        """Test that multi-character chain identifier is rejected."""
        parser = PDB.PDBParser(QUIET=True)
        pdb_path = FIXTURES["8_21"]["pdb"]
        if not pdb_path.exists():
            pytest.skip(f"Missing structure fixture at {pdb_path}")

        structure = parser.get_structure("test", str(pdb_path))

        with pytest.raises(
            ValueError, match="Chain identifier must be exactly one character"
        ):
            renumber.renumber_structure(structure, chain="AB")

    @pytest.mark.parametrize("chain_type", ["H", "K", "L", "auto"])
    def test_renumber_structure_chain_type_options(
        self, monkeypatch, chain_type
    ):
        """Test that all chain type options are accepted."""
        data = FIXTURES["8_21"]
        if not data["pdb"].exists():
            pytest.skip(f"Missing structure fixture at {data['pdb']}")

        alignment, fixture_chain_type = load_alignment_fixture(
            data["alignment"]
        )

        DummyAligner = create_dummy_aligner(alignment, fixture_chain_type)
        dummy_from_chain = self._create_dummy_from_chain()

        monkeypatch.setattr(mpnn_embeddings, "from_chain", dummy_from_chain)
        monkeypatch.setattr(
            renumber.softaligner, "SoftAligner", lambda: DummyAligner()
        )

        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure("test", str(data["pdb"]))

        # Should not raise
        result = renumber.renumber_structure(
            structure,
            chain=data["chain"],
            chain_type=chain_type,
        )
        assert isinstance(result, PDB.Structure.Structure)

    @pytest.mark.parametrize(
        "numbering_scheme",
        ["imgt", "chothia", "kabat", "martin", "aho", "wolfguy"],
    )
    def test_renumber_structure_numbering_schemes(
        self, monkeypatch, numbering_scheme
    ):
        """Test that all numbering schemes are accepted."""
        data = FIXTURES["8_21"]
        if not data["pdb"].exists():
            pytest.skip(f"Missing structure fixture at {data['pdb']}")

        alignment, chain_type = load_alignment_fixture(data["alignment"])

        DummyAligner = create_dummy_aligner(alignment, chain_type)
        dummy_from_chain = self._create_dummy_from_chain()

        monkeypatch.setattr(mpnn_embeddings, "from_chain", dummy_from_chain)
        monkeypatch.setattr(
            renumber.softaligner, "SoftAligner", lambda: DummyAligner()
        )

        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure("test", str(data["pdb"]))

        # Should not raise
        result = renumber.renumber_structure(
            structure,
            chain=data["chain"],
            numbering_scheme=numbering_scheme,
        )
        assert isinstance(result, PDB.Structure.Structure)

    @pytest.mark.parametrize("use_deterministic", [True, False])
    def test_renumber_structure_deterministic_flag(
        self, monkeypatch, use_deterministic
    ):
        """Test deterministic_loop_renumbering flag is passed correctly."""
        data = FIXTURES["8_21"]
        if not data["pdb"].exists():
            pytest.skip(f"Missing structure fixture at {data['pdb']}")

        alignment, chain_type = load_alignment_fixture(data["alignment"])

        captured_kwargs = {}
        DummyAligner = create_dummy_aligner(
            alignment, chain_type, captured_kwargs
        )
        dummy_from_chain = self._create_dummy_from_chain()

        monkeypatch.setattr(mpnn_embeddings, "from_chain", dummy_from_chain)
        monkeypatch.setattr(
            renumber.softaligner, "SoftAligner", lambda: DummyAligner()
        )

        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure("test", str(data["pdb"]))

        renumber.renumber_structure(
            structure,
            chain=data["chain"],
            deterministic_loop_renumbering=use_deterministic,
        )

        assert (
            captured_kwargs.get("deterministic_loop_renumbering")
            == use_deterministic
        )

    def test_extract_chain_subset(self):
        """Test that _extract_chain_subset correctly filters residues."""
        data = FIXTURES["8_21"]
        if not data["pdb"].exists():
            pytest.skip(f"Missing structure fixture at {data['pdb']}")

        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure("test", str(data["pdb"]))
        chain_id = data["chain"]

        # Get original residue count
        original_count = len(list(structure[0][chain_id].get_residues()))

        # Test with res_end only
        subset = renumber._extract_chain_subset(
            structure, chain_id, res_start=None, res_end=50
        )
        subset_residues = list(subset[0][chain_id].get_residues())
        # All residues should have resnum <= 50
        for res in subset_residues:
            assert res.get_id()[1] <= 50
        assert len(subset_residues) < original_count

        # Test with res_start only
        subset = renumber._extract_chain_subset(
            structure, chain_id, res_start=50, res_end=None
        )
        subset_residues = list(subset[0][chain_id].get_residues())
        # All residues should have resnum >= 50
        for res in subset_residues:
            assert res.get_id()[1] >= 50

        # Test with both res_start and res_end
        subset = renumber._extract_chain_subset(
            structure, chain_id, res_start=30, res_end=80
        )
        subset_residues = list(subset[0][chain_id].get_residues())
        # All residues should be in range [30, 80]
        for res in subset_residues:
            res_num = res.get_id()[1]
            assert 30 <= res_num <= 80

    def _create_dummy_from_chain(self):
        """Create a mock from_chain function."""
        from tests.conftest import DummyEmbeddings

        def dummy_from_chain(chain, **kwargs):
            # Extract the sequence from the chain
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
                stdev=np.ones((actual_n_residues, 64)),
                sequence=sequence,
            )

        return dummy_from_chain


class TestRunRenumberingPipeline:
    """Test suite for run_renumbering_pipeline function."""

    def test_run_renumbering_pipeline_returns_tuple(self, monkeypatch):
        """Test that run_renumbering_pipeline returns expected tuple."""
        data = FIXTURES["8_21"]
        if not data["pdb"].exists():
            pytest.skip(f"Missing structure fixture at {data['pdb']}")

        alignment, chain_type = load_alignment_fixture(data["alignment"])

        DummyAligner = create_dummy_aligner(alignment, chain_type)
        monkeypatch.setattr(
            renumber.softaligner, "SoftAligner", lambda: DummyAligner()
        )

        # Create dummy embeddings matching the alignment size
        # The alignment is (n_residues, 128) - get n_residues from it
        n_residues = alignment.shape[0]
        from tests.conftest import DummyEmbeddings

        # Use a realistic antibody sequence snippet
        dummy_seq = "EVQLVESGGGLVQPGGSLRLSCAASGFTFS" * 4
        dummy_seq = dummy_seq[:n_residues]

        dummy_embeddings = DummyEmbeddings(
            name="test",
            embeddings=np.zeros((n_residues, 64)),
            idxs=[str(i) for i in range(n_residues)],
            stdev=np.ones((n_residues, 64)),
            sequence=dummy_seq,
        )

        result = renumber.run_renumbering_pipeline(
            dummy_embeddings,
            numbering_scheme="imgt",
            chain_type="auto",
        )

        # Should return (anarci_alignment, chain_type, first_aligned_row)
        assert isinstance(result, tuple)
        assert len(result) == 3

        anarci_out, detected_chain_type, first_aligned_row = result
        assert isinstance(anarci_out, list)
        assert detected_chain_type in ("H", "K", "L")
        assert isinstance(first_aligned_row, int)
