"""Tests verifying file and in-memory BioPython extraction match."""

from pathlib import Path

import numpy as np
import pytest
from Bio.PDB import PDBParser

from sabr.embeddings.inputs import MPNNInputs, get_inputs


class TestBioPythonExtractionEquivalence:
    """Tests verifying parser paths produce identical MPNNInputs."""

    @pytest.fixture
    def pdb_files(self):
        data_dir = Path(__file__).parent / "data"
        return [
            (data_dir / "12e8_imgt.pdb", "H"),
            (data_dir / "test_heavy_chain.pdb", "F"),
            (data_dir / "test_insertion_codes.pdb", "A"),
            (data_dir / "8_21_renumbered.pdb", "A"),
        ]

    def _parse_file(self, pdb_path: Path, chain_id: str) -> MPNNInputs:
        return get_inputs(str(pdb_path), chain=chain_id)

    def _parse_chain(self, pdb_path: Path, chain_id: str) -> MPNNInputs:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("test", str(pdb_path))
        return get_inputs(structure[0][chain_id])

    def test_coordinates_are_equivalent(self, pdb_files):
        for pdb_path, chain_id in pdb_files:
            if not pdb_path.exists():
                pytest.skip(f"Test file {pdb_path} not found")

            file_inputs = self._parse_file(pdb_path, chain_id)
            chain_inputs = self._parse_chain(pdb_path, chain_id)

            np.testing.assert_allclose(
                file_inputs.coords,
                chain_inputs.coords,
                rtol=1e-5,
                atol=1e-5,
            )

    def test_residue_ids_are_equivalent(self, pdb_files):
        for pdb_path, chain_id in pdb_files:
            if not pdb_path.exists():
                pytest.skip(f"Test file {pdb_path} not found")

            file_inputs = self._parse_file(pdb_path, chain_id)
            chain_inputs = self._parse_chain(pdb_path, chain_id)

            assert file_inputs.residue_ids == chain_inputs.residue_ids

    def test_sequences_are_equivalent(self, pdb_files):
        for pdb_path, chain_id in pdb_files:
            if not pdb_path.exists():
                pytest.skip(f"Test file {pdb_path} not found")

            file_inputs = self._parse_file(pdb_path, chain_id)
            chain_inputs = self._parse_chain(pdb_path, chain_id)

            assert file_inputs.sequence == chain_inputs.sequence

    def test_insertion_codes_handled_consistently(self):
        pdb_path = Path(__file__).parent / "data" / "test_insertion_codes.pdb"
        chain_id = "A"

        file_inputs = self._parse_file(pdb_path, chain_id)
        chain_inputs = self._parse_chain(pdb_path, chain_id)

        assert file_inputs.residue_ids == chain_inputs.residue_ids
        assert "52A" in file_inputs.residue_ids
        assert "52B" in file_inputs.residue_ids
