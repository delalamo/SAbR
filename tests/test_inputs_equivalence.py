"""Tests verifying Gemmi and BioPython parsers produce identical results.

These tests ensure that the GemmiResidueAdapter and BioPythonResidueAdapter
produce equivalent MPNNInputs when parsing the same PDB file, which is
critical for reproducibility regardless of which parser path is used.
"""

from pathlib import Path

import gemmi
import numpy as np
import pytest
from Bio.PDB import PDBParser

from sabr.embeddings.inputs import (
    BioPythonResidueAdapter,
    GemmiResidueAdapter,
    MPNNInputs,
    _extract_from_chain,
)


class TestGemmiBioPythonEquivalence:
    """Tests verifying Gemmi and BioPython parsers produce identical results."""

    @pytest.fixture
    def pdb_files(self):
        """Return list of PDB files to test for equivalence."""
        data_dir = Path(__file__).parent / "data"
        return [
            (data_dir / "12e8_imgt.pdb", "H"),
            (data_dir / "test_heavy_chain.pdb", "F"),
            (data_dir / "test_insertion_codes.pdb", "A"),
            (data_dir / "8_21_renumbered.pdb", "A"),
        ]

    def _parse_with_gemmi(self, pdb_path: Path, chain_id: str) -> MPNNInputs:
        """Parse PDB file using Gemmi adapter."""
        structure = gemmi.read_structure(str(pdb_path))
        for ch in structure[0]:
            if ch.name == chain_id:
                adapter = GemmiResidueAdapter(ch)
                return _extract_from_chain(adapter, str(pdb_path))
        raise ValueError(f"Chain {chain_id} not found")

    def _parse_with_biopython(
        self, pdb_path: Path, chain_id: str
    ) -> MPNNInputs:
        """Parse PDB file using BioPython adapter."""
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("test", str(pdb_path))
        chain = structure[0][chain_id]
        adapter = BioPythonResidueAdapter(chain)
        return _extract_from_chain(adapter, str(pdb_path))

    def test_coordinates_are_equivalent(self, pdb_files):
        """Test that Gemmi and BioPython produce identical coordinates."""
        for pdb_path, chain_id in pdb_files:
            if not pdb_path.exists():
                pytest.skip(f"Test file {pdb_path} not found")

            gemmi_inputs = self._parse_with_gemmi(pdb_path, chain_id)
            biopython_inputs = self._parse_with_biopython(pdb_path, chain_id)

            np.testing.assert_allclose(
                gemmi_inputs.coords,
                biopython_inputs.coords,
                rtol=1e-5,
                atol=1e-5,
                err_msg=(
                    f"Coordinate mismatch for {pdb_path.name} chain {chain_id}"
                ),
            )

    def test_residue_ids_are_equivalent(self, pdb_files):
        """Test that Gemmi and BioPython produce identical residue IDs."""
        for pdb_path, chain_id in pdb_files:
            if not pdb_path.exists():
                pytest.skip(f"Test file {pdb_path} not found")

            gemmi_inputs = self._parse_with_gemmi(pdb_path, chain_id)
            biopython_inputs = self._parse_with_biopython(pdb_path, chain_id)

            assert gemmi_inputs.residue_ids == biopython_inputs.residue_ids, (
                f"Residue ID mismatch for {pdb_path.name} chain {chain_id}:\n"
                f"Gemmi:     {gemmi_inputs.residue_ids}\n"
                f"BioPython: {biopython_inputs.residue_ids}"
            )

    def test_sequences_are_equivalent(self, pdb_files):
        """Test that Gemmi and BioPython produce identical sequences."""
        for pdb_path, chain_id in pdb_files:
            if not pdb_path.exists():
                pytest.skip(f"Test file {pdb_path} not found")

            gemmi_inputs = self._parse_with_gemmi(pdb_path, chain_id)
            biopython_inputs = self._parse_with_biopython(pdb_path, chain_id)

            assert gemmi_inputs.sequence == biopython_inputs.sequence, (
                f"Sequence mismatch for {pdb_path.name} chain {chain_id}:\n"
                f"Gemmi:     {gemmi_inputs.sequence}\n"
                f"BioPython: {biopython_inputs.sequence}"
            )

    def test_number_of_residues_match(self, pdb_files):
        """Test that both parsers extract the same number of residues."""
        for pdb_path, chain_id in pdb_files:
            if not pdb_path.exists():
                pytest.skip(f"Test file {pdb_path} not found")

            gemmi_inputs = self._parse_with_gemmi(pdb_path, chain_id)
            biopython_inputs = self._parse_with_biopython(pdb_path, chain_id)

            assert gemmi_inputs.coords.shape == biopython_inputs.coords.shape, (
                f"Shape mismatch for {pdb_path.name} chain {chain_id}:\n"
                f"Gemmi shape:     {gemmi_inputs.coords.shape}\n"
                f"BioPython shape: {biopython_inputs.coords.shape}"
            )

    def test_mask_and_indices_are_equivalent(self, pdb_files):
        """Test that mask and residue_indices arrays are equivalent."""
        for pdb_path, chain_id in pdb_files:
            if not pdb_path.exists():
                pytest.skip(f"Test file {pdb_path} not found")

            gemmi_inputs = self._parse_with_gemmi(pdb_path, chain_id)
            biopython_inputs = self._parse_with_biopython(pdb_path, chain_id)

            np.testing.assert_array_equal(
                gemmi_inputs.mask,
                biopython_inputs.mask,
                err_msg=f"Mask mismatch for {pdb_path.name} chain {chain_id}",
            )
            np.testing.assert_array_equal(
                gemmi_inputs.residue_indices,
                biopython_inputs.residue_indices,
                err_msg=(
                    f"Residue indices mismatch for "
                    f"{pdb_path.name} chain {chain_id}"
                ),
            )

    def test_insertion_codes_handled_consistently(self):
        """Test that insertion codes are parsed identically by both parsers."""
        pdb_path = Path(__file__).parent / "data" / "test_insertion_codes.pdb"
        chain_id = "A"

        if not pdb_path.exists():
            pytest.skip(f"Test file {pdb_path} not found")

        gemmi_inputs = self._parse_with_gemmi(pdb_path, chain_id)
        biopython_inputs = self._parse_with_biopython(pdb_path, chain_id)

        # Specifically check insertion code handling
        assert gemmi_inputs.residue_ids == biopython_inputs.residue_ids
        assert "52A" in gemmi_inputs.residue_ids
        assert "52B" in gemmi_inputs.residue_ids
