#!/usr/bin/env python3
"""
Unit and integration tests for NPZ input functionality.

These tests verify that:
1. NPZ files can be loaded correctly (both old and new formats)
2. NPZ input produces identical results to PDB input
3. CLI handles NPZ files properly
"""

import subprocess
from pathlib import Path

import numpy as np
import pytest

from sabr import mpnn_embeddings


class TestNPZLoading:
    """Test NPZ file loading functionality."""

    def test_load_new_format_npz(self, tmp_path):
        """Test loading NPZ files with the new format (includes 'name' key)."""
        # Create a test NPZ file with new format
        npz_file = tmp_path / "test_new_format.npz"

        test_data = {
            "name": "test_structure",
            "embeddings": np.random.rand(100, 64),
            "idxs": np.arange(1, 101),
            "stdev": np.ones((100, 64)),
            "sequence": "ACDEFGHIKLMNPQRSTVWY" * 5,
        }

        np.savez(npz_file, **test_data)

        # Load it
        loaded = mpnn_embeddings.from_npz(str(npz_file))

        # Verify
        assert loaded.name == "test_structure"
        assert loaded.embeddings.shape == (100, 64)
        assert len(loaded.idxs) == 100
        assert loaded.sequence is not None

    def test_load_old_format_npz(self, tmp_path):
        """Test loading NPZ files without 'name' key (old format)."""
        # Create a test NPZ file without 'name' key
        npz_file = tmp_path / "test_old_format.npz"

        test_data = {
            "embeddings": np.random.rand(100, 64),
            "idxs": np.arange(1, 101),
            "stdev": np.ones((100, 64)),
            "sequence": "ACDEFGHIKLMNPQRSTVWY" * 5,
        }

        np.savez(npz_file, **test_data)

        # Load it - should use filename as name
        loaded = mpnn_embeddings.from_npz(str(npz_file))

        # Verify
        assert loaded.name == "test_old_format"  # From filename
        assert loaded.embeddings.shape == (100, 64)
        assert len(loaded.idxs) == 100

    def test_load_existing_test_data(self):
        """Test loading the existing test NPZ files."""
        test_data_dir = Path("tests/data")

        # Test heavy chain embeddings
        h_npz = test_data_dir / "12e8_imgt_H_embeddings.npz"
        if h_npz.exists():
            loaded = mpnn_embeddings.from_npz(str(h_npz))
            assert loaded.embeddings.shape[1] == 64  # Embedding dimension
            assert len(loaded.idxs) > 0
            assert loaded.name == "12e8_imgt_H_embeddings"  # From filename

        # Test light chain embeddings
        l_npz = test_data_dir / "12e8_imgt_L_embeddings.npz"
        if l_npz.exists():
            loaded = mpnn_embeddings.from_npz(str(l_npz))
            assert loaded.embeddings.shape[1] == 64
            assert len(loaded.idxs) > 0


class TestNPZvsPDBEquivalence:
    """Test that NPZ input produces identical results to PDB input."""

    @pytest.mark.slow
    @pytest.mark.integration
    def test_12e8_heavy_chain_equivalence(self, tmp_path):
        """Test PDB and NPZ inputs give identical results for 12e8 H chain."""
        test_data_dir = Path("tests/data")
        pdb_file = test_data_dir / "12e8_imgt.pdb"
        npz_file = test_data_dir / "12e8_imgt_H_embeddings.npz"

        if not pdb_file.exists() or not npz_file.exists():
            pytest.skip("Test data not available")

        output_pdb = tmp_path / "from_pdb.pdb"
        output_npz = tmp_path / "from_npz.pdb"

        # Run with PDB input
        result = subprocess.run(
            [
                "sabr",
                "-i",
                str(pdb_file),
                "-c",
                "H",
                "-o",
                str(output_pdb),
                "-n",
                "imgt",
                "--overwrite",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"PDB processing failed: {result.stderr}"

        # Run with NPZ input
        result = subprocess.run(
            [
                "sabr",
                "-i",
                str(npz_file),
                "-c",
                "H",
                "--pdb-file",
                str(pdb_file),
                "-o",
                str(output_npz),
                "-n",
                "imgt",
                "--overwrite",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"NPZ processing failed: {result.stderr}"

        # Compare outputs (ignoring REMARK lines)
        with open(output_pdb) as f1, open(output_npz) as f2:
            lines1 = [line for line in f1 if not line.startswith("REMARK")]
            lines2 = [line for line in f2 if not line.startswith("REMARK")]

        assert lines1 == lines2, "PDB and NPZ outputs differ!"

    @pytest.mark.slow
    @pytest.mark.integration
    def test_12e8_light_chain_equivalence(self, tmp_path):
        """Test PDB and NPZ inputs give identical results for 12e8 L chain."""
        test_data_dir = Path("tests/data")
        pdb_file = test_data_dir / "12e8_imgt.pdb"
        npz_file = test_data_dir / "12e8_imgt_L_embeddings.npz"

        if not pdb_file.exists() or not npz_file.exists():
            pytest.skip("Test data not available")

        output_pdb = tmp_path / "from_pdb.pdb"
        output_npz = tmp_path / "from_npz.pdb"

        # Run with PDB input
        result = subprocess.run(
            [
                "sabr",
                "-i",
                str(pdb_file),
                "-c",
                "L",
                "-o",
                str(output_pdb),
                "-n",
                "imgt",
                "--overwrite",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"PDB processing failed: {result.stderr}"

        # Run with NPZ input
        result = subprocess.run(
            [
                "sabr",
                "-i",
                str(npz_file),
                "-c",
                "L",
                "--pdb-file",
                str(pdb_file),
                "-o",
                str(output_npz),
                "-n",
                "imgt",
                "--overwrite",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"NPZ processing failed: {result.stderr}"

        # Compare outputs
        with open(output_pdb) as f1, open(output_npz) as f2:
            lines1 = [line for line in f1 if not line.startswith("REMARK")]
            lines2 = [line for line in f2 if not line.startswith("REMARK")]

        assert lines1 == lines2, "PDB and NPZ outputs differ!"


class TestCLIValidation:
    """Test CLI input validation for NPZ files."""

    def test_npz_requires_pdb_file(self, tmp_path):
        """Test that NPZ input requires --pdb-file argument."""
        npz_file = tmp_path / "test.npz"
        output = tmp_path / "output.pdb"

        # Create a dummy NPZ file
        np.savez(
            npz_file,
            embeddings=np.random.rand(50, 64),
            idxs=np.arange(50),
            stdev=np.ones((50, 64)),
        )

        # Try to run without --pdb-file
        result = subprocess.run(
            [
                "sabr",
                "-i",
                str(npz_file),
                "-c",
                "H",
                "-o",
                str(output),
                "-n",
                "imgt",
            ],
            capture_output=True,
            text=True,
        )

        # Should fail with error message about missing --pdb-file
        assert result.returncode != 0
        assert (
            "pdb-file" in result.stderr.lower()
            or "pdb file" in result.stderr.lower()
        )

    def test_npz_requires_chain(self, tmp_path):
        """Test that NPZ input requires chain ID."""
        npz_file = tmp_path / "test.npz"
        pdb_file = tmp_path / "test.pdb"
        output = tmp_path / "output.pdb"

        # Create dummy files
        np.savez(
            npz_file,
            embeddings=np.random.rand(50, 64),
            idxs=np.arange(50),
            stdev=np.ones((50, 64)),
        )
        pdb_file.write_text(
            "ATOM      1  CA  ALA A   1       0.000   0.000   0.000\n"
        )

        # Try to run without -c
        result = subprocess.run(
            [
                "sabr",
                "-i",
                str(npz_file),
                "--pdb-file",
                str(pdb_file),
                "-o",
                str(output),
                "-n",
                "imgt",
            ],
            capture_output=True,
            text=True,
        )

        # Should fail - chain is required
        assert result.returncode != 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
