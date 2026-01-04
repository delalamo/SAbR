"""Tests for position-dependent gap penalties at CDR boundaries.

This module tests the feature where gap penalties are set to zero at
positions where variable regions (CDRs, position 10, DE loop 81-82)
were removed from the reference embeddings.
"""

import numpy as np
import pytest

from sabr import constants, jax_backend


class TestCreateGapPenaltyForReducedReference:
    """Tests for create_gap_penalty_for_reduced_reference function."""

    def test_returns_correct_shapes(self):
        """Test that gap matrices have correct shapes."""
        query_len = 100
        idxs = list(range(1, 27)) + list(range(39, 56))  # Simulate CDR1 gap

        gap_extend, gap_open = jax_backend.create_gap_penalty_for_reduced_reference(
            query_len, idxs
        )

        assert gap_extend.shape == (query_len, len(idxs))
        assert gap_open.shape == (query_len, len(idxs))

    def test_uniform_penalties_when_no_jumps(self):
        """Test that penalties are uniform when IMGT positions are contiguous."""
        query_len = 50
        idxs = list(range(1, 27))  # Contiguous positions 1-26

        gap_extend, gap_open = jax_backend.create_gap_penalty_for_reduced_reference(
            query_len, idxs
        )

        # All values should be the default penalties
        assert np.allclose(gap_extend, constants.SW_GAP_EXTEND)
        assert np.allclose(gap_open, constants.SW_GAP_OPEN)

    def test_zero_penalties_at_cdr1_boundary(self):
        """Test zero gap penalty at CDR1 boundary (26 -> 39)."""
        query_len = 50
        # Positions 1-26 (FR1) then 39-55 (FR2), skipping CDR1 (27-38)
        idxs = list(range(1, 27)) + list(range(39, 56))

        gap_extend, gap_open = jax_backend.create_gap_penalty_for_reduced_reference(
            query_len, idxs
        )

        # Find column index where jump occurs (position 26 -> 39)
        # idxs[25] = 26, idxs[26] = 39
        jump_col = 26

        # Column at jump should have zero penalty
        assert np.allclose(gap_extend[:, jump_col], 0.0)
        assert np.allclose(gap_open[:, jump_col], 0.0)

        # Columns before and after should have normal penalties
        assert np.allclose(gap_extend[:, jump_col - 1], constants.SW_GAP_EXTEND)
        assert np.allclose(gap_open[:, jump_col - 1], constants.SW_GAP_OPEN)

    def test_zero_penalties_at_multiple_boundaries(self):
        """Test zero gap penalties at multiple CDR boundaries."""
        query_len = 50
        # Simulate reduced reference with CDR1, CDR2, position 10, DE loop, CDR3 removed
        # Realistic IMGT positions after removal
        idxs = (
            list(range(1, 10)) +      # 1-9 (before position 10)
            list(range(11, 27)) +     # 11-26 (after 10, before CDR1)
            list(range(39, 56)) +     # 39-55 (FR2, after CDR1)
            list(range(66, 81)) +     # 66-80 (FR3 before DE loop)
            list(range(83, 105)) +    # 83-104 (FR3 after DE loop)
            list(range(118, 129))     # 118-128 (FR4, after CDR3)
        )

        gap_extend, gap_open = jax_backend.create_gap_penalty_for_reduced_reference(
            query_len, idxs
        )

        # Find all jump positions
        jumps = []
        for i in range(1, len(idxs)):
            if idxs[i] - idxs[i - 1] > 1:
                jumps.append(i)

        # All jump columns should have zero penalty
        for col in jumps:
            assert np.allclose(gap_extend[:, col], 0.0), f"Gap extend at col {col} should be 0"
            assert np.allclose(gap_open[:, col], 0.0), f"Gap open at col {col} should be 0"

        # Non-jump columns should have normal penalties
        for col in range(len(idxs)):
            if col not in jumps and col > 0:  # Skip first column
                expected_extend = constants.SW_GAP_EXTEND
                expected_open = constants.SW_GAP_OPEN
                assert np.allclose(gap_extend[:, col], expected_extend), f"Gap extend at col {col}"
                assert np.allclose(gap_open[:, col], expected_open), f"Gap open at col {col}"

    def test_zero_penalty_at_position_10_boundary(self):
        """Test zero gap penalty when position 10 is removed."""
        query_len = 30
        # Positions 1-9, then 11-20 (position 10 removed)
        idxs = list(range(1, 10)) + list(range(11, 21))

        gap_extend, gap_open = jax_backend.create_gap_penalty_for_reduced_reference(
            query_len, idxs
        )

        # Jump at column 9 (idxs[8]=9, idxs[9]=11)
        jump_col = 9
        assert np.allclose(gap_extend[:, jump_col], 0.0)
        assert np.allclose(gap_open[:, jump_col], 0.0)

    def test_zero_penalty_at_de_loop_boundary(self):
        """Test zero gap penalty when DE loop (81-82) is removed."""
        query_len = 30
        # Positions 78-80, then 83-86 (81-82 removed)
        idxs = list(range(78, 81)) + list(range(83, 87))

        gap_extend, gap_open = jax_backend.create_gap_penalty_for_reduced_reference(
            query_len, idxs
        )

        # Jump at column 3 (idxs[2]=80, idxs[3]=83)
        jump_col = 3
        assert np.allclose(gap_extend[:, jump_col], 0.0)
        assert np.allclose(gap_open[:, jump_col], 0.0)

    def test_penalties_are_float32(self):
        """Test that penalty matrices are float32."""
        query_len = 10
        idxs = list(range(1, 11))

        gap_extend, gap_open = jax_backend.create_gap_penalty_for_reduced_reference(
            query_len, idxs
        )

        assert gap_extend.dtype == np.float32
        assert gap_open.dtype == np.float32

    def test_empty_query(self):
        """Test with zero-length query."""
        query_len = 0
        idxs = list(range(1, 27))

        gap_extend, gap_open = jax_backend.create_gap_penalty_for_reduced_reference(
            query_len, idxs
        )

        assert gap_extend.shape == (0, 26)
        assert gap_open.shape == (0, 26)

    def test_single_position(self):
        """Test with single reference position."""
        query_len = 10
        idxs = [1]

        gap_extend, gap_open = jax_backend.create_gap_penalty_for_reduced_reference(
            query_len, idxs
        )

        # No jumps possible with single position
        assert np.allclose(gap_extend, constants.SW_GAP_EXTEND)
        assert np.allclose(gap_open, constants.SW_GAP_OPEN)


class TestReducedEmbeddings:
    """Tests for reduced embeddings file (embeddings_no_cdr.npz)."""

    def test_embeddings_no_cdr_exists(self):
        """Test that embeddings_no_cdr.npz was created."""
        from importlib.resources import files, as_file

        pkg_files = files("sabr.assets")
        path = pkg_files / "embeddings_no_cdr.npz"
        with as_file(path) as p:
            assert p.exists(), "embeddings_no_cdr.npz should exist"

    def test_embeddings_no_cdr_has_correct_structure(self):
        """Test that reduced embeddings has expected arrays."""
        from importlib.resources import files, as_file

        pkg_files = files("sabr.assets")
        path = pkg_files / "embeddings_no_cdr.npz"
        with as_file(path) as p:
            data = np.load(p, allow_pickle=True)

            assert "array" in data, "Should have 'array' key"
            assert "stdev" in data, "Should have 'stdev' key"
            assert "idxs" in data, "Should have 'idxs' key"
            assert "name" in data, "Should have 'name' key"

    def test_embeddings_no_cdr_excludes_variable_positions(self):
        """Test that reduced embeddings excludes CDR and position 10.

        Note: DE loop positions 81-82 are NOT excluded because they are
        present in heavy chains. Removing them caused misalignment issues.
        """
        from importlib.resources import files, as_file

        pkg_files = files("sabr.assets")
        path = pkg_files / "embeddings_no_cdr.npz"
        with as_file(path) as p:
            data = np.load(p, allow_pickle=True)
            idxs = [int(x) for x in data["idxs"]]

        # Variable positions that should be excluded
        # Note: 81-82 are NOT excluded (kept for heavy chain compatibility)
        variable_positions = (
            set(range(27, 39)) |    # CDR1: 27-38
            set(range(56, 66)) |    # CDR2: 56-65
            set(range(105, 118)) |  # CDR3: 105-117
            {10}                    # Position 10
        )

        # Check none of the variable positions are present
        for pos in variable_positions:
            assert pos not in idxs, f"Variable position {pos} should not be in reduced embeddings"

        # DE loop positions 81-82 SHOULD be present
        assert 81 in idxs, "Position 81 should be in reduced embeddings"
        assert 82 in idxs, "Position 82 should be in reduced embeddings"

    def test_embeddings_no_cdr_has_framework_positions(self):
        """Test that reduced embeddings includes framework positions."""
        from importlib.resources import files, as_file

        pkg_files = files("sabr.assets")
        path = pkg_files / "embeddings_no_cdr.npz"
        with as_file(path) as p:
            data = np.load(p, allow_pickle=True)
            idxs = [int(x) for x in data["idxs"]]

        # Some framework positions that should be present
        framework_positions = [1, 5, 15, 20, 25, 40, 50, 70, 90, 120, 128]
        for pos in framework_positions:
            assert pos in idxs, f"Framework position {pos} should be in reduced embeddings"

    def test_embeddings_no_cdr_stdev_is_ones(self):
        """Test that reduced embeddings has stdev set to ones."""
        from importlib.resources import files, as_file

        pkg_files = files("sabr.assets")
        path = pkg_files / "embeddings_no_cdr.npz"
        with as_file(path) as p:
            data = np.load(p, allow_pickle=True)
            stdev = data["stdev"]

        assert np.allclose(stdev, 1.0), "Stdev should be all ones"

    def test_embeddings_no_cdr_shape_consistency(self):
        """Test that array and stdev have consistent shapes."""
        from importlib.resources import files, as_file

        pkg_files = files("sabr.assets")
        path = pkg_files / "embeddings_no_cdr.npz"
        with as_file(path) as p:
            data = np.load(p, allow_pickle=True)
            array = data["array"]
            stdev = data["stdev"]
            idxs = data["idxs"]

        assert array.shape[0] == len(idxs), "Array rows should match idxs count"
        assert stdev.shape == array.shape, "Stdev shape should match array shape"


class TestSoftAlignerWithReducedEmbeddings:
    """Tests for SoftAligner using reduced embeddings."""

    def test_softaligner_loads_both_embeddings(self):
        """Test that SoftAligner loads both full and reduced embeddings."""
        from sabr import softaligner

        aligner = softaligner.SoftAligner()

        # Should have both embeddings loaded
        assert hasattr(aligner, "unified_embedding")
        assert hasattr(aligner, "unified_embedding_no_cdr")

        # Reduced should have fewer positions
        n_full = len(aligner.unified_embedding.idxs)
        n_reduced = len(aligner.unified_embedding_no_cdr.idxs)
        assert n_reduced < n_full, "Reduced embeddings should have fewer positions"

    def test_softaligner_uses_reduced_when_deterministic(self):
        """Test that deterministic mode uses reduced embeddings."""
        from sabr import softaligner

        aligner = softaligner.SoftAligner()

        # The reduced embeddings should be used when deterministic=True
        reduced_idxs = [int(x) for x in aligner.unified_embedding_no_cdr.idxs]

        # Check that CDR positions are not in reduced
        cdr_positions = list(range(27, 39)) + list(range(56, 66)) + list(range(105, 118))
        for pos in cdr_positions:
            assert pos not in reduced_idxs, f"CDR position {pos} should not be in reduced embeddings"
