"""Tests for position-dependent gap penalties at CDR positions.

This module tests the feature where gap penalties are set to zero for all
positions within CDR regions (CDR1, CDR2, CDR3) as defined in constants.py.
"""

import numpy as np

from sabr import constants
from sabr.alignment.backend import create_gap_penalty_for_reduced_reference


class TestCreateGapPenaltyForReducedReference:
    """Tests for create_gap_penalty_for_reduced_reference function."""

    def test_returns_correct_shapes(self):
        """Test that gap matrices have correct shapes."""
        query_len = 100
        idxs = list(range(1, 27)) + list(range(39, 56))

        gap_extend, gap_open = create_gap_penalty_for_reduced_reference(
            query_len, idxs
        )

        assert gap_extend.shape == (query_len, len(idxs))
        assert gap_open.shape == (query_len, len(idxs))

    def test_uniform_penalties_for_framework_only(self):
        """Test uniform penalties when all positions are in framework."""
        query_len = 50
        # FR1 positions only (1-26, all framework)
        idxs = list(range(1, 27))

        gap_extend, gap_open = create_gap_penalty_for_reduced_reference(
            query_len, idxs
        )

        # All FR1 positions should have normal penalties
        assert np.allclose(gap_extend, constants.SW_GAP_EXTEND)
        assert np.allclose(gap_open, constants.SW_GAP_OPEN)

    def test_zero_penalties_for_cdr1_positions(self):
        """Test zero gap penalty for CDR1 positions (27-38)."""
        query_len = 50
        # Include FR1 (1-26), CDR1 (27-38), and FR2 (39-55)
        idxs = list(range(1, 56))

        gap_extend, gap_open = create_gap_penalty_for_reduced_reference(
            query_len, idxs
        )

        # CDR1 positions 27-38 should have zero penalty
        for imgt_pos in range(27, 39):
            col = idxs.index(imgt_pos)
            assert np.allclose(
                gap_extend[:, col], 0.0
            ), f"CDR1 pos {imgt_pos} should have zero gap_extend"
            assert np.allclose(
                gap_open[:, col], 0.0
            ), f"CDR1 pos {imgt_pos} should have zero gap_open"

        # FR1 and FR2 positions should have normal penalties
        for imgt_pos in list(range(1, 27)) + list(range(39, 56)):
            col = idxs.index(imgt_pos)
            assert np.allclose(
                gap_extend[:, col], constants.SW_GAP_EXTEND
            ), f"FR pos {imgt_pos} should have normal gap_extend"

    def test_zero_penalties_for_cdr2_positions(self):
        """Test zero gap penalty for CDR2 positions (56-65)."""
        query_len = 50
        # Include FR2 (39-55), CDR2 (56-65), and FR3 (66-80)
        idxs = list(range(39, 81))

        gap_extend, gap_open = create_gap_penalty_for_reduced_reference(
            query_len, idxs
        )

        # CDR2 positions 56-65 should have zero penalty
        for imgt_pos in range(56, 66):
            col = idxs.index(imgt_pos)
            assert np.allclose(gap_extend[:, col], 0.0)
            assert np.allclose(gap_open[:, col], 0.0)

        # FR2 and FR3 positions should have normal penalties
        for imgt_pos in list(range(39, 56)) + list(range(66, 81)):
            col = idxs.index(imgt_pos)
            assert np.allclose(gap_extend[:, col], constants.SW_GAP_EXTEND)

    def test_zero_penalties_for_cdr3_positions(self):
        """Test zero gap penalty for CDR3 positions (105-117)."""
        query_len = 50
        # Include FR3 end (100-104), CDR3 (105-117), and FR4 (118-128)
        idxs = list(range(100, 129))

        gap_extend, gap_open = create_gap_penalty_for_reduced_reference(
            query_len, idxs
        )

        # CDR3 positions 105-117 should have zero penalty
        for imgt_pos in range(105, 118):
            col = idxs.index(imgt_pos)
            assert np.allclose(gap_extend[:, col], 0.0)
            assert np.allclose(gap_open[:, col], 0.0)

        # FR3 and FR4 positions should have normal penalties
        for imgt_pos in list(range(100, 105)) + list(range(118, 129)):
            col = idxs.index(imgt_pos)
            assert np.allclose(gap_extend[:, col], constants.SW_GAP_EXTEND)

    def test_dtype_is_float32(self):
        """Test that gap matrices have float32 dtype."""
        query_len = 50
        idxs = list(range(1, 27))

        gap_extend, gap_open = create_gap_penalty_for_reduced_reference(
            query_len, idxs
        )

        assert gap_extend.dtype == np.float32
        assert gap_open.dtype == np.float32

    def test_with_real_reference_positions(self):
        """Test with actual reference embedding positions."""
        from importlib.resources import as_file, files

        query_len = 120

        path = files("sabr.assets") / "embeddings.npz"
        with as_file(path) as p:
            data = np.load(p, allow_pickle=True)
            idxs = [int(x) for x in data["idxs"]]

        gap_extend, gap_open = create_gap_penalty_for_reduced_reference(
            query_len, idxs
        )

        # Build set of CDR positions
        cdr_positions = set()
        for _name, (start, end) in constants.IMGT_LOOPS.items():
            cdr_positions.update(range(start, end + 1))

        # All CDR columns should have zero penalty
        cdr_cols = [i for i, pos in enumerate(idxs) if pos in cdr_positions]
        for col in cdr_cols:
            assert np.allclose(gap_extend[:, col], 0.0)
            assert np.allclose(gap_open[:, col], 0.0)

        # All non-CDR columns should have normal penalties
        non_cdr_cols = [
            i for i, pos in enumerate(idxs) if pos not in cdr_positions
        ]
        for col in non_cdr_cols:
            assert np.allclose(gap_extend[:, col], constants.SW_GAP_EXTEND)
            assert np.allclose(gap_open[:, col], constants.SW_GAP_OPEN)

    def test_cdr_positions_match_constants(self):
        """Verify CDR positions used match constants.IMGT_LOOPS."""
        # CDR1: 27-38, CDR2: 56-65, CDR3: 105-117
        assert constants.IMGT_LOOPS["CDR1"] == (27, 38)
        assert constants.IMGT_LOOPS["CDR2"] == (56, 65)
        assert constants.IMGT_LOOPS["CDR3"] == (105, 117)
