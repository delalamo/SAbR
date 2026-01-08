"""Tests for position-dependent gap penalties at CDR boundaries.

This module tests the feature where gap penalties are set to zero at
positions where variable regions (CDRs) were removed from the reference
embeddings.
"""

import numpy as np

from sabr import constants
from sabr.alignment.backend import create_gap_penalty_for_reduced_reference


class TestCreateGapPenaltyForReducedReference:
    """Tests for create_gap_penalty_for_reduced_reference function."""

    def test_returns_correct_shapes(self):
        """Test that gap matrices have correct shapes."""
        query_len = 100
        idxs = list(range(1, 27)) + list(range(39, 56))  # Simulate CDR1 gap

        gap_extend, gap_open = create_gap_penalty_for_reduced_reference(
            query_len, idxs
        )

        assert gap_extend.shape == (query_len, len(idxs))
        assert gap_open.shape == (query_len, len(idxs))

    def test_uniform_penalties_when_no_jumps(self):
        """Test uniform penalties when IMGT positions are contiguous."""
        query_len = 50
        idxs = list(range(1, 27))  # Contiguous positions 1-26

        gap_extend, gap_open = create_gap_penalty_for_reduced_reference(
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

        gap_extend, gap_open = create_gap_penalty_for_reduced_reference(
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
        # Simulate reduced ref with CDR1 and CDR2 removed
        idxs = (
            list(range(1, 27))  # 1-26 (before CDR1)
            + list(range(39, 56))  # 39-55 (FR2, after CDR1)
            + list(range(66, 105))  # 66-104 (FR3, after CDR2)
        )

        gap_extend, gap_open = create_gap_penalty_for_reduced_reference(
            query_len, idxs
        )

        # Find all jump positions
        jumps = []
        for i in range(1, len(idxs)):
            if idxs[i] - idxs[i - 1] > 1:
                jumps.append(i)

        # All jump columns should have zero penalty
        for col in jumps:
            assert np.allclose(
                gap_extend[:, col], 0.0
            ), f"Gap extend at col {col} should be 0"
            assert np.allclose(
                gap_open[:, col], 0.0
            ), f"Gap open at col {col} should be 0"

        # Non-jump columns should have normal penalties
        non_jumps = [i for i in range(len(idxs)) if i not in jumps]
        for col in non_jumps:
            assert np.allclose(
                gap_extend[:, col], constants.SW_GAP_EXTEND
            ), f"Gap extend at col {col} should be {constants.SW_GAP_EXTEND}"

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
        """Test with actual reference embedding positions from embeddings.npz."""
        # The real reference has 122 positions with jumps at:
        # 31->35 (CDR1 gap of 3)
        # 59->62 (CDR2 gap of 2)
        # 72->74 (FR3 gap of 1)
        query_len = 120

        # Load actual positions from embeddings
        import numpy as np
        from importlib.resources import as_file, files

        path = files("sabr.assets") / "embeddings.npz"
        with as_file(path) as p:
            data = np.load(p, allow_pickle=True)
            idxs = [int(x) for x in data["idxs"]]

        gap_extend, gap_open = create_gap_penalty_for_reduced_reference(
            query_len, idxs
        )

        # Find jumps in actual reference
        jump_cols = []
        for i in range(1, len(idxs)):
            if idxs[i] - idxs[i - 1] > 1:
                jump_cols.append(i)

        # Should have 3 jumps
        assert len(jump_cols) == 3

        # All should have zero penalty
        for col in jump_cols:
            assert np.allclose(gap_extend[:, col], 0.0)
            assert np.allclose(gap_open[:, col], 0.0)
