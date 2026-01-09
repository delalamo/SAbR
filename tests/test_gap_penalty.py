"""Tests for position-dependent gap penalties.

This module tests the feature where gap penalties are set to zero for:
1. Jump positions in the reference (where idxs[i] - idxs[i-1] > 1)
2. IMGT position 10 (commonly absent in antibody sequences)
3. CDR positions when query is shorter than reference (excess capacity)
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

    def test_zero_penalty_at_jump_positions(self):
        """Test zero penalty at positions where reference has jumps."""
        query_len = 100
        # Reference with jumps: 1-10, then 15-20 (jump at position 15)
        idxs = list(range(1, 11)) + list(range(15, 21))

        gap_extend, gap_open = create_gap_penalty_for_reduced_reference(
            query_len, idxs
        )

        # Column 10 (where idxs jumps from 10 to 15) should have zero penalty
        jump_col = 10  # This is where idxs[10]=15, idxs[9]=10, jump of 5
        assert np.allclose(gap_extend[:, jump_col], 0.0)
        assert np.allclose(gap_open[:, jump_col], 0.0)

        # Non-jump columns (except position 10) should have normal penalties
        for col in range(len(idxs)):
            if col == jump_col:
                continue
            if idxs[col] == 10:
                continue  # Position 10 also has zero penalty
            assert np.allclose(gap_extend[:, col], constants.SW_GAP_EXTEND)
            assert np.allclose(gap_open[:, col], constants.SW_GAP_OPEN)

    def test_zero_penalty_for_position_10(self):
        """Test zero gap penalty for IMGT position 10."""
        query_len = 50
        idxs = list(range(1, 20))

        gap_extend, gap_open = create_gap_penalty_for_reduced_reference(
            query_len, idxs
        )

        # Position 10 should have zero penalty
        col_10 = idxs.index(10)
        assert np.allclose(gap_extend[:, col_10], 0.0)
        assert np.allclose(gap_open[:, col_10], 0.0)

        # Adjacent positions should have normal penalties
        col_9 = idxs.index(9)
        col_11 = idxs.index(11)
        assert np.allclose(gap_extend[:, col_9], constants.SW_GAP_EXTEND)
        assert np.allclose(gap_extend[:, col_11], constants.SW_GAP_EXTEND)

    def test_no_cdr_zeroing_when_query_longer_than_reference(self):
        """Test that CDR positions have normal penalties when query >= ref."""
        query_len = 150  # Longer than reference
        # Reference with CDR1 positions (27-38)
        idxs = list(range(1, 56))

        gap_extend, gap_open = create_gap_penalty_for_reduced_reference(
            query_len, idxs
        )

        # CDR1 positions should have NORMAL penalties (query is long enough)
        for imgt_pos in range(27, 39):
            col = idxs.index(imgt_pos)
            assert np.allclose(
                gap_extend[:, col], constants.SW_GAP_EXTEND
            ), f"CDR1 pos {imgt_pos} should have normal penalty"

    def test_cdr_zeroing_when_query_shorter_than_reference(self):
        """Test CDR positions get zero penalty when query < reference."""
        # Reference has 122 positions, query has 100
        # This means 22 "excess" positions in reference
        from importlib.resources import as_file, files

        path = files("sabr.assets") / "embeddings.npz"
        with as_file(path) as p:
            data = np.load(p, allow_pickle=True)
            idxs = [int(x) for x in data["idxs"]]

        query_len = 100  # Shorter than reference (122)
        target_len = len(idxs)
        excess = target_len - query_len  # 22 excess positions

        gap_extend, gap_open = create_gap_penalty_for_reduced_reference(
            query_len, idxs
        )

        # Some CDR positions at the end of each CDR should have zero penalty
        # to accommodate shorter CDRs
        zero_count = 0
        for _cdr_name, (cdr_start, cdr_end) in constants.IMGT_LOOPS.items():
            cdr_cols = [
                col for col, pos in enumerate(idxs)
                if cdr_start <= pos <= cdr_end
            ]
            for col in cdr_cols:
                if np.allclose(gap_extend[:, col], 0.0):
                    zero_count += 1

        # Should have some CDR positions zeroed out
        assert zero_count > 0, "Expected some CDR positions to have zero penalty"

    def test_real_reference_with_jumps(self):
        """Test with actual reference embedding that has jumps."""
        from importlib.resources import as_file, files

        path = files("sabr.assets") / "embeddings.npz"
        with as_file(path) as p:
            data = np.load(p, allow_pickle=True)
            idxs = [int(x) for x in data["idxs"]]

        # Use query length equal to reference so no CDR zeroing
        query_len = len(idxs)

        gap_extend, gap_open = create_gap_penalty_for_reduced_reference(
            query_len, idxs
        )

        # Find jump positions in reference
        jump_cols = []
        for col in range(1, len(idxs)):
            if idxs[col] - idxs[col - 1] > 1:
                jump_cols.append(col)

        # Reference should have jumps at 31->35, 59->62, 72->74
        assert len(jump_cols) == 3, f"Expected 3 jumps, got {len(jump_cols)}"

        # Jump columns should have zero penalty
        for col in jump_cols:
            assert np.allclose(gap_extend[:, col], 0.0)
            assert np.allclose(gap_open[:, col], 0.0)

        # Position 10 should have zero penalty
        col_10 = idxs.index(10)
        assert np.allclose(gap_extend[:, col_10], 0.0)

        # Framework positions (non-jump, non-pos10) should have normal penalty
        for col, pos in enumerate(idxs):
            if col in jump_cols:
                continue
            if pos == 10:
                continue
            # Check it's a framework position
            is_cdr = any(
                cdr_start <= pos <= cdr_end
                for cdr_start, cdr_end in constants.IMGT_LOOPS.values()
            )
            if not is_cdr:
                assert np.allclose(
                    gap_extend[:, col], constants.SW_GAP_EXTEND
                ), f"FR pos {pos} should have normal penalty"

    def test_dtype_is_float32(self):
        """Test that gap matrices have float32 dtype."""
        query_len = 50
        idxs = list(range(1, 27))

        gap_extend, gap_open = create_gap_penalty_for_reduced_reference(
            query_len, idxs
        )

        assert gap_extend.dtype == np.float32
        assert gap_open.dtype == np.float32

    def test_consecutive_indices_no_jumps(self):
        """Test that consecutive indices produce no jump-based zeroing."""
        query_len = 100
        # Consecutive indices with no jumps
        idxs = list(range(1, 50))

        gap_extend, gap_open = create_gap_penalty_for_reduced_reference(
            query_len, idxs
        )

        # Only position 10 should have zero penalty (no jumps)
        for col, pos in enumerate(idxs):
            if pos == 10:
                assert np.allclose(gap_extend[:, col], 0.0)
            else:
                assert np.allclose(gap_extend[:, col], constants.SW_GAP_EXTEND)

    def test_cdr_positions_match_constants(self):
        """Verify CDR positions used match constants.IMGT_LOOPS."""
        assert constants.IMGT_LOOPS["CDR1"] == (27, 38)
        assert constants.IMGT_LOOPS["CDR2"] == (56, 65)
        assert constants.IMGT_LOOPS["CDR3"] == (105, 117)
