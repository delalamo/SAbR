"""Tests for position-dependent gap penalties.

This module tests the feature where gap_open penalties are set to zero for:
1. CDR positions (IMGT 27-38, 56-65, 105-117)
2. IMGT position 10 (commonly absent in antibody sequences)

Gap extend penalties remain at normal values everywhere to limit insertions.
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

    def test_gap_extend_always_normal(self):
        """Test that gap_extend has normal penalties everywhere."""
        query_len = 100
        idxs = list(range(1, 128))

        gap_extend, gap_open = create_gap_penalty_for_reduced_reference(
            query_len, idxs
        )

        # gap_extend should be SW_GAP_EXTEND everywhere
        assert np.allclose(gap_extend, constants.SW_GAP_EXTEND)

    def test_zero_gap_open_for_position_10(self):
        """Test zero gap_open penalty for IMGT position 10."""
        query_len = 50
        idxs = list(range(1, 20))

        gap_extend, gap_open = create_gap_penalty_for_reduced_reference(
            query_len, idxs
        )

        # Position 10 should have zero gap_open
        col_10 = idxs.index(10)
        assert np.allclose(gap_open[:, col_10], 0.0)

        # gap_extend should still be normal at position 10
        assert np.allclose(gap_extend[:, col_10], constants.SW_GAP_EXTEND)

        # Adjacent non-CDR positions should have normal gap_open
        col_9 = idxs.index(9)
        col_11 = idxs.index(11)
        assert np.allclose(gap_open[:, col_9], constants.SW_GAP_OPEN)
        assert np.allclose(gap_open[:, col_11], constants.SW_GAP_OPEN)

    def test_zero_gap_open_for_cdr_positions(self):
        """Test that CDR positions have zero gap_open penalties."""
        query_len = 130
        idxs = list(range(1, 128))

        gap_extend, gap_open = create_gap_penalty_for_reduced_reference(
            query_len, idxs
        )

        # CDR positions should have zero gap_open
        cdr_positions = set()
        for cdr_name in ["CDR1", "CDR2", "CDR3"]:
            cdr_positions.update(constants.IMGT_REGIONS[cdr_name])

        for col, pos in enumerate(idxs):
            if pos in cdr_positions:
                assert np.allclose(
                    gap_open[:, col], 0.0
                ), f"CDR pos {pos} should have zero gap_open"
            elif pos == 10:
                assert np.allclose(
                    gap_open[:, col], 0.0
                ), "Position 10 should have zero gap_open"
            else:
                assert np.allclose(
                    gap_open[:, col], constants.SW_GAP_OPEN
                ), f"FR pos {pos} should have normal gap_open"

    def test_framework_positions_have_normal_gap_open(self):
        """Test that framework positions have normal gap_open penalties."""
        query_len = 130
        # Just FR1 and FR2 positions (no CDRs, no position 10)
        idxs = list(range(1, 10)) + list(range(11, 27)) + list(range(39, 56))

        gap_extend, gap_open = create_gap_penalty_for_reduced_reference(
            query_len, idxs
        )

        # All positions should have normal gap_open (no CDRs, no pos 10)
        assert np.allclose(gap_open, constants.SW_GAP_OPEN)

    def test_real_reference_embeddings(self):
        """Test with actual reference embedding indices."""
        from importlib.resources import as_file, files

        path = files("sabr.assets") / "embeddings.npz"
        with as_file(path) as p:
            data = np.load(p, allow_pickle=True)
            idxs = [int(x) for x in data["idxs"]]

        query_len = len(idxs)

        gap_extend, gap_open = create_gap_penalty_for_reduced_reference(
            query_len, idxs
        )

        # gap_extend should be normal everywhere
        assert np.allclose(gap_extend, constants.SW_GAP_EXTEND)

        # Build CDR position set
        cdr_positions = set()
        for cdr_name in ["CDR1", "CDR2", "CDR3"]:
            cdr_positions.update(constants.IMGT_REGIONS[cdr_name])

        # Check gap_open values
        for col, pos in enumerate(idxs):
            if pos in cdr_positions or pos == 10:
                assert np.allclose(
                    gap_open[:, col], 0.0
                ), f"Position {pos} should have zero gap_open"
            else:
                assert np.allclose(
                    gap_open[:, col], constants.SW_GAP_OPEN
                ), f"Position {pos} should have normal gap_open"

    def test_dtype_is_float32(self):
        """Test that gap matrices have float32 dtype."""
        query_len = 50
        idxs = list(range(1, 27))

        gap_extend, gap_open = create_gap_penalty_for_reduced_reference(
            query_len, idxs
        )

        assert gap_extend.dtype == np.float32
        assert gap_open.dtype == np.float32

    def test_cdr_regions_match_constants(self):
        """Verify CDR regions used match constants.IMGT_REGIONS."""
        assert constants.IMGT_REGIONS["CDR1"] == list(range(27, 39))
        assert constants.IMGT_REGIONS["CDR2"] == list(range(56, 66))
        assert constants.IMGT_REGIONS["CDR3"] == list(range(105, 118))

    def test_with_anchors(self):
        """Test gap penalties with anchor positions 0 and 129."""
        query_len = 100
        # Anchors at 0 and 129, real positions in between
        idxs = [0] + list(range(1, 50)) + [129]

        gap_extend, gap_open = create_gap_penalty_for_reduced_reference(
            query_len, idxs, include_anchors=True
        )

        # Anchor positions (0 and 129) are not CDR positions
        # so they should have normal gap_open
        assert np.allclose(gap_open[:, 0], constants.SW_GAP_OPEN)
        assert np.allclose(gap_open[:, -1], constants.SW_GAP_OPEN)

        # Position 10 should still have zero gap_open
        col_10 = idxs.index(10)
        assert np.allclose(gap_open[:, col_10], 0.0)
