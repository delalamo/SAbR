"""Tests for position-dependent gap penalties.

This module tests the feature where gap_open penalties are set to zero for
CDR positions (IMGT 27-38, 56-65, 105-117).

Gap extend penalties remain at normal values everywhere to limit insertions.
"""

import numpy as np

from sabr.alignment.backend import (
    SW_GAP_EXTEND,
    SW_GAP_OPEN,
    create_gap_penalty_for_reduced_reference,
)
from sabr.numbering.imgt import IMGT_REGIONS


class TestCreateGapPenaltyForReducedReference:
    """Tests for create_gap_penalty_for_reduced_reference function."""

    def test_returns_correct_shapes(self):
        """Test that gap matrices have correct shapes."""
        query_len = 100
        idxs = list(range(1, 27)) + list(range(39, 56))

        gap_extend, gap_open = create_gap_penalty_for_reduced_reference(query_len, idxs)

        assert gap_extend.shape == (query_len, len(idxs))
        assert gap_open.shape == (query_len, len(idxs))

    def test_gap_extend_always_normal(self):
        """Test that gap_extend has normal penalties everywhere."""
        query_len = 100
        idxs = list(range(1, 128))

        gap_extend, gap_open = create_gap_penalty_for_reduced_reference(query_len, idxs)

        # gap_extend should be SW_GAP_EXTEND everywhere
        assert np.allclose(gap_extend, SW_GAP_EXTEND)

    def test_zero_gap_open_for_cdr_positions(self):
        """Test that CDR positions have zero gap_open penalties."""
        query_len = 130
        idxs = list(range(1, 128))

        gap_extend, gap_open = create_gap_penalty_for_reduced_reference(query_len, idxs)

        # CDR positions should have zero gap_open
        cdr_positions = set()
        for cdr_name in ["CDR1", "CDR2", "CDR3"]:
            cdr_positions.update(IMGT_REGIONS[cdr_name])

        for col, pos in enumerate(idxs):
            if pos in cdr_positions:
                assert np.allclose(gap_open[:, col], 0.0), (
                    f"CDR pos {pos} should have zero gap_open"
                )
            else:
                assert np.allclose(gap_open[:, col], SW_GAP_OPEN), (
                    f"FR pos {pos} should have normal gap_open"
                )

    def test_framework_positions_have_normal_gap_open(self):
        """Test that framework positions have normal gap_open penalties."""
        query_len = 130
        # Just FR1 and FR2 positions (no CDRs)
        idxs = list(range(1, 27)) + list(range(39, 56))

        gap_extend, gap_open = create_gap_penalty_for_reduced_reference(query_len, idxs)

        # All positions should have normal gap_open because no CDR columns are present.
        assert np.allclose(gap_open, SW_GAP_OPEN)

    def test_real_reference_embeddings(self):
        """Test with actual reference embedding indices."""
        from importlib.resources import as_file, files

        path = files("sabr.assets") / "embeddings.npz"
        with as_file(path) as p:
            data = np.load(p, allow_pickle=True)
            split_data = data["arr_0"].item()
            idxs = [int(x) for x in split_data["H"]["idxs"]]

        query_len = len(idxs)

        gap_extend, gap_open = create_gap_penalty_for_reduced_reference(query_len, idxs)

        # gap_extend should be normal everywhere
        assert np.allclose(gap_extend, SW_GAP_EXTEND)

        # Build CDR position set
        cdr_positions = set()
        for cdr_name in ["CDR1", "CDR2", "CDR3"]:
            cdr_positions.update(IMGT_REGIONS[cdr_name])

        # Check gap_open values
        for col, pos in enumerate(idxs):
            if pos in cdr_positions:
                assert np.allclose(gap_open[:, col], 0.0), (
                    f"Position {pos} should have zero gap_open"
                )
            else:
                assert np.allclose(gap_open[:, col], SW_GAP_OPEN), (
                    f"Position {pos} should have normal gap_open"
                )

    def test_dtype_is_float32(self):
        """Test that gap matrices have float32 dtype."""
        query_len = 50
        idxs = list(range(1, 27))

        gap_extend, gap_open = create_gap_penalty_for_reduced_reference(query_len, idxs)

        assert gap_extend.dtype == np.float32
        assert gap_open.dtype == np.float32

    def test_cdr_regions_match_constants(self):
        """Verify CDR regions used match IMGT region constants."""
        assert IMGT_REGIONS["CDR1"] == list(range(27, 39))
        assert IMGT_REGIONS["CDR2"] == list(range(56, 66))
        assert IMGT_REGIONS["CDR3"] == list(range(105, 118))
