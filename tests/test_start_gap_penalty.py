"""Tests for start gap penalty feature.

The start gap penalty penalizes alignments where the query starts after
reference position 1. This encourages N-terminal alignment by treating
position 0 as a "ghost" aligned residue.

Penalty formula for starting at reference column j > 0:
    penalty = gap_open + gap_extend * (j - 1)
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from softalign import SW


class TestSWStartGapPenalty:
    """Tests for start gap penalty in Smith-Waterman alignment."""

    def test_sw_affine_accepts_penalize_start_gap_parameter(self):
        """Test that sw_affine returns a function that accepts the parameter."""
        sw_func = SW.sw_affine(batch=False)

        # Create a simple similarity matrix
        sim = jnp.array([[1.0, 0.5], [0.5, 1.0]])
        lens = (2, 2)
        gap = -0.5
        open_pen = -1.0
        temp = 1.0

        # Should work with penalize_start_gap=False (default)
        score_no_penalty, aln_no_penalty = sw_func(
            sim, lens, gap, open_pen, temp, penalize_start_gap=False
        )
        assert score_no_penalty is not None
        assert aln_no_penalty is not None

        # Should work with penalize_start_gap=True
        score_with_penalty, aln_with_penalty = sw_func(
            sim, lens, gap, open_pen, temp, penalize_start_gap=True
        )
        assert score_with_penalty is not None
        assert aln_with_penalty is not None

    def test_start_gap_penalty_no_effect_on_position_1_start(self):
        """Test that penalty has no effect when alignment starts at position 1.

        When the optimal alignment already starts at the first reference
        column (column 0), the start gap penalty should not change the score.
        """
        sw_func = SW.sw_affine(batch=False)

        # Create a similarity matrix that strongly favors starting at column 0
        # 3x3 matrix where diagonal is strongly preferred
        sim = jnp.array([
            [10.0, -5.0, -5.0],
            [-5.0, 10.0, -5.0],
            [-5.0, -5.0, 10.0],
        ])
        lens = (3, 3)
        gap = -0.5
        open_pen = -1.0
        temp = 0.1  # Low temperature for more deterministic alignment

        score_no_penalty, _ = sw_func(sim, lens, gap, open_pen, temp, penalize_start_gap=False)
        score_with_penalty, _ = sw_func(sim, lens, gap, open_pen, temp, penalize_start_gap=True)

        # Scores should be very similar when alignment starts at position 1
        # Small difference may exist due to soft-max behavior
        assert abs(float(score_no_penalty) - float(score_with_penalty)) < 1.0

    def test_start_gap_penalty_reduces_score_for_late_start(self):
        """Test that penalty reduces score when alignment starts late.

        When the optimal alignment would start at column j > 0, the
        penalize_start_gap=True should reduce the score.
        """
        sw_func = SW.sw_affine(batch=False)

        # Create a similarity matrix that favors starting at column 2
        # (i.e., skipping columns 0 and 1)
        sim = jnp.array([
            [-10.0, -10.0, 10.0, 10.0],
            [-10.0, -10.0, 10.0, 10.0],
        ])
        lens = (2, 4)
        gap = -0.5
        open_pen = -1.0
        temp = 0.1

        score_no_penalty, _ = sw_func(sim, lens, gap, open_pen, temp, penalize_start_gap=False)
        score_with_penalty, _ = sw_func(sim, lens, gap, open_pen, temp, penalize_start_gap=True)

        # Score with penalty should be lower because alignment starts late
        assert float(score_with_penalty) < float(score_no_penalty)

    def test_start_gap_penalty_amount(self):
        """Test that the penalty amount matches expected formula.

        For starting at column j > 0:
        penalty = gap_open + gap_extend * (j - 1)

        For j=2: penalty = open + gap * 1
        For j=3: penalty = open + gap * 2
        """
        sw_func = SW.sw_affine(batch=False)

        gap = -0.5
        open_pen = -1.0

        # Matrix that forces alignment to start at column 2 (0-indexed)
        # Very negative values at columns 0,1 force skipping them
        sim = jnp.array([
            [-10.0, -10.0, 10.0, 10.0],
            [-10.0, -10.0, 10.0, 10.0],
        ])
        lens = (2, 4)
        temp = 0.1

        score_no_pen, _ = sw_func(sim, lens, gap, open_pen, temp, penalize_start_gap=False)
        score_with_pen, _ = sw_func(sim, lens, gap, open_pen, temp, penalize_start_gap=True)

        # Expected penalty for starting at col 2: open + gap * 1 = -1.0 + -0.5
        expected_penalty = open_pen + gap * 1  # = -1.5
        score_diff = float(score_with_pen) - float(score_no_pen)

        # The difference should be close to the expected penalty
        # Allow some tolerance due to soft-max behavior
        assert abs(score_diff - expected_penalty) < 0.5

    def test_batched_sw_affine_with_start_gap_penalty(self):
        """Test that batched version works with start gap penalty."""
        sw_func = SW.sw_affine(batch=True)

        # Batch of 2 similarity matrices
        sim = jnp.array([
            [[1.0, 0.5], [0.5, 1.0]],
            [[1.0, 0.3], [0.3, 1.0]],
        ])
        lens = jnp.array([[2, 2], [2, 2]])
        gap = -0.5
        open_pen = -1.0
        temp = 1.0

        # For batched version, must pass gap_matrix and open_matrix (as None) positionally
        # because vmap expects all 8 arguments
        scores, alns = sw_func(sim, lens, gap, open_pen, temp, None, None, False)
        assert scores.shape == (2,)
        assert alns.shape == (2, 2, 2)

        scores_pen, alns_pen = sw_func(sim, lens, gap, open_pen, temp, None, None, True)
        assert scores_pen.shape == (2,)
        assert alns_pen.shape == (2, 2, 2)

    def test_backward_compatibility_default_false(self):
        """Test that default behavior (no penalty) is preserved."""
        sw_func = SW.sw_affine(batch=False)

        sim = jnp.array([[1.0, 0.5], [0.5, 1.0]])
        lens = (2, 2)
        gap = -0.5
        open_pen = -1.0
        temp = 1.0

        # Call with penalize_start_gap=False should match original behavior
        score, aln = sw_func(sim, lens, gap, open_pen, temp, penalize_start_gap=False)

        # Score should be reasonable (positive for this favorable matrix)
        assert float(score) > 0


class TestStartGapPenaltyIntegration:
    """Integration tests for start gap penalty through the full stack."""

    @pytest.mark.slow
    def test_softaligner_accepts_penalize_start_gap(self):
        """Test that SoftAligner can be initialized with penalize_start_gap."""
        from sabr import softaligner

        # Should accept the parameter without error
        aligner_true = softaligner.SoftAligner(penalize_start_gap=True)
        aligner_false = softaligner.SoftAligner(penalize_start_gap=False)

        assert aligner_true.penalize_start_gap is True
        assert aligner_false.penalize_start_gap is False

    @pytest.mark.slow
    def test_alignment_backend_accepts_penalize_start_gap(self):
        """Test that AlignmentBackend.align accepts penalize_start_gap."""
        from sabr import jax_backend

        backend = jax_backend.AlignmentBackend()

        # Create mock embeddings
        input_emb = np.random.randn(10, 64).astype(np.float32)
        target_emb = np.random.randn(20, 64).astype(np.float32)
        target_stdev = np.ones((20, 64), dtype=np.float32)

        # Should work with both settings
        aln_false, sim_false, score_false = backend.align(
            input_emb, target_emb, target_stdev,
            temperature=1.0, penalize_start_gap=False
        )
        aln_true, sim_true, score_true = backend.align(
            input_emb, target_emb, target_stdev,
            temperature=1.0, penalize_start_gap=True
        )

        assert aln_false.shape == aln_true.shape
        assert sim_false.shape == sim_true.shape


class TestStartGapPenaltyEdgeCases:
    """Edge case tests for start gap penalty."""

    def test_small_matrix(self):
        """Test with a small 2x2 matrix (minimum size for sw_affine)."""
        sw_func = SW.sw_affine(batch=False)

        # Note: sw_affine uses x[:-1,:-1] internally, so minimum usable size is 2x2
        sim = jnp.array([[5.0, 3.0], [3.0, 5.0]])
        lens = (2, 2)
        gap = -0.5
        open_pen = -1.0
        temp = 1.0

        # Should work without error
        score, aln = sw_func(sim, lens, gap, open_pen, temp, penalize_start_gap=True)
        assert score is not None
        assert aln.shape == (2, 2)

    def test_longer_query_than_reference(self):
        """Test with query longer than reference."""
        sw_func = SW.sw_affine(batch=False)

        sim = jnp.array([
            [1.0, 0.5, 0.3],
            [0.5, 1.0, 0.3],
            [0.3, 0.3, 1.0],
            [0.2, 0.2, 0.5],
        ])
        lens = (4, 3)
        gap = -0.5
        open_pen = -1.0
        temp = 1.0

        # Should work without error
        score, aln = sw_func(sim, lens, gap, open_pen, temp, penalize_start_gap=True)
        assert score is not None
        assert aln.shape == (4, 3)

    def test_equal_length_sequences(self):
        """Test with equal length query and reference."""
        sw_func = SW.sw_affine(batch=False)

        n = 5
        sim = jnp.eye(n) * 10 - 5  # Diagonal favored
        lens = (n, n)
        gap = -0.5
        open_pen = -1.0
        temp = 0.5

        score_no_pen, _ = sw_func(sim, lens, gap, open_pen, temp, penalize_start_gap=False)
        score_with_pen, _ = sw_func(sim, lens, gap, open_pen, temp, penalize_start_gap=True)

        # For diagonal alignment starting at position 1, scores should be close
        assert abs(float(score_no_pen) - float(score_with_pen)) < 2.0
