#!/usr/bin/env python3
"""Tests for start and end gap penalty functionality."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sabr.nn import smith_waterman


class TestSwAffineGapPenalties:
    """Tests for gap penalty parameters in sw_affine function."""

    @pytest.fixture
    def simple_similarity_matrix(self):
        """Create a simple similarity matrix for testing."""
        # 5x6 similarity matrix (query x reference)
        sim = jnp.array(
            [
                [1.0, 0.5, 0.2, 0.1, 0.1, 0.0],
                [0.5, 1.0, 0.5, 0.2, 0.1, 0.0],
                [0.2, 0.5, 1.0, 0.5, 0.2, 0.0],
                [0.1, 0.2, 0.5, 1.0, 0.5, 0.0],
                [0.0, 0.1, 0.2, 0.5, 1.0, 0.0],
            ]
        )
        return sim[None, :, :]  # Add batch dimension

    @pytest.fixture
    def diagonal_similarity_matrix(self):
        """Create a diagonal matrix where best alignment is diagonal."""
        n = 8
        sim = jnp.zeros((n, n))
        for i in range(n):
            sim = sim.at[i, i].set(1.0)
        return sim[None, :, :]

    def test_sw_affine_accepts_gap_penalty_params(self):
        """Verify sw_affine accepts penalize_start_gap and penalize_end_gap."""
        sw_func = smith_waterman.sw_affine(batch=True)

        sim = jnp.ones((1, 5, 5))
        lengths = jnp.array([[5, 5]])

        # Should not raise with gap penalty parameters (all positional)
        score, alignment = sw_func(sim, lengths, -0.5, -1.0, 1.0, True, True)

        assert score.shape == (1,)
        # Alignment is gradient of score w.r.t. input, same shape as input
        assert alignment.shape == (1, 5, 5)

    def test_start_gap_penalty_prefers_early_start(
        self, diagonal_similarity_matrix
    ):
        """Verify start gap penalty prefers alignments at position 0."""
        sw_func = smith_waterman.sw_affine(batch=True)

        n = 8

        # Matrix A: diagonal starting at column 0 (full coverage)
        sim_early = jnp.zeros((1, n, n))
        for i in range(n):
            sim_early = sim_early.at[0, i, i].set(1.0)

        # Matrix B: diagonal starting at column 3 (late start)
        sim_late = jnp.zeros((1, n, n))
        for i in range(5):
            sim_late = sim_late.at[0, i, i + 3].set(1.0)

        lengths = jnp.array([[n, n]])

        # With start gap penalty enabled for both
        score_early, _ = sw_func(
            sim_early, lengths, -0.1, -0.5, 1.0, True, False
        )
        score_late, _ = sw_func(sim_late, lengths, -0.1, -0.5, 1.0, True, False)

        # Early start should score higher (less penalty)
        assert float(score_early[0]) > float(score_late[0])

    def test_end_gap_penalty_prefers_full_coverage(
        self, diagonal_similarity_matrix
    ):
        """Verify end gap penalty prefers full-length alignments."""
        sw_func = smith_waterman.sw_affine(batch=True)

        n = 8

        # Matrix A: diagonal covering full reference (ends at last position)
        sim_full = jnp.zeros((1, n, n))
        for i in range(n):
            sim_full = sim_full.at[0, i, i].set(1.0)

        # Matrix B: diagonal ending early (ends at column 4)
        sim_early_end = jnp.zeros((1, n, n))
        for i in range(5):
            sim_early_end = sim_early_end.at[0, i, i].set(1.0)

        lengths = jnp.array([[n, n]])

        # With end gap penalty enabled for both
        score_full, _ = sw_func(sim_full, lengths, -0.1, -0.5, 1.0, False, True)
        score_early, _ = sw_func(
            sim_early_end, lengths, -0.1, -0.5, 1.0, False, True
        )

        # Full coverage should score higher (no end penalty)
        assert float(score_full[0]) > float(score_early[0])

    def test_full_coverage_scores_highest_with_both_penalties(
        self, diagonal_similarity_matrix
    ):
        """Verify full coverage scores highest with both penalties."""
        sw_func = smith_waterman.sw_affine(batch=True)

        n = 8

        # Full coverage diagonal
        sim_full = jnp.zeros((1, n, n))
        for i in range(n):
            sim_full = sim_full.at[0, i, i].set(1.0)

        # Partial coverage (late start, early end)
        sim_partial = jnp.zeros((1, n, n))
        for i in range(4):
            sim_partial = sim_partial.at[0, i, i + 2].set(1.0)

        lengths = jnp.array([[n, n]])

        # With both penalties enabled
        score_full, _ = sw_func(sim_full, lengths, -0.1, -0.5, 1.0, True, True)
        score_partial, _ = sw_func(
            sim_partial, lengths, -0.1, -0.5, 1.0, True, True
        )

        # Full coverage should score higher than partial
        assert float(score_full[0]) > float(score_partial[0])

    def test_combined_penalties_can_be_applied(self, simple_similarity_matrix):
        """Verify that both start and end penalties can be applied together."""
        sw_func = smith_waterman.sw_affine(batch=True)

        lengths = jnp.array([[5, 6]])

        # All combinations should work without errors
        score_none, _ = sw_func(
            simple_similarity_matrix, lengths, -0.1, -0.5, 1.0, False, False
        )
        score_start, _ = sw_func(
            simple_similarity_matrix, lengths, -0.1, -0.5, 1.0, True, False
        )
        score_end, _ = sw_func(
            simple_similarity_matrix, lengths, -0.1, -0.5, 1.0, False, True
        )
        score_both, _ = sw_func(
            simple_similarity_matrix, lengths, -0.1, -0.5, 1.0, True, True
        )

        # All should produce valid scores (not NaN or Inf)
        assert jnp.isfinite(score_none[0])
        assert jnp.isfinite(score_start[0])
        assert jnp.isfinite(score_end[0])
        assert jnp.isfinite(score_both[0])

    def test_penalty_uses_actual_sequence_length(self):
        """Verify end penalty uses real_b (actual length) not matrix size."""
        sw_func = smith_waterman.sw_affine(batch=True)

        # Create larger matrix but specify shorter actual length
        sim = jnp.ones((1, 10, 10))

        # Actual length is 6, not 10
        lengths = jnp.array([[6, 6]])

        score_no_penalty, _ = sw_func(
            sim, lengths, -0.1, -0.5, 1.0, False, False
        )

        score_with_penalty, _ = sw_func(
            sim, lengths, -0.1, -0.5, 1.0, False, True
        )

        # Scores should be similar since we're within the actual length
        diff = abs(float(score_with_penalty[0]) - float(score_no_penalty[0]))
        # Should not have huge difference since alignment covers actual length
        assert diff < 1.0

    def test_gradient_flows_with_gap_penalties(self):
        """Verify gradients can flow through gap penalty computation."""
        sw_func = smith_waterman.sw_affine(batch=False)

        sim = jnp.ones((5, 5))
        lengths = jnp.array([5, 5])

        def loss_fn(sim_matrix):
            score, _ = sw_func(sim_matrix, lengths, -0.1, -0.5, 1.0, True, True)
            return score

        # Should be able to compute gradient
        grad = jax.grad(loss_fn)(sim)
        assert grad.shape == sim.shape
        assert not jnp.any(jnp.isnan(grad))

    def test_unbatched_default_gap_penalty_values(self):
        """Verify default gap penalty values are False (unbatched)."""
        # Batched version requires all positional args due to vmap signature
        sw_func = smith_waterman.sw_affine(batch=False)

        sim = jnp.ones((5, 5))
        lengths = jnp.array([5, 5])

        # Call without specifying gap penalty params (should use defaults)
        score_default, _ = sw_func(sim, lengths, -0.1, -0.5, 1.0)

        # Call with explicit False values
        score_explicit, _ = sw_func(sim, lengths, -0.1, -0.5, 1.0, False, False)

        # Should produce same results
        np.testing.assert_allclose(
            float(score_default), float(score_explicit), rtol=1e-5
        )


class TestSwAffineJitCompatibility:
    """Tests for JIT compatibility of gap penalty parameters."""

    def test_jit_with_static_gap_penalties(self):
        """Verify JIT works with static gap penalty values."""
        sw_func = jax.jit(smith_waterman.sw_affine(batch=True))

        sim = jnp.ones((1, 5, 5))
        lengths = jnp.array([[5, 5]])

        # Should work with JIT
        score, alignment = sw_func(sim, lengths, -0.1, -0.5, 1.0, True, True)

        assert score.shape == (1,)

    def test_vmap_with_gap_penalties(self):
        """Verify vmap works correctly with gap penalties."""
        sw_func = smith_waterman.sw_affine(batch=True)

        # Batch of 3 similarity matrices
        sim = jnp.ones((3, 5, 5))
        lengths = jnp.array([[5, 5], [5, 5], [5, 5]])

        # With gap penalties
        score_with, alignment_with = sw_func(
            sim, lengths, -0.1, -0.5, 1.0, True, True
        )
        assert score_with.shape == (3,)
        # Alignment is gradient of score w.r.t. input, same shape as input
        assert alignment_with.shape == (3, 5, 5)

        # Without gap penalties
        score_without, alignment_without = sw_func(
            sim, lengths, -0.1, -0.5, 1.0, False, False
        )
        assert score_without.shape == (3,)
        assert alignment_without.shape == (3, 5, 5)
