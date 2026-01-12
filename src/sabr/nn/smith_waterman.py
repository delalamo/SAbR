#!/usr/bin/env python3
"""Differentiable Smith-Waterman alignment algorithms.

This module provides smooth, differentiable implementations of the
Smith-Waterman local alignment algorithm. These functions use the
log-sum-exp trick to create soft versions of the dynamic programming
recurrences, enabling gradient-based optimization.

Code adapted from the Smooth-Smith-Waterman paper by Sergey Ovchinnikov
and Sam Petti (Spring 2021).

Key functions:
    sw: Basic Smith-Waterman with linear gap penalty
    sw_affine: Smith-Waterman with affine gap penalty (gap open + extend)
"""

import jax
import jax.numpy as jnp


def sw(unroll: int = 2, batch: bool = True, NINF: float = -1e30):
    """Create a differentiable Smith-Waterman alignment function.

    This function returns a traceback function that computes both the
    alignment score and the soft alignment matrix via backpropagation.

    Args:
        unroll: Loop unrolling factor for jax.lax.scan.
        batch: If True, return a batched version using vmap.
        NINF: Negative infinity value for masking.

    Returns:
        A function that takes (sim_matrix, lengths, gap, temp) and returns
        (scores, soft_alignment).
    """

    def rotate(x):
        """Rotate matrix for striped dynamic programming."""
        a, b = x.shape
        ar = jnp.arange(a)[::-1, None]
        br = jnp.arange(b)[None, :]
        i, j = (br - ar) + (a - 1), (ar + br) // 2
        n, m = (a + b - 1), (a + b) // 2
        output = {
            "x": jnp.full([n, m], NINF).at[i, j].set(x),
            "o": (jnp.arange(n) + a % 2) % 2,
        }
        return output, (jnp.full(m, NINF), jnp.full(m, NINF)), (i, j)

    def sco(x, lengths, gap=0, temp=1.0):
        """Compute scoring matrix using soft maximum."""

        def _soft_maximum(x, axis=None, mask=None):
            def _logsumexp(y):
                y = jnp.maximum(y, NINF)
                if mask is None:
                    return jax.nn.logsumexp(y, axis=axis)
                else:
                    return y.max(axis) + jnp.log(
                        jnp.sum(
                            mask * jnp.exp(y - y.max(axis, keepdims=True)),
                            axis=axis,
                        )
                    )

            return temp * _logsumexp(x / temp)

        def _cond(cond, true, false):
            return cond * true + (1 - cond) * false

        def _pad(x, shape):
            return jnp.pad(x, shape, constant_values=(NINF, NINF))

        def _step(prev, sm):
            h2, h1 = prev
            h1_T = _cond(sm["o"], _pad(h1[:-1], [1, 0]), _pad(h1[1:], [0, 1]))

            Align = h2 + sm["x"]
            Turn_0 = h1 + gap
            Turn_1 = h1_T + gap
            Sky = sm["x"]

            h0 = jnp.stack([Align, Turn_0, Turn_1, Sky], -1)
            h0 = _soft_maximum(h0, -1)
            return (h1, h0), h0

        a, b = x.shape
        real_a, real_b = lengths
        mask = (jnp.arange(a) < real_a)[:, None] * (jnp.arange(b) < real_b)[
            None, :
        ]
        x = x + NINF * (1 - mask)

        sm, prev, idx = rotate(x[:-1, :-1])
        hij = jax.lax.scan(_step, prev, sm, unroll=unroll)[-1][idx]
        return _soft_maximum(hij + x[1:, 1:], mask=mask[1:, 1:])

    traceback = jax.value_and_grad(sco)

    if batch:
        return jax.vmap(traceback, (0, 0, None, None))
    else:
        return traceback


def sw_affine(
    restrict_turns: bool = True,
    penalize_turns: bool = True,
    batch: bool = True,
    unroll: int = 2,
    NINF: float = -1e30,
):
    """Create a differentiable Smith-Waterman function with affine gap penalty.

    Affine gap penalties distinguish between opening a new gap (expensive)
    and extending an existing gap (cheaper), which better models biological
    insertion/deletion events.

    Args:
        restrict_turns: If True, restrict state transitions.
        penalize_turns: If True, apply open penalty on direction changes.
        batch: If True, return a batched version using vmap.
        unroll: Loop unrolling factor for jax.lax.scan.
        NINF: Negative infinity value for masking.

    Returns:
        A function that takes (sim_matrix, lengths, gap, open, temp,
        gap_matrix, open_matrix) and returns (scores, soft_alignment).
        gap_matrix and open_matrix are optional position-dependent penalties.
    """

    def rotate(x):
        """Rotate matrix for vectorized dynamic programming."""
        a, b = x.shape
        ar = jnp.arange(a)[::-1, None]
        br = jnp.arange(b)[None, :]
        i, j = (br - ar) + (a - 1), (ar + br) // 2
        n, m = (a + b - 1), (a + b) // 2
        output = {
            "x": jnp.full([n, m], NINF).at[i, j].set(x),
            "o": (jnp.arange(n) + a % 2) % 2,
        }
        return output, (jnp.full((m, 3), NINF), jnp.full((m, 3), NINF)), (i, j)

    def rotate_gap_matrix(x):
        """Rotate a gap penalty matrix for striped DP, using 0 as fill value."""
        a, b = x.shape
        ar = jnp.arange(a)[::-1, None]
        br = jnp.arange(b)[None, :]
        i, j = (br - ar) + (a - 1), (ar + br) // 2
        n, m = (a + b - 1), (a + b) // 2
        # Use 0.0 as fill value (no penalty outside valid region)
        return jnp.full([n, m], 0.0).at[i, j].set(x)

    def sco(
        x,
        lengths,
        gap=0.0,
        open=0.0,
        temp=1.0,
        gap_matrix=None,
        open_matrix=None,
    ):
        """Fill the scoring matrix with affine gap penalties.

        Args:
            x: Similarity matrix (query_len, target_len).
            lengths: Tuple of (real_query_len, real_target_len).
            gap: Scalar gap extension penalty (used if gap_matrix is None).
            open: Scalar gap open penalty (used if open_matrix is None).
            temp: Temperature for soft maximum.
            gap_matrix: Optional position-dependent gap extension penalties.
            open_matrix: Optional position-dependent gap open penalties.
        """
        use_matrix_gaps = gap_matrix is not None and open_matrix is not None

        def _soft_maximum(x, axis=None, mask=None):
            def _logsumexp(y):
                y = jnp.maximum(y, NINF)
                if mask is None:
                    return jax.nn.logsumexp(y, axis=axis)
                else:
                    return y.max(axis) + jnp.log(
                        jnp.sum(
                            mask * jnp.exp(y - y.max(axis, keepdims=True)),
                            axis=axis,
                        )
                    )

            return temp * _logsumexp(x / temp)

        def _cond(cond, true, false):
            return cond * true + (1 - cond) * false

        def _pad(x, shape):
            return jnp.pad(x, shape, constant_values=(NINF, NINF))

        def _step_scalar(prev, sm):
            """Step using scalar gap penalties."""
            h2, h1 = prev

            Align = jnp.pad(h2, [[0, 0], [0, 1]]) + sm["x"][:, None]
            Right = _cond(sm["o"], _pad(h1[:-1], ([1, 0], [0, 0])), h1)
            Down = _cond(sm["o"], h1, _pad(h1[1:], ([0, 1], [0, 0])))

            if penalize_turns:
                Right = Right + jnp.stack([open, gap, open])
                Down = Down + jnp.stack([open, open, gap])
            else:
                gap_pen = jnp.stack([open, gap, gap])
                Right = Right + gap_pen
                Down = Down + gap_pen

            if restrict_turns:
                Right = Right[:, :2]

            h0_Align = _soft_maximum(Align, -1)
            h0_Right = _soft_maximum(Right, -1)
            h0_Down = _soft_maximum(Down, -1)
            h0 = jnp.stack([h0_Align, h0_Right, h0_Down], axis=-1)
            return (h1, h0), h0

        def _step_matrix(prev, sm):
            """Step using position-dependent gap penalties from matrices."""
            h2, h1 = prev

            Align = jnp.pad(h2, [[0, 0], [0, 1]]) + sm["x"][:, None]
            Right = _cond(sm["o"], _pad(h1[:-1], ([1, 0], [0, 0])), h1)
            Down = _cond(sm["o"], h1, _pad(h1[1:], ([0, 1], [0, 0])))

            # Get position-dependent gap penalties from rotated matrices
            gap_vals = sm["gap"]  # shape: (m,)
            open_vals = sm["open"]  # shape: (m,)

            if penalize_turns:
                # Right: [open, gap, open] per position
                Right = Right + jnp.stack(
                    [open_vals, gap_vals, open_vals], axis=-1
                )
                # Down: [open, open, gap] per position
                Down = Down + jnp.stack(
                    [open_vals, open_vals, gap_vals], axis=-1
                )
            else:
                gap_pen = jnp.stack([open_vals, gap_vals, gap_vals], axis=-1)
                Right = Right + gap_pen
                Down = Down + gap_pen

            if restrict_turns:
                Right = Right[:, :2]

            h0_Align = _soft_maximum(Align, -1)
            h0_Right = _soft_maximum(Right, -1)
            h0_Down = _soft_maximum(Down, -1)
            h0 = jnp.stack([h0_Align, h0_Right, h0_Down], axis=-1)
            return (h1, h0), h0

        a, b = x.shape
        real_a, real_b = lengths
        mask = (jnp.arange(a) < real_a)[:, None] * (jnp.arange(b) < real_b)[
            None, :
        ]
        x = x + NINF * (1 - mask)

        sm, prev, idx = rotate(x[:-1, :-1])

        if use_matrix_gaps:
            # Rotate gap matrices the same way as similarity matrix
            rotated_gap = rotate_gap_matrix(gap_matrix[:-1, :-1])
            rotated_open = rotate_gap_matrix(open_matrix[:-1, :-1])
            sm["gap"] = rotated_gap
            sm["open"] = rotated_open
            hij = jax.lax.scan(_step_matrix, prev, sm, unroll=unroll)[-1][idx]
        else:
            hij = jax.lax.scan(_step_scalar, prev, sm, unroll=unroll)[-1][idx]

        return _soft_maximum(hij + x[1:, 1:, None], mask=mask[1:, 1:, None])

    traceback = jax.value_and_grad(sco)

    if batch:
        # gap_matrix and open_matrix are batched along axis 0 if provided
        return jax.vmap(traceback, (0, 0, None, None, None, 0, 0))
    else:
        return traceback
