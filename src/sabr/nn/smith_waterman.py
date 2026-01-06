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
        penalize_start_gap, penalize_end_gap) and returns
        (scores, soft_alignment).
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

    def rotate_penalty(penalty_1d, a, b):
        """Rotate 1D penalty array into 2D rotated matrix format."""
        ar = jnp.arange(a)[::-1, None]
        br = jnp.arange(b)[None, :]
        i, j = (br - ar) + (a - 1), (ar + br) // 2
        n, m = (a + b - 1), (a + b) // 2

        penalty_2d = penalty_1d[jnp.arange(b)]
        penalty_2d = jnp.broadcast_to(penalty_2d[None, :], (a, b))

        rotated = jnp.full([n, m], 0.0).at[i, j].set(penalty_2d)
        return rotated

    def sco(
        x,
        lengths,
        gap=0.0,
        open=0.0,
        temp=1.0,
        penalize_start_gap=False,
        penalize_end_gap=False,
    ):
        """Fill the scoring matrix with affine gap penalties.

        Args:
            x: Similarity matrix [a, b].
            lengths: Actual sequence lengths [real_a, real_b].
            gap: Gap extension penalty.
            open: Gap opening penalty.
            temp: Temperature for soft maximum.
            penalize_start_gap: If True, penalize alignments starting after
                position 1 of the reference (N-terminus anchoring).
            penalize_end_gap: If True, penalize alignments ending before the
                last position of the reference (C-terminus anchoring).

        Returns:
            Soft maximum alignment score.
        """

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

        def _step_standard(prev, sm):
            """Standard step without start gap penalty."""
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

        def _step_with_sky(prev, inputs):
            """Step with Sky term for start gap penalty."""
            sm, sky_penalty = inputs
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

            Sky = (sm["x"] + sky_penalty)[:, None]
            Sky = jnp.broadcast_to(Sky, (Sky.shape[0], 3))

            h0_Align = _soft_maximum(Align, -1)
            h0_Right = _soft_maximum(Right, -1)
            h0_Down = _soft_maximum(Down, -1)
            h0_Sky = _soft_maximum(Sky, -1)
            h0 = jnp.stack([h0_Align, h0_Right, h0_Down, h0_Sky], axis=-1)
            h0 = _soft_maximum(h0, -1)
            h0 = jnp.broadcast_to(h0[:, None], (h0.shape[0], 3))
            return (h1, h0), h0

        a, b = x.shape
        real_a, real_b = lengths
        mask = (jnp.arange(a) < real_a)[:, None] * (jnp.arange(b) < real_b)[
            None, :
        ]
        x = x + NINF * (1 - mask)

        sm, prev, idx = rotate(x[:-1, :-1])

        def _run_with_sky():
            col_indices = jnp.arange(b - 1)
            start_penalty = jnp.where(
                col_indices == 0, 0.0, open + gap * (col_indices - 1)
            )
            rotated_start_penalty = rotate_penalty(start_penalty, a - 1, b - 1)
            inputs = (sm, rotated_start_penalty)
            return jax.lax.scan(
                _step_with_sky,
                prev,
                inputs,
                unroll=unroll,
            )[
                -1
            ][idx]

        def _run_standard():
            return jax.lax.scan(_step_standard, prev, sm, unroll=unroll)[-1][
                idx
            ]

        hij = jax.lax.cond(penalize_start_gap, _run_with_sky, _run_standard)

        # Compute end gap penalty if enabled
        # Penalize ending before the last valid reference position
        last_col = real_b - 2  # -1 for 0-indexing, -1 for x[1:,1:]
        end_col_indices = jnp.arange(b - 1)
        end_penalty_1d = jnp.where(
            end_col_indices >= last_col,
            0.0,
            open + gap * (last_col - end_col_indices - 1),
        )
        end_penalty_2d = jnp.broadcast_to(
            end_penalty_1d[None, :], (a - 1, b - 1)
        )

        def _apply_end_penalty(h):
            return h + end_penalty_2d[:, :, None]

        def _no_end_penalty(h):
            return h

        hij = jax.lax.cond(
            penalize_end_gap, _apply_end_penalty, _no_end_penalty, hij
        )

        return _soft_maximum(hij + x[1:, 1:, None], mask=mask[1:, 1:, None])

    traceback = jax.value_and_grad(sco)

    if batch:
        return jax.vmap(traceback, (0, 0, None, None, None, None, None))
    else:
        return traceback
