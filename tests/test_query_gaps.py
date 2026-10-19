"""Query breaks waive deletion opening, never extension or insertion costs."""

import numpy as np
import pytest

from sabr.alignment import _affine_value_and_grad


@pytest.mark.parametrize("length", [1, 2, 5])
@pytest.mark.parametrize("left", [0, 1, 3])
def test_query_boundary_has_exact_opening_discount(length, left):
    n = left + 3
    similarity = np.full((n, n + length), -100, dtype=np.float32)
    for row in range(n):
        similarity[row, row + (length if row > left else 0)] = 20
    args = (similarity, similarity.shape, 0.001, -1.0, -3.0)
    old, _ = _affine_value_and_grad(*args)
    new, grad = _affine_value_and_grad(
        *args, query_gap_indices=frozenset({left})
    )
    assert new == pytest.approx(n * 20 - (length - 1))
    assert new - old == pytest.approx(3.0)
    np.testing.assert_array_equal(grad, similarity == 20)
    wrong, _ = _affine_value_and_grad(
        *args, query_gap_indices=frozenset({left + 1})
    )
    assert wrong == old
    transposed = (similarity.T, similarity.T.shape, 0.001, -1.0, -3.0)
    old_t, _ = _affine_value_and_grad(*transposed)
    new_t, _ = _affine_value_and_grad(
        *transposed, query_gap_indices=frozenset({left})
    )
    assert new_t == old_t


def test_multiple_breaks_and_score_only_agree():
    x = np.full((4, 6), -100, dtype=np.float32)
    x[np.arange(4), [0, 2, 4, 5]] = 20
    args = (x, x.shape, 0.001, -1.0, -3.0)
    old, _ = _affine_value_and_grad(*args)
    new, _ = _affine_value_and_grad(*args, query_gap_indices=frozenset({0, 1}))
    score, grad = _affine_value_and_grad(
        *args, query_gap_indices=frozenset({0, 1}), with_gradient=False
    )
    assert new - old == pytest.approx(6.0)
    assert score == new
    assert grad is None


def test_gap_aware_reverse_pass_matches_finite_difference():
    x = np.random.default_rng(226).normal(size=(5, 7)).astype(np.float32)

    def run(y):
        return _affine_value_and_grad(
            y, y.shape, 0.7, -0.2, -2.5, query_gap_indices=frozenset({1, 3})
        )

    _, grad = run(x)
    for i, j in np.ndindex(x.shape):
        plus, minus = x.copy(), x.copy()
        plus[i, j] += 0.005
        minus[i, j] -= 0.005
        numerical = (run(plus)[0] - run(minus)[0]) / 0.01
        assert grad[i, j] == pytest.approx(numerical, abs=3e-4)


def test_public_api_rescues_vhh_de_loop_deletion():
    from benchmarks.antibody.run import load_case, load_cases
    from sabr import renumber_structure
    from sabr.structure import extract_chain

    case = next(c for c in load_cases() if c["id"] == "1mel_A")
    structure, labels = load_case(case)
    deleted = next(e["row"] for e in labels if e["imgt_position"] == 81)
    chain = structure[0][case["chain"]]
    chain.detach_child(list(chain)[deleted].id)
    data = extract_chain(structure, case["chain"], None)
    assert data.gap_indices == frozenset({deleted - 1})
    with pytest.raises(ValueError, match="structural gap"):
        renumber_structure(structure, case["chain"])
    with pytest.warns(UserWarning, match="Skipping DE loop"):
        output = renumber_structure(
            structure, case["chain"], dangerously_allow_structural_gaps=True
        )
    truth = [
        (e["imgt_position"], e["insertion_code"])
        for e in labels
        if e["row"] != deleted
    ]
    assert [
        (r.id[1], r.id[2].strip()) for r in output[0][case["chain"]]
    ] == truth
