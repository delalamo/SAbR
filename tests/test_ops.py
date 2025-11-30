import numpy as np
import pytest

from sabr import constants, mpnn_embeddings, ops, softalign_output


class DummyModel:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def align(self, input_array, target_array, lens, temperature):
        n_in = input_array.shape[1]
        n_out = target_array.shape[1]
        alignment = np.ones((1, n_in, n_out), dtype=int)
        sim_matrix = np.full((1, n_in, n_out), 2.0, dtype=float)
        score = np.array([temperature], dtype=float)
        return alignment, sim_matrix, score

    def MPNN(self, X1, mask1, chain1, res1):
        length = res1.shape[-1]
        emb = np.ones((1, length, constants.EMBED_DIM), dtype=float)
        return emb


def test_align_fn_returns_softalign_output(monkeypatch):
    monkeypatch.setattr(ops.END_TO_END_MODELS, "END_TO_END", DummyModel)

    # MPNN embeddings
    input = mpnn_embeddings.MPNNEmbeddings(
        name="test1",
        embeddings=np.ones((2, constants.EMBED_DIM), dtype=float),
        stdev=np.ones((3, constants.EMBED_DIM), dtype=float),
        idxs=list(range(2)),
    )
    targ = mpnn_embeddings.MPNNEmbeddings(
        name="test2",
        embeddings=np.ones((3, constants.EMBED_DIM), dtype=float),
        stdev=np.ones((3, constants.EMBED_DIM), dtype=float),
        idxs=list(range(3)),
    )

    result = ops.align_fn(input, targ)

    assert isinstance(result, softalign_output.SoftAlignOutput)
    assert result.alignment.shape == (2, 3)
    assert result.sim_matrix.shape == (2, 3)
    assert np.all(np.isfinite(result.score))


def test_embed_fn_returns_embeddings(monkeypatch):
    def fake_get_input_mpnn(pdbfile, chain):
        length = 2
        ids = [f"id_{i}" for i in range(length)]
        X = np.zeros((1, length, 1, 3), dtype=float)
        mask = np.zeros((1, length), dtype=float)
        chain_idx = np.zeros((1, length), dtype=int)
        res = np.zeros((1, length), dtype=int)
        return X, mask, chain_idx, res, ids

    monkeypatch.setattr(ops.Input_MPNN, "get_inputs_mpnn", fake_get_input_mpnn)
    monkeypatch.setattr(ops.END_TO_END_MODELS, "END_TO_END", DummyModel)

    result = ops.embed_fn("fake.pdb", chains="A")

    assert isinstance(result, mpnn_embeddings.MPNNEmbeddings)
    assert result.embeddings.shape == (2, constants.EMBED_DIM)
    assert result.idxs == ["id_0", "id_1"]


def test_embed_fn_rejects_multi_chain_input(monkeypatch):
    monkeypatch.setattr(ops.END_TO_END_MODELS, "END_TO_END", DummyModel)
    with pytest.raises(NotImplementedError):
        ops.embed_fn("fake.pdb", chains="AB")


def test_embed_fn_id_mismatch_raises_error(monkeypatch):
    """Test ValueError when IDs length doesn't match embeddings rows."""

    def fake_get_input_mpnn_mismatch(pdbfile, chain):
        length = 3
        ids = ["id_0", "id_1"]  # Only 2 IDs, but length is 3
        X = np.zeros((1, length, 1, 3), dtype=float)
        mask = np.zeros((1, length), dtype=float)
        chain_idx = np.zeros((1, length), dtype=int)
        res = np.zeros((1, length), dtype=int)
        return X, mask, chain_idx, res, ids

    monkeypatch.setattr(
        ops.Input_MPNN, "get_inputs_mpnn", fake_get_input_mpnn_mismatch
    )
    monkeypatch.setattr(ops.END_TO_END_MODELS, "END_TO_END", DummyModel)

    with pytest.raises(
        ValueError, match="IDs length.*does not match embeddings rows"
    ):
        ops.embed_fn("fake.pdb", chains="A")


def test_align_fn_temperature_parameter(monkeypatch):
    """Test that temperature parameter is passed through correctly."""
    captured_temperature = []

    class TempCapturingModel:
        def __init__(self, *args, **kwargs):
            pass

        def align(self, input_array, target_array, lens, temperature):
            captured_temperature.append(temperature)
            n_in = input_array.shape[1]
            n_out = target_array.shape[1]
            alignment = np.ones((1, n_in, n_out), dtype=int)
            sim_matrix = np.full((1, n_in, n_out), 2.0, dtype=float)
            score = np.array([1.0], dtype=float)
            return alignment, sim_matrix, score

    monkeypatch.setattr(ops.END_TO_END_MODELS, "END_TO_END", TempCapturingModel)

    input = mpnn_embeddings.MPNNEmbeddings(
        name="test1",
        embeddings=np.ones((2, constants.EMBED_DIM), dtype=float),
        stdev=np.ones((2, constants.EMBED_DIM), dtype=float),
        idxs=["1", "2"],
    )
    targ = mpnn_embeddings.MPNNEmbeddings(
        name="test2",
        embeddings=np.ones((3, constants.EMBED_DIM), dtype=float),
        stdev=np.ones((3, constants.EMBED_DIM), dtype=float),
        idxs=["1", "2", "3"],
    )

    custom_temp = 0.5
    ops.align_fn(input, targ, temperature=custom_temp)

    assert len(captured_temperature) == 1
    assert captured_temperature[0] == custom_temp


def test_align_fn_stdev_normalization(monkeypatch):
    """Test that target embeddings are normalized by stdev."""
    captured_target = []

    class ArrayCapturingModel:
        def __init__(self, *args, **kwargs):
            pass

        def align(self, input_array, target_array, lens, temperature):
            captured_target.append(target_array.copy())
            n_in = input_array.shape[1]
            n_out = target_array.shape[1]
            alignment = np.ones((1, n_in, n_out), dtype=int)
            sim_matrix = np.full((1, n_in, n_out), 2.0, dtype=float)
            score = np.array([1.0], dtype=float)
            return alignment, sim_matrix, score

    monkeypatch.setattr(
        ops.END_TO_END_MODELS, "END_TO_END", ArrayCapturingModel
    )

    input = mpnn_embeddings.MPNNEmbeddings(
        name="test1",
        embeddings=np.ones((2, constants.EMBED_DIM), dtype=float),
        stdev=np.ones((2, constants.EMBED_DIM), dtype=float),
        idxs=["1", "2"],
    )

    # Target with stdev = 2.0
    target_embeddings = np.ones((3, constants.EMBED_DIM), dtype=float) * 4.0
    target_stdev = np.ones((3, constants.EMBED_DIM), dtype=float) * 2.0

    targ = mpnn_embeddings.MPNNEmbeddings(
        name="test2",
        embeddings=target_embeddings,
        stdev=target_stdev,
        idxs=["1", "2", "3"],
    )

    ops.align_fn(input, targ)

    # Target should be divided by stdev: 4.0 / 2.0 = 2.0
    assert len(captured_target) == 1
    expected_normalized = target_embeddings / target_stdev
    np.testing.assert_array_almost_equal(
        captured_target[0][0], expected_normalized
    )
