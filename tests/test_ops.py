import numpy as np

from sabr import (
    constants,
    model,
    mpnn_embeddings,
    softalign_output,
    softaligner,
)


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
    monkeypatch.setattr(model.END_TO_END_MODELS, "END_TO_END", DummyModel)

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

    result = softaligner._align_fn(input, targ)

    assert isinstance(result, softalign_output.SoftAlignOutput)
    assert result.alignment.shape == (2, 3)
    assert result.sim_matrix.shape == (2, 3)
    assert np.all(np.isfinite(result.score))


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

    monkeypatch.setattr(
        model.END_TO_END_MODELS, "END_TO_END", TempCapturingModel
    )

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
    softaligner._align_fn(input, targ, temperature=custom_temp)

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
        model.END_TO_END_MODELS, "END_TO_END", ArrayCapturingModel
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

    softaligner._align_fn(input, targ)

    # Target should be divided by stdev: 4.0 / 2.0 = 2.0
    assert len(captured_target) == 1
    expected_normalized = target_embeddings / target_stdev
    np.testing.assert_array_almost_equal(
        captured_target[0][0], expected_normalized
    )
