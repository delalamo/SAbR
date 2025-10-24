import numpy as np
import pytest

from sabr import constants, ops, types


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
    input = types.MPNNEmbeddings(
        embeddings=np.ones((2, constants.EMBED_DIM), dtype=float),
        idxs=list(range(2)),
    )
    targ = types.MPNNEmbeddings(
        embeddings=np.ones((3, constants.EMBED_DIM), dtype=float),
        idxs=list(range(3)),
    )

    result = ops.align_fn(input, targ)

    assert isinstance(result, types.SoftAlignOutput)
    assert result.alignment.shape == (2, 3)
    assert result.sim_matrix.shape == (2, 3)
    assert result.score == pytest.approx(0.5)


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

    assert isinstance(result, types.MPNNEmbeddings)
    assert result.embeddings.shape == (2, constants.EMBED_DIM)
    assert result.idxs == ["id_0", "id_1"]


def test_embed_fn_rejects_multi_chain_input(monkeypatch):
    monkeypatch.setattr(ops.END_TO_END_MODELS, "END_TO_END", DummyModel)
    with pytest.raises(NotImplementedError):
        ops.embed_fn("fake.pdb", chains="AB")
