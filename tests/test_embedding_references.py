import numpy as np
import pytest

from sabr.embeddings.references import (
    ReferenceEmbeddings,
    load_reference_embeddings_from_npz,
    resolve_reference_embeddings_name,
)
from sabr.nn.config import EMBED_DIM


def _write_reference_npz(path, labels=("H", "K", "L")):
    split = {
        label: {
            "array": np.zeros((2, EMBED_DIM), dtype=np.float32),
            "idxs": np.array(["1", "2"]),
        }
        for label in labels
    }
    np.savez(path, arr_0=split)


def test_reference_embeddings_validate_position_rows():
    embeddings = ReferenceEmbeddings(
        name="H",
        embeddings=np.zeros((2, EMBED_DIM), dtype=np.float32),
        positions=[1, 2],
    )

    assert embeddings.idxs == ["1", "2"]


def test_reference_loader_requires_split_hkl_schema(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez(path, array=np.zeros((2, EMBED_DIM), dtype=np.float32))

    with pytest.raises(ValueError, match="split H/K/L schema"):
        load_reference_embeddings_from_npz(path)


def test_reference_loader_rejects_missing_labels(tmp_path):
    path = tmp_path / "missing_labels.npz"
    _write_reference_npz(path, labels=("H", "K"))

    with pytest.raises(ValueError, match="labelled exactly H, K, and L"):
        load_reference_embeddings_from_npz(path)


def test_reference_loader_returns_typed_reference_embeddings(tmp_path):
    path = tmp_path / "refs.npz"
    _write_reference_npz(path)

    embeddings = load_reference_embeddings_from_npz(path)

    assert set(embeddings) == {"H", "K", "L"}
    assert all(isinstance(value, ReferenceEmbeddings) for value in embeddings.values())
    assert embeddings["H"].positions == [1, 2]


def test_resolve_reference_embeddings_name_from_noise_level():
    assert resolve_reference_embeddings_name(None) == "embeddings.npz"
    assert resolve_reference_embeddings_name("0.5") == "embeddings_noise_0.5.npz"
