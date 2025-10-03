import numpy as np
import pytest

from sabr import types  # adjust import to where the class lives


def test_mpnnembeddings_shape_mismatch_raises():
    # embeddings has 2 rows, idxs has 3 items -> mismatch
    embedding = np.zeros((2, 5), dtype=float)
    idx = ["a", "b", "c"]

    with pytest.raises(ValueError) as excinfo:
        types.MPNNEmbeddings(name="test_case", embeddings=embedding, idxs=idx)

    # Check key parts of the error message
    msg = str(excinfo.value)
    assert "embeddings.shape[0] (2) must match len(idxs) (3)" in msg
    assert "Error raised for test_case" in msg
