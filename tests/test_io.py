# tests/test_io.py
import numpy as np
import numpy.testing as npt
import pytest

from sabr.io import load_data


def test_load_data_roundtrip(tmp_path):
    # create a temp "embeddings" dir with a dummy .npy file
    embeddings_dir = tmp_path / "embeddings"
    embeddings_dir.mkdir()
    fname = "dummy.npy"
    original = np.arange(6).reshape(2, 3)
    np.save(embeddings_dir / fname, original)

    # pass the directory as the 'package' root
    loaded = load_data(fname, package=embeddings_dir)

    npt.assert_array_equal(loaded, original)


def test_load_data_missing_file(tmp_path):
    embeddings_dir = tmp_path / "embeddings"
    embeddings_dir.mkdir()

    with pytest.raises(FileNotFoundError):
        load_data("no_such.npy", package=embeddings_dir)
