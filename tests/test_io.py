# tests/test_io.py
import numpy as np
import numpy.testing as npt
import pytest

from sabr.io import load_data


def test_load_data_roundtrip(tmp_path):
    """
    Test that data saved as a NumPy array can be loaded correctly using the
    load_data function. This test creates a temporary directory, saves a dummy
    NumPy array to a file, loads the array back using load_data, and asserts
    that the loaded array matches the original. This ensures roundtrip
    integrity for saving and loading embeddings.
    """

    embeddings_dir = tmp_path / "embeddings"
    embeddings_dir.mkdir()
    fname = "dummy.npy"
    original = np.arange(6).reshape(2, 3)
    np.save(embeddings_dir / fname, original)

    loaded = load_data(fname, package=embeddings_dir)

    npt.assert_array_equal(loaded, original)


def test_load_data_missing_file(tmp_path):
    """
    Test that `load_data` raises a FileNotFoundError when attempting to load
    a non-existent file from the specified embeddings directory.
    """

    embeddings_dir = tmp_path / "embeddings"
    embeddings_dir.mkdir()

    with pytest.raises(FileNotFoundError):
        load_data("no_such.npy", package=embeddings_dir)


def test_embeddings_npz_structure(tmp_path):
    """
    Test the structure and integrity of an embeddings .npz file.
    This test creates a temporary directory and generates an .npz file
    containing multiple embedding arrays, each stored as a dictionary with
    'array' and 'idxs' keys. It verifies that the file contains the expected
    keys, that each entry is a 0-dimensional object array holding a
    dictionary, and that the arrays and index lists are consistent in shape
    and content. The test also checks that the index lists match the expected
    sequence for each array.
    Args:
        tmp_path (pathlib.Path): Temporary directory provided by pytest.
    Raises:
        AssertionError: If contents of loaded file don't match expectations
    """

    embeddings_dir = tmp_path / "embeddings"
    embeddings_dir.mkdir()

    keys = ["camelid_H", "human_L", "human_H", "mouse_L", "mouse_H"]

    payload = {}
    for i, key in enumerate(keys, start=1):
        arr = np.arange(i * 6, dtype=np.int32).reshape(
            i * 2, 3
        )  # shape (2i, 3)
        idxs = list(range(arr.shape[0]))  # len(idxs) == arr.shape[0]
        payload[key] = np.array({"array": arr, "idxs": idxs}, dtype=object)

    np.savez(embeddings_dir / "embeddings.npz", **payload)

    loaded = load_data("embeddings.npz", package=embeddings_dir)

    assert set(loaded.files) == set(keys)

    for key in keys:
        obj = loaded[key]
        assert (
            isinstance(obj, np.ndarray)
            and obj.dtype == object
            and obj.shape == ()
        )
        d = obj.item()
        assert isinstance(d, dict)
        assert set(d.keys()) == {"array", "idxs"}

        arr = d["array"]
        idxs = d["idxs"]

        assert isinstance(arr, np.ndarray)
        assert isinstance(idxs, (list, tuple, np.ndarray))
        assert arr.shape[0] == len(idxs)

        npt.assert_array_equal(np.asarray(idxs), np.arange(arr.shape[0]))
