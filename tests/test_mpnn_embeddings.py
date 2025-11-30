import numpy as np
import pytest

from sabr import constants, mpnn_embeddings


def test_mpnnembeddings_valid_creation_with_defaults():
    """Test successful creation with default stdev (None)."""
    embedding = np.random.rand(3, constants.EMBED_DIM)
    idxs = ["1", "2", "3"]

    mp = mpnn_embeddings.MPNNEmbeddings(
        name="test", embeddings=embedding, idxs=idxs
    )

    assert mp.name == "test"
    assert mp.embeddings.shape == (3, constants.EMBED_DIM)
    assert mp.idxs == idxs
    assert mp.stdev is not None
    assert mp.stdev.shape == (3, constants.EMBED_DIM)
    assert np.allclose(mp.stdev, np.ones_like(embedding))


def test_mpnnembeddings_embeddings_idxs_mismatch():
    """Test ValueError when embeddings rows don't match idxs length."""
    embedding = np.zeros((2, constants.EMBED_DIM), dtype=float)
    idxs = ["a", "b", "c"]

    with pytest.raises(ValueError) as excinfo:
        mpnn_embeddings.MPNNEmbeddings(
            name="test_case", embeddings=embedding, idxs=idxs
        )

    msg = str(excinfo.value)
    assert "embeddings.shape[0] (2) must match len(idxs) (3)" in msg
    assert "Error raised for test_case" in msg


def test_mpnnembeddings_wrong_embedding_dimension():
    """Test ValueError when embeddings dimension is wrong."""
    wrong_dim = 32
    embedding = np.zeros((3, wrong_dim), dtype=float)
    idxs = ["1", "2", "3"]

    with pytest.raises(ValueError) as excinfo:
        mpnn_embeddings.MPNNEmbeddings(
            name="wrong_dim", embeddings=embedding, idxs=idxs
        )

    msg = str(excinfo.value)
    assert f"embeddings.shape[1] ({wrong_dim})" in msg
    assert f"constants.EMBED_DIM ({constants.EMBED_DIM})" in msg


def test_mpnnembeddings_1d_stdev_correct_length():
    """Test with 1D stdev of correct length (broadcasts to all rows)."""
    embedding = np.random.rand(3, constants.EMBED_DIM)
    idxs = ["1", "2", "3"]
    stdev_1d = np.ones(constants.EMBED_DIM) * 0.5

    mp = mpnn_embeddings.MPNNEmbeddings(
        name="test", embeddings=embedding, idxs=idxs, stdev=stdev_1d
    )

    assert mp.stdev.shape == (3, constants.EMBED_DIM)
    # Each row should be the broadcasted 1D stdev
    for row in mp.stdev:
        assert np.allclose(row, stdev_1d)


def test_mpnnembeddings_1d_stdev_wrong_length():
    """Test ValueError when 1D stdev has wrong length."""
    embedding = np.random.rand(3, constants.EMBED_DIM)
    idxs = ["1", "2", "3"]
    wrong_length = 32
    stdev_1d = np.ones(wrong_length)

    with pytest.raises(ValueError) as excinfo:
        mpnn_embeddings.MPNNEmbeddings(
            name="bad_stdev", embeddings=embedding, idxs=idxs, stdev=stdev_1d
        )

    msg = str(excinfo.value)
    assert "1D stdev must have length" in msg
    assert f"{constants.EMBED_DIM}" in msg
    assert f"got {wrong_length}" in msg


def test_mpnnembeddings_2d_stdev_matching_shape():
    """Test with 2D stdev matching embeddings shape exactly."""
    embedding = np.random.rand(3, constants.EMBED_DIM)
    idxs = ["1", "2", "3"]
    stdev_2d = np.random.rand(3, constants.EMBED_DIM) * 0.5 + 0.5

    mp = mpnn_embeddings.MPNNEmbeddings(
        name="test", embeddings=embedding, idxs=idxs, stdev=stdev_2d
    )

    assert mp.stdev.shape == (3, constants.EMBED_DIM)
    assert np.allclose(mp.stdev, stdev_2d)


def test_mpnnembeddings_2d_stdev_single_row_broadcasts():
    """Test with 2D stdev (1, EMBED_DIM) - should broadcast to all rows."""
    embedding = np.random.rand(3, constants.EMBED_DIM)
    idxs = ["1", "2", "3"]
    stdev_2d = np.ones((1, constants.EMBED_DIM)) * 0.7

    mp = mpnn_embeddings.MPNNEmbeddings(
        name="test", embeddings=embedding, idxs=idxs, stdev=stdev_2d
    )

    assert mp.stdev.shape == (3, constants.EMBED_DIM)
    # All rows should be broadcasted
    for row in mp.stdev:
        assert np.allclose(row, stdev_2d[0])


def test_mpnnembeddings_2d_stdev_more_rows_truncates():
    """Test with 2D stdev having more rows than embeddings - should truncate."""
    embedding = np.random.rand(3, constants.EMBED_DIM)
    idxs = ["1", "2", "3"]
    # Stdev with 5 rows, but embeddings only has 3
    stdev_2d = np.arange(5 * constants.EMBED_DIM).reshape(
        5, constants.EMBED_DIM
    )

    mp = mpnn_embeddings.MPNNEmbeddings(
        name="test", embeddings=embedding, idxs=idxs, stdev=stdev_2d
    )

    assert mp.stdev.shape == (3, constants.EMBED_DIM)
    # Should use first 3 rows of stdev_2d
    assert np.allclose(mp.stdev, stdev_2d[:3, :])


def test_mpnnembeddings_2d_stdev_fewer_rows_error():
    """Test ValueError when 2D stdev has fewer rows than embeddings."""
    embedding = np.random.rand(5, constants.EMBED_DIM)
    idxs = ["1", "2", "3", "4", "5"]
    stdev_2d = np.ones((3, constants.EMBED_DIM))

    with pytest.raises(ValueError) as excinfo:
        mpnn_embeddings.MPNNEmbeddings(
            name="fewer_stdev", embeddings=embedding, idxs=idxs, stdev=stdev_2d
        )

    msg = str(excinfo.value)
    assert "stdev rows fewer than embeddings rows are not allowed" in msg
    assert "stdev rows=3" in msg
    assert "embeddings rows=5" in msg


def test_mpnnembeddings_2d_stdev_wrong_embed_dim():
    """Test ValueError when 2D stdev has wrong embedding dimension."""
    embedding = np.random.rand(3, constants.EMBED_DIM)
    idxs = ["1", "2", "3"]
    wrong_dim = 32
    stdev_2d = np.ones((3, wrong_dim))

    with pytest.raises(ValueError) as excinfo:
        mpnn_embeddings.MPNNEmbeddings(
            name="wrong_dim_stdev",
            embeddings=embedding,
            idxs=idxs,
            stdev=stdev_2d,
        )

    msg = str(excinfo.value)
    assert f"stdev.shape[1] ({wrong_dim})" in msg
    assert f"constants.EMBED_DIM ({constants.EMBED_DIM})" in msg


def test_mpnnembeddings_3d_stdev_error():
    """Test ValueError when stdev has ndim > 2."""
    embedding = np.random.rand(3, constants.EMBED_DIM)
    idxs = ["1", "2", "3"]
    stdev_3d = np.ones((3, constants.EMBED_DIM, 2))

    with pytest.raises(ValueError) as excinfo:
        mpnn_embeddings.MPNNEmbeddings(
            name="3d_stdev", embeddings=embedding, idxs=idxs, stdev=stdev_3d
        )

    msg = str(excinfo.value)
    assert "stdev must be 1D or 2D array compatible with embeddings" in msg
    assert "got ndim=3" in msg


def test_mpnnembeddings_stdev_as_list():
    """Test that stdev is converted to numpy array via np.asarray."""
    embedding = np.random.rand(3, constants.EMBED_DIM)
    idxs = ["1", "2", "3"]
    stdev_list = [0.5] * constants.EMBED_DIM

    mp = mpnn_embeddings.MPNNEmbeddings(
        name="test", embeddings=embedding, idxs=idxs, stdev=stdev_list
    )

    assert isinstance(mp.stdev, np.ndarray)
    assert mp.stdev.shape == (3, constants.EMBED_DIM)
