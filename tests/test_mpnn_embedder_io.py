"""Tests for MPNNEmbeddings save/load functionality without JAX dependencies."""

import tempfile
from pathlib import Path

import numpy as np

from sabr import mpnn_embeddings


def create_test_embedding(include_sequence=True):
    """Create a test MPNNEmbeddings object."""
    return mpnn_embeddings.MPNNEmbeddings(
        name="test_chain",
        embeddings=np.random.rand(10, 64),
        idxs=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
        stdev=np.random.rand(10, 64),
        sequence="ACDEFGHIKL" if include_sequence else None,
    )


def test_to_npz_creates_file():
    """Test that to_npz creates a file."""
    embedding = create_test_embedding()

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_embedding.npz"
        embedding.to_npz(str(output_path))

        assert output_path.exists()
        assert output_path.suffix == ".npz"


def test_from_npz_returns_embedding():
    """Test that from_npz returns an MPNNEmbeddings object."""
    embedding = create_test_embedding()

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_embedding.npz"
        embedding.to_npz(str(output_path))

        loaded_embedding = mpnn_embeddings.MPNNEmbeddings.from_npz(
            str(output_path)
        )

        assert isinstance(loaded_embedding, mpnn_embeddings.MPNNEmbeddings)


def test_save_and_load_preserves_name():
    """Test that save and load preserves the name field."""
    embedding = create_test_embedding()

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_embedding.npz"
        embedding.to_npz(str(output_path))

        loaded_embedding = mpnn_embeddings.MPNNEmbeddings.from_npz(
            str(output_path)
        )

        assert loaded_embedding.name == embedding.name


def test_save_and_load_preserves_embeddings():
    """Test that save and load preserves the embeddings array."""
    embedding = create_test_embedding()

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_embedding.npz"
        embedding.to_npz(str(output_path))

        loaded_embedding = mpnn_embeddings.MPNNEmbeddings.from_npz(
            str(output_path)
        )

        np.testing.assert_array_equal(
            loaded_embedding.embeddings, embedding.embeddings
        )


def test_save_and_load_preserves_idxs():
    """Test that save and load preserves the idxs list."""
    embedding = create_test_embedding()

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_embedding.npz"
        embedding.to_npz(str(output_path))

        loaded_embedding = mpnn_embeddings.MPNNEmbeddings.from_npz(
            str(output_path)
        )

        assert loaded_embedding.idxs == embedding.idxs


def test_save_and_load_preserves_stdev():
    """Test that save and load preserves the stdev array."""
    embedding = create_test_embedding()

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_embedding.npz"
        embedding.to_npz(str(output_path))

        loaded_embedding = mpnn_embeddings.MPNNEmbeddings.from_npz(
            str(output_path)
        )

        np.testing.assert_array_equal(loaded_embedding.stdev, embedding.stdev)


def test_round_trip_with_different_idxs_formats():
    """Test save/load with different idx formats (strings, numbers)."""
    embedding = mpnn_embeddings.MPNNEmbeddings(
        name="mixed_idxs",
        embeddings=np.random.rand(5, 64),
        idxs=["1", "2A", "3", "4B", "5"],
        stdev=np.random.rand(5, 64),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_embedding.npz"
        embedding.to_npz(str(output_path))

        loaded_embedding = mpnn_embeddings.MPNNEmbeddings.from_npz(
            str(output_path)
        )

        assert loaded_embedding.idxs == embedding.idxs
        assert loaded_embedding.name == embedding.name
        np.testing.assert_array_equal(
            loaded_embedding.embeddings, embedding.embeddings
        )
        np.testing.assert_array_equal(loaded_embedding.stdev, embedding.stdev)


def test_save_and_load_with_pathlib():
    """Test that save/load works with pathlib.Path objects."""
    embedding = create_test_embedding()

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_embedding.npz"
        embedding.to_npz(output_path)

        loaded_embedding = mpnn_embeddings.MPNNEmbeddings.from_npz(output_path)

        assert loaded_embedding.name == embedding.name
        np.testing.assert_array_equal(
            loaded_embedding.embeddings, embedding.embeddings
        )


def test_save_and_load_preserves_sequence():
    """Test that save and load preserves the sequence field."""
    embedding = create_test_embedding(include_sequence=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_embedding.npz"
        embedding.to_npz(str(output_path))

        loaded_embedding = mpnn_embeddings.MPNNEmbeddings.from_npz(
            str(output_path)
        )

        assert loaded_embedding.sequence == embedding.sequence
        assert loaded_embedding.sequence == "ACDEFGHIKL"


def test_save_and_load_without_sequence():
    """Test that save/load works when sequence is None."""
    embedding = create_test_embedding(include_sequence=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_embedding.npz"
        embedding.to_npz(str(output_path))

        loaded_embedding = mpnn_embeddings.MPNNEmbeddings.from_npz(
            str(output_path)
        )

        assert loaded_embedding.sequence is None
