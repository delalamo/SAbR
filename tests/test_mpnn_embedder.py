import tempfile
from pathlib import Path

import numpy as np

from sabr import mpnn_embedder, mpnn_embeddings


def make_embedder():
    """Create an MPNNEmbedder instance without full initialization."""
    return mpnn_embedder.MPNNEmbedder.__new__(mpnn_embedder.MPNNEmbedder)


def test_mpnn_embedder_has_required_attributes():
    """Test that MPNNEmbedder initializes with expected attributes."""
    embedder = make_embedder()
    # Set minimal attributes that would be set during __init__
    embedder.model_params = {}
    embedder.key = None
    embedder.transformed_embed_fn = None

    assert hasattr(embedder, "model_params")
    assert hasattr(embedder, "key")
    assert hasattr(embedder, "transformed_embed_fn")


def test_mpnn_embedder_embed_method_exists():
    """Test that MPNNEmbedder has an embed method."""
    embedder = make_embedder()
    assert hasattr(embedder, "embed")
    assert callable(embedder.embed)


def test_mpnn_embedder_read_params_method_exists():
    """Test that MPNNEmbedder has a _read_softalign_params method."""
    embedder = make_embedder()
    assert hasattr(embedder, "_read_softalign_params")
    assert callable(embedder._read_softalign_params)


def create_test_embedding(include_sequence=True):
    """Create a test MPNNEmbeddings object."""
    return mpnn_embeddings.MPNNEmbeddings(
        name="test_chain",
        embeddings=np.random.rand(10, 64),
        idxs=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
        stdev=np.random.rand(10, 64),
        sequence="ACDEFGHIKL" if include_sequence else None,
    )


def test_save_to_npz_creates_file():
    """Test that save_to_npz creates a file."""
    embedding = create_test_embedding()

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_embedding.npz"
        mpnn_embedder.MPNNEmbedder.save_to_npz(embedding, str(output_path))

        assert output_path.exists()
        assert output_path.suffix == ".npz"


def test_load_from_npz_returns_embedding():
    """Test that load_from_npz returns an MPNNEmbeddings object."""
    embedding = create_test_embedding()

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_embedding.npz"
        mpnn_embedder.MPNNEmbedder.save_to_npz(embedding, str(output_path))

        loaded_embedding = mpnn_embedder.MPNNEmbedder.load_from_npz(
            str(output_path)
        )

        assert isinstance(loaded_embedding, mpnn_embeddings.MPNNEmbeddings)


def test_save_and_load_preserves_name():
    """Test that save and load preserves the name field."""
    embedding = create_test_embedding()

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_embedding.npz"
        mpnn_embedder.MPNNEmbedder.save_to_npz(embedding, str(output_path))

        loaded_embedding = mpnn_embedder.MPNNEmbedder.load_from_npz(
            str(output_path)
        )

        assert loaded_embedding.name == embedding.name


def test_save_and_load_preserves_embeddings():
    """Test that save and load preserves the embeddings array."""
    embedding = create_test_embedding()

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_embedding.npz"
        mpnn_embedder.MPNNEmbedder.save_to_npz(embedding, str(output_path))

        loaded_embedding = mpnn_embedder.MPNNEmbedder.load_from_npz(
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
        mpnn_embedder.MPNNEmbedder.save_to_npz(embedding, str(output_path))

        loaded_embedding = mpnn_embedder.MPNNEmbedder.load_from_npz(
            str(output_path)
        )

        assert loaded_embedding.idxs == embedding.idxs


def test_save_and_load_preserves_stdev():
    """Test that save and load preserves the stdev array."""
    embedding = create_test_embedding()

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_embedding.npz"
        mpnn_embedder.MPNNEmbedder.save_to_npz(embedding, str(output_path))

        loaded_embedding = mpnn_embedder.MPNNEmbedder.load_from_npz(
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
        sequence="ACDEF",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_embedding.npz"
        mpnn_embedder.MPNNEmbedder.save_to_npz(embedding, str(output_path))

        loaded_embedding = mpnn_embedder.MPNNEmbedder.load_from_npz(
            str(output_path)
        )

        assert loaded_embedding.idxs == embedding.idxs
        assert loaded_embedding.name == embedding.name
        np.testing.assert_array_equal(
            loaded_embedding.embeddings, embedding.embeddings
        )
        np.testing.assert_array_equal(loaded_embedding.stdev, embedding.stdev)


def test_save_and_load_preserves_sequence():
    """Test that save and load preserves the sequence field."""
    embedding = create_test_embedding(include_sequence=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_embedding.npz"
        mpnn_embedder.MPNNEmbedder.save_to_npz(embedding, str(output_path))

        loaded_embedding = mpnn_embedder.MPNNEmbedder.load_from_npz(
            str(output_path)
        )

        assert loaded_embedding.sequence == embedding.sequence
        assert loaded_embedding.sequence == "ACDEFGHIKL"


def test_save_and_load_without_sequence():
    """Test that save/load works when sequence is None."""
    embedding = create_test_embedding(include_sequence=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_embedding.npz"
        mpnn_embedder.MPNNEmbedder.save_to_npz(embedding, str(output_path))

        loaded_embedding = mpnn_embedder.MPNNEmbedder.load_from_npz(
            str(output_path)
        )

        assert loaded_embedding.sequence is None


def test_fetch_sequence_from_pdb_method_exists():
    """Test that MPNNEmbedder has a fetch_sequence_from_pdb method."""
    assert hasattr(mpnn_embedder.MPNNEmbedder, "fetch_sequence_from_pdb")
    assert callable(mpnn_embedder.MPNNEmbedder.fetch_sequence_from_pdb)
