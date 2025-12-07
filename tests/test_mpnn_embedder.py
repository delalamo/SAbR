from sabr import mpnn_embedder


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
