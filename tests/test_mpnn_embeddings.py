import tempfile
from pathlib import Path

import numpy as np
import pytest
from Bio import SeqIO

from sabr import constants, model, mpnn_embeddings


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


@pytest.mark.parametrize(
    "description,stdev_factory,n_embed_rows,expected_shape,error_match",
    [
        # Valid cases
        (
            "1d_correct_length",
            lambda: np.ones(constants.EMBED_DIM) * 0.5,
            3,
            (3, constants.EMBED_DIM),
            None,
        ),
        (
            "2d_matching_shape",
            lambda: np.random.rand(3, constants.EMBED_DIM) * 0.5 + 0.5,
            3,
            (3, constants.EMBED_DIM),
            None,
        ),
        (
            "2d_single_row_broadcasts",
            lambda: np.ones((1, constants.EMBED_DIM)) * 0.7,
            3,
            (3, constants.EMBED_DIM),
            None,
        ),
        (
            "2d_more_rows_truncates",
            lambda: np.arange(5 * constants.EMBED_DIM).reshape(
                5, constants.EMBED_DIM
            ),
            3,
            (3, constants.EMBED_DIM),
            None,
        ),
        # Error cases
        (
            "1d_wrong_length",
            lambda: np.ones(32),
            3,
            None,
            "1D stdev must have length",
        ),
        (
            "2d_fewer_rows_error",
            lambda: np.ones((3, constants.EMBED_DIM)),
            5,
            None,
            "stdev rows fewer than embeddings rows are not allowed",
        ),
        (
            "2d_wrong_embed_dim",
            lambda: np.ones((3, 32)),
            3,
            None,
            "stdev.shape[1] (32)",
        ),
        (
            "3d_error",
            lambda: np.ones((3, constants.EMBED_DIM, 2)),
            3,
            None,
            "stdev must be 1D or 2D array compatible with embeddings",
        ),
    ],
)
def test_mpnnembeddings_stdev_validation(
    description, stdev_factory, n_embed_rows, expected_shape, error_match
):
    """Test stdev validation with various configurations."""
    embedding = np.random.rand(n_embed_rows, constants.EMBED_DIM)
    idxs = [str(i) for i in range(1, n_embed_rows + 1)]
    stdev = stdev_factory()

    if error_match:
        with pytest.raises(ValueError, match=error_match):
            mpnn_embeddings.MPNNEmbeddings(
                name=f"test_{description}",
                embeddings=embedding,
                idxs=idxs,
                stdev=stdev,
            )
    else:
        mp = mpnn_embeddings.MPNNEmbeddings(
            name=f"test_{description}",
            embeddings=embedding,
            idxs=idxs,
            stdev=stdev,
        )
        assert (
            mp.stdev.shape == expected_shape
        ), f"{description}: shape mismatch"


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


def test_from_pdb_is_callable():
    """Test that from_pdb function exists and is callable."""
    assert hasattr(mpnn_embeddings, "from_pdb")
    assert callable(mpnn_embeddings.from_pdb)


def test_from_npz_is_callable():
    """Test that from_npz function exists and is callable."""
    assert hasattr(mpnn_embeddings, "from_npz")
    assert callable(mpnn_embeddings.from_npz)


def test_save_is_callable():
    """Test that save method exists and is callable."""
    embedding = create_test_embedding()
    assert hasattr(embedding, "save")
    assert callable(embedding.save)


def create_test_embedding(
    include_sequence: bool = True,
) -> mpnn_embeddings.MPNNEmbeddings:
    """Create a test MPNNEmbeddings object."""
    return mpnn_embeddings.MPNNEmbeddings(
        name="test_chain",
        embeddings=np.random.rand(10, 64),
        idxs=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
        stdev=np.random.rand(10, 64),
        sequence="ACDEFGHIKL" if include_sequence else None,
    )


def test_save_creates_file():
    """Test that save creates a file."""
    embedding = create_test_embedding()

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_embedding.npz"
        embedding.save(str(output_path))

        assert output_path.exists()
        assert output_path.suffix == ".npz"


def test_from_npz_returns_embedding():
    """Test that from_npz returns an MPNNEmbeddings object."""
    embedding = create_test_embedding()

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_embedding.npz"
        embedding.save(str(output_path))

        loaded_embedding = mpnn_embeddings.from_npz(str(output_path))

        assert isinstance(loaded_embedding, mpnn_embeddings.MPNNEmbeddings)


def test_save_and_load_preserves_all_fields():
    """Test that save and load preserves all fields."""
    embedding = create_test_embedding(include_sequence=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_embedding.npz"
        embedding.save(str(output_path))

        loaded = mpnn_embeddings.from_npz(str(output_path))

        assert loaded.name == embedding.name
        np.testing.assert_array_equal(loaded.embeddings, embedding.embeddings)
        assert loaded.idxs == embedding.idxs
        np.testing.assert_array_equal(loaded.stdev, embedding.stdev)
        assert loaded.sequence == embedding.sequence


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
        embedding.save(str(output_path))

        loaded_embedding = mpnn_embeddings.from_npz(str(output_path))

        assert loaded_embedding.idxs == embedding.idxs
        assert loaded_embedding.name == embedding.name
        np.testing.assert_array_equal(
            loaded_embedding.embeddings, embedding.embeddings
        )
        np.testing.assert_array_equal(loaded_embedding.stdev, embedding.stdev)


def test_save_and_load_without_sequence():
    """Test that save/load works when sequence is None."""
    embedding = create_test_embedding(include_sequence=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_embedding.npz"
        embedding.save(str(output_path))

        loaded_embedding = mpnn_embeddings.from_npz(str(output_path))

        assert loaded_embedding.sequence is None


class DummyModel:
    """Dummy model for testing _embed_pdb function."""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def MPNN(self, X1, mask1, chain1, res1):
        length = res1.shape[-1]
        emb = np.ones((1, length, constants.EMBED_DIM), dtype=float)
        return emb


def test_embed_pdb_returns_embeddings(monkeypatch):
    """Test that _embed_pdb returns MPNNEmbeddings."""

    def fake_get_input_mpnn(pdbfile, chain):
        length = 2
        ids = [f"id_{i}" for i in range(length)]
        X = np.zeros((1, length, 4, 3), dtype=float)
        mask = np.zeros((1, length), dtype=float)
        chain_idx = np.zeros((1, length), dtype=int)
        res = np.zeros((1, length), dtype=int)
        sequence = "A" * length  # Mock sequence matching length
        return mpnn_embeddings.MPNNInputs(
            coords=X,
            mask=mask,
            chain_ids=chain_idx,
            residue_indices=res,
            residue_ids=ids,
            sequence=sequence,
        )

    monkeypatch.setattr(
        mpnn_embeddings, "_get_inputs_mpnn", fake_get_input_mpnn
    )
    monkeypatch.setattr(model.END_TO_END_MODELS, "END_TO_END", DummyModel)

    result = mpnn_embeddings._embed_pdb("fake.pdb", chains="A")

    assert isinstance(result, mpnn_embeddings.MPNNEmbeddings)
    assert result.embeddings.shape == (2, constants.EMBED_DIM)
    assert result.idxs == ["id_0", "id_1"]


def test_embed_pdb_rejects_multi_chain_input(monkeypatch):
    """Test that _embed_pdb rejects multi-chain input."""
    monkeypatch.setattr(model.END_TO_END_MODELS, "END_TO_END", DummyModel)
    with pytest.raises(NotImplementedError):
        mpnn_embeddings._embed_pdb("fake.pdb", chains="AB")


def test_embed_pdb_id_mismatch_raises_error(monkeypatch):
    """Test ValueError when IDs length doesn't match embeddings rows."""

    def fake_get_input_mpnn_mismatch(pdbfile, chain):
        length = 3
        ids = ["id_0", "id_1"]  # Only 2 IDs, but length is 3
        X = np.zeros((1, length, 4, 3), dtype=float)
        mask = np.zeros((1, length), dtype=float)
        chain_idx = np.zeros((1, length), dtype=int)
        res = np.zeros((1, length), dtype=int)
        sequence = "A" * length  # Mock sequence matching coord length
        return mpnn_embeddings.MPNNInputs(
            coords=X,
            mask=mask,
            chain_ids=chain_idx,
            residue_indices=res,
            residue_ids=ids,
            sequence=sequence,
        )

    monkeypatch.setattr(
        mpnn_embeddings,
        "_get_inputs_mpnn",
        fake_get_input_mpnn_mismatch,
    )
    monkeypatch.setattr(model.END_TO_END_MODELS, "END_TO_END", DummyModel)

    with pytest.raises(
        ValueError, match="IDs length.*does not match embeddings rows"
    ):
        mpnn_embeddings._embed_pdb("fake.pdb", chains="A")


def test_get_inputs_mpnn_matches_softalign():
    """Verify that _get_inputs_mpnn produces same output as softalign."""
    from softalign import Input_MPNN

    test_pdb = Path(__file__).parent / "data" / "12e8_imgt.pdb"
    chain = "H"

    # Get outputs from both implementations
    inputs = mpnn_embeddings._get_inputs_mpnn(str(test_pdb), chain=chain)
    old_X, old_mask, old_chain, old_res, old_ids = Input_MPNN.get_inputs_mpnn(
        str(test_pdb), chain=chain
    )

    # Compare shapes
    assert (
        inputs.coords.shape == old_X.shape
    ), f"X shape mismatch: {inputs.coords.shape} vs {old_X.shape}"
    assert (
        inputs.mask.shape == old_mask.shape
    ), f"mask shape mismatch: {inputs.mask.shape} vs {old_mask.shape}"
    assert (
        inputs.chain_ids.shape == old_chain.shape
    ), f"chain shape mismatch: {inputs.chain_ids.shape} vs {old_chain.shape}"
    assert (
        inputs.residue_indices.shape == old_res.shape
    ), f"res shape mismatch: {inputs.residue_indices.shape} vs {old_res.shape}"
    assert len(inputs.residue_ids) == len(
        old_ids
    ), f"ids length mismatch: {len(inputs.residue_ids)} vs {len(old_ids)}"

    # Compare residue IDs (softalign returns numpy array, we return list)
    old_ids_list = list(old_ids)
    assert (
        inputs.residue_ids == old_ids_list
    ), f"IDs mismatch: {inputs.residue_ids} vs {old_ids_list}"

    # Compare coordinates (allowing small numerical differences)
    np.testing.assert_allclose(
        inputs.coords,
        old_X,
        rtol=1e-5,
        atol=1e-5,
        err_msg="Coordinate arrays differ",
    )

    # Compare mask values
    np.testing.assert_array_equal(
        inputs.mask, old_mask, err_msg="Mask arrays differ"
    )

    # Compare chain values
    np.testing.assert_array_equal(
        inputs.chain_ids, old_chain, err_msg="Chain arrays differ"
    )


def test_get_inputs_mpnn_sequence_matches_seqio():
    """Verify that _get_inputs_mpnn sequence matches BioPython SeqIO pdb-atom.

    Note: _get_inputs_mpnn only includes residues with complete backbone atoms
    (N, CA, C), while SeqIO includes all residues (with X for those missing
    backbone atoms). So we compare after removing X residues from SeqIO output.
    """
    test_pdb = Path(__file__).parent / "data" / "12e8_imgt.pdb"
    chain = "H"

    # Get sequence from _get_inputs_mpnn
    inputs = mpnn_embeddings._get_inputs_mpnn(str(test_pdb), chain=chain)
    mpnn_sequence = inputs.sequence

    # Get sequence from BioPython SeqIO pdb-atom
    seqio_sequence = None
    for record in SeqIO.parse(str(test_pdb), "pdb-atom"):
        if record.id.endswith(chain):
            seqio_sequence = str(record.seq)
            break

    assert seqio_sequence is not None, f"Chain {chain} not found via SeqIO"

    # Remove X residues from SeqIO sequence (these are residues missing
    # backbone atoms, which _get_inputs_mpnn skips)
    seqio_sequence_no_x = seqio_sequence.replace("X", "")

    # Both should have the same length and content
    assert len(mpnn_sequence) == len(seqio_sequence_no_x), (
        f"Sequence length mismatch: mpnn={len(mpnn_sequence)}, "
        f"seqio (without X)={len(seqio_sequence_no_x)}"
    )
    assert mpnn_sequence == seqio_sequence_no_x, (
        f"Sequence mismatch:\n"
        f"mpnn:           {mpnn_sequence}\n"
        f"seqio (no X):   {seqio_sequence_no_x}"
    )


def test_get_inputs_mpnn_parses_cif_file():
    """Test that _get_inputs_mpnn correctly parses CIF files."""
    cif_file = Path(__file__).parent / "data" / "test_minimal.cif"

    inputs = mpnn_embeddings._get_inputs_mpnn(str(cif_file), chain="A")

    assert isinstance(inputs, mpnn_embeddings.MPNNInputs)
    assert inputs.coords.shape[1] == 2  # 2 residues
    assert inputs.coords.shape[2] == 4  # N, CA, C, CB
    assert inputs.coords.shape[3] == 3  # x, y, z
    assert len(inputs.residue_ids) == 2
    assert inputs.sequence == "AG"


def test_get_inputs_mpnn_raises_on_missing_chain():
    """Test that requesting a non-existent chain raises ValueError."""
    test_pdb = Path(__file__).parent / "data" / "12e8_imgt.pdb"

    with pytest.raises(
        ValueError, match="Chain 'Z' not found.*Available chains"
    ):
        mpnn_embeddings._get_inputs_mpnn(str(test_pdb), chain="Z")


def test_get_inputs_mpnn_handles_insertion_codes():
    """Test that residues with insertion codes are correctly represented."""
    pdb_file = Path(__file__).parent / "data" / "test_insertion_codes.pdb"

    inputs = mpnn_embeddings._get_inputs_mpnn(str(pdb_file), chain="A")

    assert len(inputs.residue_ids) == 4
    assert inputs.residue_ids == ["52", "52A", "52B", "53"]
    assert inputs.sequence == "AGST"


def test_get_inputs_mpnn_raises_on_empty_chain():
    """Test that a chain with no valid residues raises ValueError."""
    pdb_file = Path(__file__).parent / "data" / "test_no_backbone.pdb"

    with pytest.raises(ValueError, match="No valid residues found"):
        mpnn_embeddings._get_inputs_mpnn(str(pdb_file), chain="A")
