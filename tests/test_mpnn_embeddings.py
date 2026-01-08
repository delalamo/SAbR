import tempfile
from pathlib import Path

import numpy as np
import pytest
from Bio import SeqIO

from sabr import constants
from sabr.embeddings import mpnn as mpnn_embeddings
from sabr.embeddings.inputs import MPNNInputs
from sabr.embeddings.inputs import get_inputs as _get_inputs


def test_mpnnembeddings_valid_creation_with_defaults():
    """Test successful creation with required fields."""
    embedding = np.random.rand(3, constants.EMBED_DIM)
    idxs = ["1", "2", "3"]

    mp = mpnn_embeddings.MPNNEmbeddings(
        name="test", embeddings=embedding, idxs=idxs
    )

    assert mp.name == "test"
    assert mp.embeddings.shape == (3, constants.EMBED_DIM)
    assert mp.idxs == idxs


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


def test_save_and_load_preserves_name():
    """Test that save and load preserves the name field."""
    embedding = create_test_embedding()

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_embedding.npz"
        embedding.save(str(output_path))

        loaded_embedding = mpnn_embeddings.from_npz(str(output_path))

        assert loaded_embedding.name == embedding.name


def test_save_and_load_preserves_embeddings():
    """Test that save and load preserves the embeddings array."""
    embedding = create_test_embedding()

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_embedding.npz"
        embedding.save(str(output_path))

        loaded_embedding = mpnn_embeddings.from_npz(str(output_path))

        np.testing.assert_array_equal(
            loaded_embedding.embeddings, embedding.embeddings
        )


def test_save_and_load_preserves_idxs():
    """Test that save and load preserves the idxs list."""
    embedding = create_test_embedding()

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_embedding.npz"
        embedding.save(str(output_path))

        loaded_embedding = mpnn_embeddings.from_npz(str(output_path))

        assert loaded_embedding.idxs == embedding.idxs


def test_round_trip_with_different_idxs_formats():
    """Test save/load with different idx formats (strings, numbers)."""
    embedding = mpnn_embeddings.MPNNEmbeddings(
        name="mixed_idxs",
        embeddings=np.random.rand(5, 64),
        idxs=["1", "2A", "3", "4B", "5"],
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


def test_save_and_load_preserves_sequence():
    """Test that save and load preserves the sequence field."""
    embedding = create_test_embedding(include_sequence=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_embedding.npz"
        embedding.save(str(output_path))

        loaded_embedding = mpnn_embeddings.from_npz(str(output_path))

        assert loaded_embedding.sequence == embedding.sequence
        assert loaded_embedding.sequence == "ACDEFGHIKL"


def test_save_and_load_without_sequence():
    """Test that save/load works when sequence is None."""
    embedding = create_test_embedding(include_sequence=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_embedding.npz"
        embedding.save(str(output_path))

        loaded_embedding = mpnn_embeddings.from_npz(str(output_path))

        assert loaded_embedding.sequence is None


def test_get_inputs_sequence_matches_seqio():
    """Verify that _get_inputs sequence matches BioPython SeqIO pdb-atom.

    Note: _get_inputs only includes residues with complete backbone atoms
    (N, CA, C), while SeqIO includes all residues (with X for those missing
    backbone atoms). So we compare after removing X residues from SeqIO output.
    """
    test_pdb = Path(__file__).parent / "data" / "12e8_imgt.pdb"
    chain = "H"

    # Get sequence from _get_inputs
    inputs = _get_inputs(str(test_pdb), chain=chain)
    mpnn_sequence = inputs.sequence

    # Get sequence from BioPython SeqIO pdb-atom
    seqio_sequence = None
    for record in SeqIO.parse(str(test_pdb), "pdb-atom"):
        if record.id.endswith(chain):
            seqio_sequence = str(record.seq)
            break

    assert seqio_sequence is not None, f"Chain {chain} not found via SeqIO"

    # Remove X residues from SeqIO sequence (these are residues missing
    # backbone atoms, which _get_inputs skips)
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


def test_get_inputs_parses_cif_file():
    """Test that _get_inputs correctly parses CIF files."""
    cif_file = Path(__file__).parent / "data" / "test_minimal.cif"

    inputs = _get_inputs(str(cif_file), chain="A")

    assert isinstance(inputs, MPNNInputs)
    assert inputs.coords.shape[1] == 2  # 2 residues
    assert inputs.coords.shape[2] == 4  # N, CA, C, CB
    assert inputs.coords.shape[3] == 3  # x, y, z
    assert len(inputs.residue_ids) == 2
    assert inputs.sequence == "AG"


def test_get_inputs_raises_on_missing_chain():
    """Test that requesting a non-existent chain raises ValueError."""
    test_pdb = Path(__file__).parent / "data" / "12e8_imgt.pdb"

    with pytest.raises(
        ValueError, match="Chain 'Z' not found.*Available chains"
    ):
        _get_inputs(str(test_pdb), chain="Z")


def test_get_inputs_handles_insertion_codes():
    """Test that residues with insertion codes are correctly represented."""
    pdb_file = Path(__file__).parent / "data" / "test_insertion_codes.pdb"

    inputs = _get_inputs(str(pdb_file), chain="A")

    assert len(inputs.residue_ids) == 4
    assert inputs.residue_ids == ["52", "52A", "52B", "53"]
    assert inputs.sequence == "AGST"


def test_get_inputs_raises_on_empty_chain():
    """Test that a chain with no valid residues raises ValueError."""
    pdb_file = Path(__file__).parent / "data" / "test_no_backbone.pdb"

    with pytest.raises(ValueError, match="No valid residues found"):
        _get_inputs(str(pdb_file), chain="A")


class TestGapIndices:
    """Tests for gap_indices field in MPNNEmbeddings."""

    def test_gap_indices_default_is_none(self):
        """Test that gap_indices defaults to None."""
        embedding = np.random.rand(3, constants.EMBED_DIM)
        idxs = ["1", "2", "3"]

        mp = mpnn_embeddings.MPNNEmbeddings(
            name="test", embeddings=embedding, idxs=idxs
        )

        assert mp.gap_indices is None

    def test_gap_indices_accepts_frozenset(self):
        """Test that gap_indices accepts a FrozenSet."""
        embedding = np.random.rand(5, constants.EMBED_DIM)
        idxs = ["1", "2", "3", "4", "5"]
        gap_indices = frozenset({1, 3})

        mp = mpnn_embeddings.MPNNEmbeddings(
            name="test",
            embeddings=embedding,
            idxs=idxs,
            gap_indices=gap_indices,
        )

        assert mp.gap_indices == frozenset({1, 3})
        assert 1 in mp.gap_indices
        assert 3 in mp.gap_indices
        assert 2 not in mp.gap_indices

    def test_gap_indices_empty_frozenset(self):
        """Test that gap_indices accepts an empty FrozenSet."""
        embedding = np.random.rand(3, constants.EMBED_DIM)
        idxs = ["1", "2", "3"]
        gap_indices = frozenset()

        mp = mpnn_embeddings.MPNNEmbeddings(
            name="test",
            embeddings=embedding,
            idxs=idxs,
            gap_indices=gap_indices,
        )

        assert mp.gap_indices == frozenset()
        assert len(mp.gap_indices) == 0

    def test_gap_indices_is_immutable(self):
        """Test that gap_indices is immutable (FrozenSet)."""
        embedding = np.random.rand(3, constants.EMBED_DIM)
        idxs = ["1", "2", "3"]
        gap_indices = frozenset({0})

        mp = mpnn_embeddings.MPNNEmbeddings(
            name="test",
            embeddings=embedding,
            idxs=idxs,
            gap_indices=gap_indices,
        )

        # FrozenSet should not have add/remove methods that work
        assert isinstance(mp.gap_indices, frozenset)
        with pytest.raises(AttributeError):
            mp.gap_indices.add(1)


class TestComputeGapIndices:
    """Tests for _compute_gap_indices helper function."""

    def test_returns_none_for_single_residue(self):
        """Test that single residue returns None (no gaps possible)."""
        coords = np.zeros((1, 1, 4, 3))

        result = mpnn_embeddings._compute_gap_indices(coords)

        assert result is None

    def test_returns_frozenset_for_multiple_residues(self):
        """Test that multiple residues return a FrozenSet."""
        coords = np.zeros((1, 3, 4, 3))
        # Set up normal peptide bonds
        coords[0, 0, 2, :] = [0, 0, 0]  # C of residue 0
        coords[0, 1, 0, :] = [1.3, 0, 0]  # N of residue 1
        coords[0, 1, 2, :] = [2.6, 0, 0]  # C of residue 1
        coords[0, 2, 0, :] = [3.9, 0, 0]  # N of residue 2

        result = mpnn_embeddings._compute_gap_indices(coords)

        assert isinstance(result, frozenset)
        assert len(result) == 0  # No gaps

    def test_detects_gap_in_coordinates(self):
        """Test that gap is detected from coordinates."""
        coords = np.zeros((1, 3, 4, 3))
        # Set up a gap between residues 0 and 1
        coords[0, 0, 2, :] = [0, 0, 0]  # C of residue 0
        coords[0, 1, 0, :] = [10, 0, 0]  # N of residue 1 (far away = gap)
        coords[0, 1, 2, :] = [11.3, 0, 0]  # C of residue 1
        coords[0, 2, 0, :] = [12.6, 0, 0]  # N of residue 2

        result = mpnn_embeddings._compute_gap_indices(coords)

        assert 0 in result  # Gap after residue 0

    def test_filters_by_keep_indices(self):
        """Test that keep_indices filters the coordinates."""
        coords = np.zeros((1, 5, 4, 3))
        # Set up coordinates with a gap at index 2 (between original 2 and 3)
        for i in range(5):
            coords[0, i, 0, :] = [i * 3.8, 0, 0]  # N
            coords[0, i, 2, :] = [i * 3.8 + 2.5, 0, 0]  # C
        # Create gap between residue 2 and 3
        coords[0, 3, 0, :] = [20, 0, 0]

        # Without filtering - gap at index 2
        result_all = mpnn_embeddings._compute_gap_indices(coords)
        assert 2 in result_all

        # Filter to only keep [0, 1, 2] - no gap in this subset
        result_filtered = mpnn_embeddings._compute_gap_indices(
            coords, keep_indices=[0, 1, 2]
        )
        assert result_filtered is not None
        assert 2 not in result_filtered

    def test_handles_3d_coords(self):
        """Test that 3D coords (without batch dim) work correctly."""
        coords = np.zeros((3, 4, 3))
        coords[0, 2, :] = [0, 0, 0]  # C of residue 0
        coords[1, 0, :] = [5, 0, 0]  # N of residue 1 (gap)
        coords[1, 2, :] = [6.3, 0, 0]  # C of residue 1
        coords[2, 0, :] = [7.6, 0, 0]  # N of residue 2

        result = mpnn_embeddings._compute_gap_indices(coords)

        assert isinstance(result, frozenset)
        assert 0 in result
