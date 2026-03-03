"""Tests for OAS MPNN noise-level reference embeddings.

Verifies that each noise-level NPZ file (0.0, 0.2, 0.5, 1.0, 2.0) loads
correctly and produces valid alignments through the SoftAligner pipeline.
"""

import numpy as np
import pytest
from click.testing import CliRunner

from sabr.alignment.soft_aligner import SoftAligner
from sabr.cli.main import main as cli_main
from sabr.embeddings.mpnn import from_pdb
from tests.conftest import FIXTURES

NOISE_LEVELS = ["0.0", "0.2", "0.5", "1.0", "2.0"]


class TestNoiseEmbeddingsLoad:
    """Tests that each noise-level NPZ file loads with the expected structure.

    Covers loading, shape, and IMGT index validity for all noise levels.
    """

    @pytest.mark.parametrize("noise_level", NOISE_LEVELS)
    def test_loads_all_chain_types(self, noise_level):
        """Each noise-level file should provide H, K, L chain embeddings."""
        embeddings_name = f"embeddings_noise_{noise_level}.npz"
        aligner = SoftAligner(embeddings_name=embeddings_name)
        assert set(aligner.embeddings.keys()) == {"H", "K", "L"}

    @pytest.mark.parametrize("noise_level", NOISE_LEVELS)
    def test_embedding_shapes(self, noise_level):
        """Each chain embedding should have shape (N, 64)."""
        embeddings_name = f"embeddings_noise_{noise_level}.npz"
        aligner = SoftAligner(embeddings_name=embeddings_name)
        for chain_type, emb in aligner.embeddings.items():
            assert (
                emb.embeddings.ndim == 2
            ), f"noise={noise_level}, chain={chain_type}: expected 2D array"
            assert emb.embeddings.shape[1] == 64, (
                f"noise={noise_level}, chain={chain_type}: "
                "expected 64-dim embeddings"
            )

    @pytest.mark.parametrize("noise_level", NOISE_LEVELS)
    def test_idxs_match_embedding_rows(self, noise_level):
        """Number of idxs must equal the number of embedding rows."""
        embeddings_name = f"embeddings_noise_{noise_level}.npz"
        aligner = SoftAligner(embeddings_name=embeddings_name)
        for chain_type, emb in aligner.embeddings.items():
            assert len(emb.idxs) == emb.embeddings.shape[0], (
                f"noise={noise_level}, chain={chain_type}: "
                f"idxs length {len(emb.idxs)} != rows {emb.embeddings.shape[0]}"
            )

    @pytest.mark.parametrize("noise_level", NOISE_LEVELS)
    def test_idxs_are_valid_imgt_positions(self, noise_level):
        """All idx values should be valid IMGT positions (1–128)."""
        embeddings_name = f"embeddings_noise_{noise_level}.npz"
        aligner = SoftAligner(embeddings_name=embeddings_name)
        for chain_type, emb in aligner.embeddings.items():
            for idx in emb.idxs:
                pos = int(idx)
                assert 1 <= pos <= 128, (
                    f"noise={noise_level}, chain={chain_type}: "
                    f"idx {idx} outside IMGT range 1-128"
                )


class TestNoiseEmbeddingsAlignment:
    """Tests that noise-level embeddings produce valid alignments on real PDBs.

    Uses the 8_21 fixture heavy chain as a representative input.
    """

    @pytest.mark.parametrize("noise_level", NOISE_LEVELS)
    def test_alignment_shape(self, noise_level):
        """Alignment output should have shape (n_query, 128)."""
        fixture = FIXTURES["8_21"]
        if not fixture["pdb"].exists():
            pytest.skip(f"Missing fixture at {fixture['pdb']}")

        input_data = from_pdb(str(fixture["pdb"]), fixture["chain"])
        embeddings_name = f"embeddings_noise_{noise_level}.npz"
        aligner = SoftAligner(embeddings_name=embeddings_name)
        result = aligner(input_data)

        n_query = input_data.embeddings.shape[0]
        assert result.alignment.shape == (n_query, 128), (
            f"noise={noise_level}: expected ({n_query}, 128), "
            f"got {result.alignment.shape}"
        )

    @pytest.mark.parametrize("noise_level", NOISE_LEVELS)
    def test_alignment_is_binary(self, noise_level):
        """Alignment matrix values should only be 0 or 1."""
        fixture = FIXTURES["8_21"]
        if not fixture["pdb"].exists():
            pytest.skip(f"Missing fixture at {fixture['pdb']}")

        input_data = from_pdb(str(fixture["pdb"]), fixture["chain"])
        embeddings_name = f"embeddings_noise_{noise_level}.npz"
        aligner = SoftAligner(embeddings_name=embeddings_name)
        result = aligner(input_data)

        unique_vals = set(np.unique(result.alignment))
        assert unique_vals <= {
            0,
            1,
        }, f"noise={noise_level}: non-binary values in alignment: {unique_vals}"

    @pytest.mark.parametrize("noise_level", NOISE_LEVELS)
    def test_alignment_each_row_at_most_one(self, noise_level):
        """Each query residue should align to at most one reference position."""
        fixture = FIXTURES["8_21"]
        if not fixture["pdb"].exists():
            pytest.skip(f"Missing fixture at {fixture['pdb']}")

        input_data = from_pdb(str(fixture["pdb"]), fixture["chain"])
        embeddings_name = f"embeddings_noise_{noise_level}.npz"
        aligner = SoftAligner(embeddings_name=embeddings_name)
        result = aligner(input_data)

        row_sums = result.alignment.sum(axis=1)
        assert (
            row_sums <= 1
        ).all(), f"noise={noise_level}: some rows have more than one assignment"

    @pytest.mark.parametrize("noise_level", NOISE_LEVELS)
    def test_chain_type_detected(self, noise_level):
        """A valid chain type should be detected from the alignment."""
        fixture = FIXTURES["8_21"]
        if not fixture["pdb"].exists():
            pytest.skip(f"Missing fixture at {fixture['pdb']}")

        input_data = from_pdb(str(fixture["pdb"]), fixture["chain"])
        embeddings_name = f"embeddings_noise_{noise_level}.npz"
        aligner = SoftAligner(embeddings_name=embeddings_name)
        result = aligner(input_data)

        assert result.chain_type in (
            "H",
            "K",
            "L",
        ), f"noise={noise_level}: unexpected chain_type={result.chain_type}"

    @pytest.mark.parametrize("noise_level", NOISE_LEVELS)
    def test_score_is_finite(self, noise_level):
        """Alignment score should be a finite number."""
        fixture = FIXTURES["8_21"]
        if not fixture["pdb"].exists():
            pytest.skip(f"Missing fixture at {fixture['pdb']}")

        input_data = from_pdb(str(fixture["pdb"]), fixture["chain"])
        embeddings_name = f"embeddings_noise_{noise_level}.npz"
        aligner = SoftAligner(embeddings_name=embeddings_name)
        result = aligner(input_data)

        assert np.isfinite(
            result.score
        ), f"noise={noise_level}: score is not finite: {result.score}"


class TestNoiseLevelCLI:
    """Tests for the --noise-level CLI argument."""

    def test_noise_level_default_omitted(self, tmp_path):
        """Running without --noise-level should succeed."""
        fixture = FIXTURES["8_21"]
        if not fixture["pdb"].exists():
            pytest.skip(f"Missing fixture at {fixture['pdb']}")

        output = tmp_path / "out.pdb"
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                "-i",
                str(fixture["pdb"]),
                "-c",
                fixture["chain"],
                "-o",
                str(output),
                "--overwrite",
            ],
        )
        assert result.exit_code == 0, f"CLI failed: {result.output}"

    @pytest.mark.parametrize("noise_level", NOISE_LEVELS)
    def test_noise_level_option_runs(self, noise_level, tmp_path):
        """--noise-level should be accepted and produce output."""
        fixture = FIXTURES["8_21"]
        if not fixture["pdb"].exists():
            pytest.skip(f"Missing fixture at {fixture['pdb']}")

        output = tmp_path / f"out_noise_{noise_level}.pdb"
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                "-i",
                str(fixture["pdb"]),
                "-c",
                fixture["chain"],
                "-o",
                str(output),
                "--noise-level",
                noise_level,
                "--overwrite",
            ],
        )
        assert (
            result.exit_code == 0
        ), f"CLI failed for noise_level={noise_level}: {result.output}"
        assert (
            output.exists()
        ), f"Output file not created for noise_level={noise_level}"

    def test_invalid_noise_level_rejected(self, tmp_path):
        """An invalid noise level should cause a non-zero exit code."""
        fixture = FIXTURES["8_21"]
        if not fixture["pdb"].exists():
            pytest.skip(f"Missing fixture at {fixture['pdb']}")

        output = tmp_path / "out.pdb"
        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                "-i",
                str(fixture["pdb"]),
                "-c",
                fixture["chain"],
                "-o",
                str(output),
                "--noise-level",
                "0.9",
            ],
        )
        assert result.exit_code != 0
