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
from sabr.renumber import RenumberResult
from sabr.types import ChainType
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
            assert emb.embeddings.ndim == 2, (
                f"noise={noise_level}, chain={chain_type}: expected 2D array"
            )
            assert emb.embeddings.shape[1] == 64, (
                f"noise={noise_level}, chain={chain_type}: expected 64-dim embeddings"
            )

    @pytest.mark.parametrize("noise_level", NOISE_LEVELS)
    def test_idxs_match_embedding_rows(self, noise_level):
        """Number of idxs must equal the number of embedding rows."""
        embeddings_name = f"embeddings_noise_{noise_level}.npz"
        aligner = SoftAligner(embeddings_name=embeddings_name)
        for chain_type, emb in aligner.embeddings.items():
            assert len(emb.positions) == emb.embeddings.shape[0], (
                f"noise={noise_level}, chain={chain_type}: "
                f"position count {len(emb.positions)} != rows {emb.embeddings.shape[0]}"
            )

    @pytest.mark.parametrize("noise_level", NOISE_LEVELS)
    def test_idxs_are_valid_imgt_positions(self, noise_level):
        """All idx values should be valid IMGT positions (1–128)."""
        embeddings_name = f"embeddings_noise_{noise_level}.npz"
        aligner = SoftAligner(embeddings_name=embeddings_name)
        for chain_type, emb in aligner.embeddings.items():
            for pos in emb.positions:
                assert 1 <= pos <= 128, (
                    f"noise={noise_level}, chain={chain_type}: "
                    f"position {pos} outside IMGT range 1-128"
                )


@pytest.mark.slow
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
        assert (row_sums <= 1).all(), (
            f"noise={noise_level}: some rows have more than one assignment"
        )

    @pytest.mark.parametrize("noise_level", NOISE_LEVELS)
    def test_chain_type_detected(self, noise_level):
        """A valid chain type should be selected from the embedding label."""
        fixture = FIXTURES["8_21"]
        if not fixture["pdb"].exists():
            pytest.skip(f"Missing fixture at {fixture['pdb']}")

        input_data = from_pdb(str(fixture["pdb"]), fixture["chain"])
        embeddings_name = f"embeddings_noise_{noise_level}.npz"
        aligner = SoftAligner(embeddings_name=embeddings_name)
        result = aligner(input_data)

        assert result.selected_chain_type in (
            ChainType.HEAVY,
            ChainType.KAPPA,
            ChainType.LAMBDA,
        ), f"noise={noise_level}: unexpected chain_type={result.selected_chain_type}"

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

        assert np.isfinite(result.score), (
            f"noise={noise_level}: score is not finite: {result.score}"
        )


class TestNoiseLevelCLI:
    """Tests for the --noise-level CLI argument."""

    def _patch_renumber_file(self, monkeypatch):
        def fake_renumber_file(
            input_path, chain_id, output_path, options, reference_embeddings_name
        ):
            output_path.write_text("RENUMBERED\n")
            return RenumberResult(
                output_path=output_path,
                chain_type=ChainType.HEAVY,
                residue_count=1,
                changed_residue_count=1,
            )

        monkeypatch.setattr("sabr.cli.main.renumber_file", fake_renumber_file)

    def test_version_option_runs(self):
        runner = CliRunner()
        result = runner.invoke(cli_main, ["--version"])

        assert result.exit_code == 0
        assert "sabr" in result.output.lower()

    def test_noise_level_default_omitted(self, monkeypatch, tmp_path):
        """Running without --noise-level should succeed."""
        self._patch_renumber_file(monkeypatch)
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

    def test_chain_type_option_sets_embedding_label(self, monkeypatch, tmp_path):
        captured = {}

        def fake_renumber_file(
            input_path, chain_id, output_path, options, reference_embeddings_name
        ):
            output_path.write_text("RENUMBERED\n")
            captured["chain_type"] = options.chain_type
            captured["reference_embeddings_name"] = reference_embeddings_name
            return RenumberResult(
                output_path=output_path,
                chain_type=options.chain_type,
                residue_count=1,
                changed_residue_count=1,
            )

        monkeypatch.setattr("sabr.cli.main.renumber_file", fake_renumber_file)
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
                "--chain-type",
                "H",
                "--overwrite",
            ],
        )

        assert result.exit_code == 0, result.output
        assert captured["chain_type"] is ChainType.HEAVY
        assert captured["reference_embeddings_name"] == "embeddings.npz"

    def test_random_seed_defaults_to_zero(self, monkeypatch, tmp_path):
        captured = {}

        def fake_renumber_file(
            input_path, chain_id, output_path, options, reference_embeddings_name
        ):
            del input_path, chain_id, reference_embeddings_name
            output_path.write_text("RENUMBERED\n")
            captured["random_seed"] = options.random_seed
            return RenumberResult(
                output_path=output_path,
                chain_type=ChainType.HEAVY,
                residue_count=1,
                changed_residue_count=1,
            )

        monkeypatch.setattr("sabr.cli.main.renumber_file", fake_renumber_file)
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

        assert result.exit_code == 0, result.output
        assert captured["random_seed"] == 0

    def test_cli_rejects_non_label_chain_type_aliases(self, tmp_path):
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
                "--chain-type",
                "heavy",
            ],
        )

        assert result.exit_code != 0
        assert "Invalid value for '-t' / '--chain-type'" in result.output

    @pytest.mark.parametrize("noise_level", NOISE_LEVELS)
    def test_noise_level_option_runs(self, monkeypatch, noise_level, tmp_path):
        """--noise-level should be accepted and produce output."""
        self._patch_renumber_file(monkeypatch)
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
        assert result.exit_code == 0, (
            f"CLI failed for noise_level={noise_level}: {result.output}"
        )
        assert output.exists(), f"Output file not created for noise_level={noise_level}"

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

    def test_reference_chain_type_option_removed(self, tmp_path):
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
                "--reference-chain-type",
                "H",
            ],
        )

        assert result.exit_code != 0
        assert "No such option" in result.output

    def test_reference_chain_type_absent_from_help(self):
        runner = CliRunner()
        result = runner.invoke(cli_main, ["--help"])

        assert result.exit_code == 0
        assert "--reference-chain-type" not in result.output
        assert "--chain-type [h|k|l|auto]" in result.output
        assert "Chain type embedding label to use" in result.output
