import subprocess
import sys


def _run_without_jax_import(code: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_public_import_does_not_import_jax():
    _run_without_jax_import(
        "import sabr, sys; "
        "assert 'jax' not in sys.modules; "
        "assert hasattr(sabr, 'renumber_file')"
    )


def test_cli_help_does_not_import_jax():
    _run_without_jax_import(
        "import sys; "
        "from click.testing import CliRunner; "
        "from sabr.cli.main import main; "
        "result = CliRunner().invoke(main, ['--help']); "
        "assert result.exit_code == 0, result.output; "
        "assert 'jax' not in sys.modules"
    )


def test_cli_version_does_not_import_jax():
    _run_without_jax_import(
        "import sys; "
        "from click.testing import CliRunner; "
        "from sabr.cli.main import main; "
        "result = CliRunner().invoke(main, ['--version']); "
        "assert result.exit_code == 0, result.output; "
        "assert 'jax' not in sys.modules"
    )
