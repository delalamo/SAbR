from importlib import resources
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest
from ANARCI import anarci
from Bio import PDB
from click.testing import CliRunner

from sabr import aln2hmm, cli, edit_pdb, util

DATA_PACKAGE = "tests.data"


def resolve_data_path(filename: str) -> Path:
    return Path(resources.files(DATA_PACKAGE) / filename)


FIXTURES = {
    "8_21": {
        "pdb": resolve_data_path("8_21_renumbered.pdb"),
        "chain": "A",
        "alignment": resolve_data_path("8_21_renumbered_alignment.npz"),
        "min_deviations": 0,
        "max_deviations": 0,
    },
    "5omm": {
        "pdb": resolve_data_path("5omm_imgt.pdb"),
        "chain": "C",
        "alignment": resolve_data_path("5omm_imgt_alignment.npz"),
        "min_deviations": 5,
        "max_deviations": 200,
    },
}


def load_alignment_fixture(path: Path):
    if not path.exists():
        pytest.skip(f"Missing alignment fixture at {path}")
    data = np.load(path, allow_pickle=True)
    alignment = data["alignment"]
    species = data["species"].item()
    return alignment, species


def run_threading_pipeline(
    pdb_path: Path, chain: str, alignment, species: str, tmp_path
):
    sequence = util.fetch_sequence_from_pdb(str(pdb_path), chain)
    sv, start, end = aln2hmm.alignment_matrix_to_state_vector(alignment)
    subsequence = "-" * start + sequence[start:end]
    anarci_out, anarci_start, anarci_end = (
        anarci.number_sequence_from_alignment(
            sv,
            subsequence,
            scheme="imgt",
            chain_type=species,
        )
    )
    output_pdb = tmp_path / f"{pdb_path.stem}_{chain}_threaded.pdb"
    deviations = edit_pdb.thread_alignment(
        str(pdb_path),
        chain,
        anarci_out,
        str(output_pdb),
        anarci_start,
        anarci_end,
        alignment_start=start,
    )
    return deviations


@pytest.mark.parametrize("fixture_key", ["8_21", "5omm"])
def test_thread_alignment_has_zero_deviations(tmp_path, fixture_key):
    data = FIXTURES[fixture_key]
    if not data["pdb"].exists():
        pytest.skip(f"Missing structure fixture at {data['pdb']}")
    alignment, species = load_alignment_fixture(data["alignment"])
    deviations = run_threading_pipeline(
        data["pdb"],
        data["chain"],
        alignment,
        species,
        tmp_path,
    )
    min_expected = data.get("min_deviations")
    max_expected = data.get("max_deviations")
    assert min_expected is not None and max_expected is not None
    assert min_expected <= deviations <= max_expected


def extract_residue_ids(
    pdb_path: Path, chain: str
) -> List[Tuple[str, int, str]]:
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("structure", pdb_path)
    residues = []
    for res in structure[0][chain]:
        hetflag, resseq, icode = res.get_id()
        if hetflag.strip():
            continue
        residues.append((hetflag, resseq, icode))
    return residues


@pytest.mark.parametrize(
    ("fixture_key", "expect_same"),
    [
        ("8_21", True),
        ("5omm", False),
    ],
)
def test_cli_respects_expected_numbering(
    monkeypatch, tmp_path, fixture_key, expect_same
):
    data = FIXTURES[fixture_key]
    if not data["pdb"].exists():
        pytest.skip(f"Missing structure fixture at {data['pdb']}")
    alignment, species = load_alignment_fixture(data["alignment"])

    class DummyResult:
        def __init__(self, alignment, species):
            self.alignment = alignment
            self.species = species

    class DummyAligner:
        def __call__(self, input_pdb, input_chain, **kwargs):
            return DummyResult(alignment, species)

    monkeypatch.setattr(cli.softaligner, "SoftAligner", lambda: DummyAligner())

    runner = CliRunner()
    output_pdb = tmp_path / f"{fixture_key}_cli.pdb"
    result = runner.invoke(
        cli.main,
        [
            "-i",
            str(data["pdb"]),
            "-c",
            data["chain"],
            "-o",
            str(output_pdb),
            "--overwrite",
        ],
    )
    assert result.exit_code == 0, result.output

    original_ids = extract_residue_ids(data["pdb"], data["chain"])
    threaded_ids = extract_residue_ids(output_pdb, data["chain"])
    assert (original_ids == threaded_ids) is expect_same
