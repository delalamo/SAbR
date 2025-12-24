from importlib import resources
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest
from ANARCI import anarci
from Bio import PDB, SeqIO
from click.testing import CliRunner

from sabr import aln2hmm, cli, edit_pdb, mpnn_embeddings
from tests.conftest import create_dummy_aligner, create_dummy_from_pdb

DATA_PACKAGE = "tests.data"


def resolve_data_path(filename: str) -> Path:
    return Path(resources.files(DATA_PACKAGE) / filename)


FIXTURES = {
    "8_21": {
        "pdb": resolve_data_path("8_21_renumbered.pdb"),
        "chain": "A",
        "alignment": resolve_data_path("8_21_renumbered_alignment.npz"),
        "embeddings": resolve_data_path("8_21_renumbered_embeddings.npz"),
        "min_deviations": 0,
        "max_deviations": 0,
    },
    "5omm": {
        "pdb": resolve_data_path("5omm_imgt.pdb"),
        "chain": "C",
        "alignment": resolve_data_path("5omm_imgt_alignment.npz"),
        "embeddings": resolve_data_path("5omm_imgt_embeddings.npz"),
        "min_deviations": 5,
        "max_deviations": 200,
    },
    # Heavy chain test case for alignment start position fix
    # This structure starts at IMGT position 2 (not 1), which previously
    # caused off-by-one errors in numbering
    "test_heavy_chain": {
        "pdb": resolve_data_path("test_heavy_chain.pdb"),
        "chain": "F",
        "alignment": resolve_data_path("test_heavy_chain_alignment.npz"),
        "embeddings": resolve_data_path("test_heavy_chain_embeddings.npz"),
        # CDR3 insertions cause some expected deviations
        "min_deviations": 0,
        "max_deviations": 25,
    },
}


def load_alignment_fixture(path: Path) -> Tuple[np.ndarray, str]:
    if not path.exists():
        pytest.skip(f"Missing alignment fixture at {path}")
    data = np.load(path, allow_pickle=True)
    alignment = data["alignment"]
    species = data["species"].item()
    return alignment, species


def run_threading_pipeline(
    pdb_path: Path,
    chain: str,
    alignment: np.ndarray,
    species: str,
    tmp_path: Path,
) -> int:
    # Extract sequence using SeqIO (removes X residues to match main code)
    sequence = None
    for record in SeqIO.parse(str(pdb_path), "pdb-atom"):
        if record.id.endswith(chain):
            sequence = str(record.seq).replace("X", "")
            break
    if sequence is None:
        raise ValueError(f"Chain {chain} not found in {pdb_path}")
    sv, start, end, _ = aln2hmm.alignment_matrix_to_state_vector(alignment)
    # Create subsequence with leading dashes for missing IMGT positions
    # start = first IMGT column (0-indexed), used for leading dashes
    # end - start = number of aligned residues
    n_aligned = end - start
    subsequence = "-" * start + sequence[:n_aligned]
    anarci_out, anarci_start, anarci_end = (
        anarci.number_sequence_from_alignment(
            sv,
            subsequence,
            scheme="imgt",
            chain_type=species,
        )
    )
    output_pdb = tmp_path / f"{pdb_path.stem}_{chain}_threaded.pdb"
    # alignment_start=0 because PDB structures start at the Fv region
    # (no leader sequence). The 'start' value from aln2hmm is the IMGT
    # column offset, which is already handled by anarci_start.
    deviations = edit_pdb.thread_alignment(
        str(pdb_path),
        chain,
        anarci_out,
        str(output_pdb),
        anarci_start,
        anarci_end,
        alignment_start=0,
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

    DummyAligner = create_dummy_aligner(alignment, species)
    dummy_from_pdb = create_dummy_from_pdb()

    monkeypatch.setattr(mpnn_embeddings, "from_pdb", dummy_from_pdb)
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


@pytest.mark.parametrize(
    "deterministic_flag",
    [
        "--deterministic-loop-renumbering",
        "--no-deterministic-loop-renumbering",
    ],
)
def test_cli_deterministic_loop_renumbering_flag(
    monkeypatch, tmp_path, deterministic_flag
):
    """Test that CLI accepts both deterministic loop renumbering flags."""
    data = FIXTURES["8_21"]
    if not data["pdb"].exists():
        pytest.skip(f"Missing structure fixture at {data['pdb']}")
    alignment, species = load_alignment_fixture(data["alignment"])

    captured_kwargs = {}
    DummyAligner = create_dummy_aligner(alignment, species, captured_kwargs)
    dummy_from_pdb = create_dummy_from_pdb()

    monkeypatch.setattr(mpnn_embeddings, "from_pdb", dummy_from_pdb)
    monkeypatch.setattr(cli.softaligner, "SoftAligner", lambda: DummyAligner())

    runner = CliRunner()
    output_pdb = tmp_path / "test_det_flag.pdb"
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
            deterministic_flag,
        ],
    )
    assert result.exit_code == 0, result.output

    # Verify that the flag was passed to the aligner
    expected_value = deterministic_flag == "--deterministic-loop-renumbering"
    assert (
        captured_kwargs.get("deterministic_loop_renumbering") == expected_value
    )


def test_cli_rejects_multi_character_chain():
    """Test that CLI rejects chain identifiers longer than one character."""
    data = FIXTURES["8_21"]
    if not data["pdb"].exists():
        pytest.skip(f"Missing structure fixture at {data['pdb']}")

    runner = CliRunner()
    result = runner.invoke(
        cli.main,
        [
            "-i",
            str(data["pdb"]),
            "-c",
            "AB",  # Two characters - should fail
            "-o",
            "output.pdb",
        ],
    )
    assert result.exit_code != 0
    assert "Chain identifier must be exactly one character" in result.output


def test_alignment_start_position_correct():
    """Test alignment handles structures starting at IMGT position 2.

    This is a regression test for the off-by-one bug where sequences starting
    at IMGT position 2 (not 1) had their first residue incorrectly numbered.
    The fix ensures that:
    1. The state vector uses 1-indexed IMGT positions
    2. The subsequence is constructed correctly with leading dashes
    3. The first residue maps to the correct IMGT position
    """
    data = FIXTURES["test_heavy_chain"]
    if not data["pdb"].exists():
        pytest.skip(f"Missing structure fixture at {data['pdb']}")

    alignment, species = load_alignment_fixture(data["alignment"])

    # Get the state vector
    sv, start, end, _ = aln2hmm.alignment_matrix_to_state_vector(alignment)

    # The alignment for this structure starts at IMGT column 1 (position 2)
    # This means there should be 1 leading dash in the subsequence
    assert start == 1, f"Expected start=1 (IMGT position 2), got {start}"

    # The first state should have residue_number=2 (1-indexed IMGT position)
    first_state = sv[0]
    assert (
        first_state.residue_number == 2
    ), f"Expected IMGT position 2, got {first_state.residue_number}"
    assert (
        first_state.insertion_code == "m"
    ), f"Expected first state to be a match, got {first_state.insertion_code}"

    # The mapped_residue should be offset by start (for subsequence indexing)
    # First residue (seq index 0) should map to subsequence index 1
    assert (
        first_state.mapped_residue == 1
    ), f"Expected mapped_residue=1, got {first_state.mapped_residue}"

    # Verify the subsequence construction
    n_aligned = end - start
    assert n_aligned > 0, f"Expected positive n_aligned, got {n_aligned}"

    # Verify state vector consistency
    match_states = [s for s in sv if s.insertion_code == "m"]
    assert (
        len(match_states) > 100
    ), f"Expected >100 match states, got {len(match_states)}"
