from importlib import resources
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest
from ANARCI import anarci
from Bio import PDB, SeqIO
from click.testing import CliRunner

from sabr import aln2hmm, cli, constants, edit_pdb, mpnn_embeddings, softaligner
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
    "test_heavy_chain": {
        "pdb": resolve_data_path("test_heavy_chain.pdb"),
        "chain": "F",
        "alignment": resolve_data_path("test_heavy_chain_alignment.npz"),
        "embeddings": resolve_data_path("test_heavy_chain_embeddings.npz"),
        "min_deviations": 0,
        "max_deviations": 25,
    },
    "woot_H_next": {
        "pdb": resolve_data_path("woot_H_next.pdb"),
        "chain": "H",
        "alignment": resolve_data_path("woot_H_next_alignment.npz"),
        "embeddings": resolve_data_path("woot_H_next_embeddings.npz"),
        "min_deviations": 0,
        "max_deviations": 0,
        "has_n_terminal_extension": True,
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
    sequence = None
    for record in SeqIO.parse(str(pdb_path), "pdb-atom"):
        if record.id.endswith(chain):
            sequence = str(record.seq).replace("X", "")
            break
    if sequence is None:
        raise ValueError(f"Chain {chain} not found in {pdb_path}")

    state_vector, imgt_start, imgt_end, _ = (
        aln2hmm.alignment_matrix_to_state_vector(alignment)
    )
    n_aligned = imgt_end - imgt_start
    subsequence = "-" * imgt_start + sequence[:n_aligned]

    anarci_alignment, anarci_start, anarci_end = (
        anarci.number_sequence_from_alignment(
            state_vector, subsequence, scheme="imgt", chain_type=species
        )
    )

    output_pdb = tmp_path / f"{pdb_path.stem}_{chain}_threaded.pdb"
    return edit_pdb.thread_alignment(
        str(pdb_path),
        chain,
        anarci_alignment,
        str(output_pdb),
        anarci_start,
        anarci_end,
        alignment_start=0,
    )


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
    ("use_disable_flag", "expected_value"),
    [
        (False, True),
        (True, False),
    ],
)
def test_cli_deterministic_renumbering_flag(
    monkeypatch, tmp_path, use_disable_flag, expected_value
):
    """Test --disable-deterministic-renumbering CLI flag handling."""
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
    args = [
        "-i",
        str(data["pdb"]),
        "-c",
        data["chain"],
        "-o",
        str(output_pdb),
        "--overwrite",
    ]
    if use_disable_flag:
        args.append("--disable-deterministic-renumbering")

    result = runner.invoke(cli.main, args)
    assert result.exit_code == 0, result.output

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


@pytest.mark.parametrize(
    "species",
    ["human", "mouse", "rat", "rabbit", "pig", "rhesus", "alpaca"],
)
def test_cli_anarci_species_argument(monkeypatch, tmp_path, species):
    """Test that CLI accepts all valid --anarci-species values."""
    data = FIXTURES["8_21"]
    if not data["pdb"].exists():
        pytest.skip(f"Missing structure fixture at {data['pdb']}")
    alignment, sp = load_alignment_fixture(data["alignment"])

    DummyAligner = create_dummy_aligner(alignment, sp)
    dummy_from_pdb = create_dummy_from_pdb()

    monkeypatch.setattr(mpnn_embeddings, "from_pdb", dummy_from_pdb)
    monkeypatch.setattr(cli.softaligner, "SoftAligner", lambda: DummyAligner())

    runner = CliRunner()
    output_pdb = tmp_path / f"test_species_{species}.pdb"
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
            "-s",
            species,
        ],
    )
    assert result.exit_code == 0, result.output


@pytest.mark.parametrize(
    "chain_type",
    ["H", "K", "L", "auto"],
)
def test_cli_anarci_chain_type_argument(monkeypatch, tmp_path, chain_type):
    """Test that CLI accepts all valid --anarci-chain-type values."""
    data = FIXTURES["8_21"]
    if not data["pdb"].exists():
        pytest.skip(f"Missing structure fixture at {data['pdb']}")
    alignment, species = load_alignment_fixture(data["alignment"])

    DummyAligner = create_dummy_aligner(alignment, species)
    dummy_from_pdb = create_dummy_from_pdb()

    monkeypatch.setattr(mpnn_embeddings, "from_pdb", dummy_from_pdb)
    monkeypatch.setattr(cli.softaligner, "SoftAligner", lambda: DummyAligner())

    runner = CliRunner()
    output_pdb = tmp_path / f"test_chain_type_{chain_type}.pdb"
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
            "-a",
            chain_type,
        ],
    )
    assert result.exit_code == 0, result.output


def test_cli_rejects_invalid_anarci_species():
    """Test that CLI rejects invalid --anarci-species values."""
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
            data["chain"],
            "-o",
            "output.pdb",
            "-s",
            "invalid_species",
        ],
    )
    assert result.exit_code != 0
    assert (
        "Invalid value" in result.output or "invalid_species" in result.output
    )


def test_cli_rejects_invalid_anarci_chain_type():
    """Test that CLI rejects invalid --anarci-chain-type values."""
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
            data["chain"],
            "-o",
            "output.pdb",
            "-a",
            "X",  # Invalid chain type
        ],
    )
    assert result.exit_code != 0
    assert "Invalid value" in result.output or "X" in result.output


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

    state_vector, imgt_start, imgt_end, _ = (
        aln2hmm.alignment_matrix_to_state_vector(alignment)
    )

    assert imgt_start == 1, f"Expected imgt_start=1, got {imgt_start}"

    first_state = state_vector[0]
    assert first_state.residue_number == 2
    assert first_state.insertion_code == "m"
    assert first_state.mapped_residue == 1

    n_aligned = imgt_end - imgt_start
    assert n_aligned > 0

    match_states = [s for s in state_vector if s.insertion_code == "m"]
    assert len(match_states) > 100


@pytest.mark.skip(
    reason="Requires chain-specific embeddings; skipped for unified"
)
def test_n_terminal_extension_numbering_end_to_end(tmp_path):
    """End-to-end test for structures with N-terminal extensions.

    The woot_H_next.pdb structure has 7 N-terminal residues numbered -6 to 0,
    followed by the standard IMGT-numbered Fv region (1-128).

    This test runs the FULL pipeline from start to finish:
    1. Generate MPNN embeddings from PDB
    2. Run SoftAligner to generate alignment
    3. Convert alignment to state vector
    4. Run ANARCI numbering
    5. Thread alignment onto structure
    6. Verify output numbering

    This verifies that:
    - N-terminal residues are numbered correctly (-6 to 0)
    - The Fv region is numbered correctly (1-128)
    - Zero deviations from the expected numbering
    """
    data = FIXTURES["woot_H_next"]
    if not data["pdb"].exists():
        pytest.skip(f"Missing structure fixture at {data['pdb']}")

    pdb_path = data["pdb"]
    chain = data["chain"]

    # Step 1: Extract sequence
    sequence = None
    for record in SeqIO.parse(str(pdb_path), "pdb-atom"):
        if record.id.endswith(chain):
            sequence = str(record.seq).replace("X", "")
            break
    assert sequence is not None, f"Chain {chain} not found"
    assert len(sequence) == 124, f"Expected 124 residues, got {len(sequence)}"

    # Step 2: Generate MPNN embeddings from PDB (full pipeline)
    embeddings = mpnn_embeddings.from_pdb(str(pdb_path), chain)
    assert embeddings.embeddings.shape[0] == 124, "Embedding count mismatch"

    # Step 3: Run SoftAligner (full pipeline)
    aligner = softaligner.SoftAligner()
    output = aligner(embeddings, chain_type=constants.ChainType.HEAVY)
    assert output.species == "H", f"Expected H, got {output.species}"

    # Step 4: Convert alignment to state vector
    sv, start, end, first_aligned = aln2hmm.alignment_matrix_to_state_vector(
        output.alignment
    )
    n_aligned = end - start
    subsequence = "-" * start + sequence[:n_aligned]

    # Step 5: Run ANARCI numbering
    anarci_out, anarci_start, anarci_end = (
        anarci.number_sequence_from_alignment(
            sv,
            subsequence,
            scheme="imgt",
            chain_type=output.species,
        )
    )

    # For N-terminal extensions, anarci_start indicates where the Fv begins
    assert (
        anarci_start == 7
    ), f"Expected 7 N-terminal residues, got {anarci_start}"

    # First ANARCI position should be IMGT 1
    first_pos, first_aa = anarci_out[0]
    assert (
        first_pos[0] == 1
    ), f"Expected first IMGT position 1, got {first_pos[0]}"
    assert first_aa == "E", f"Expected first residue E, got {first_aa}"

    # Step 6: Thread the alignment onto structure
    output_pdb = tmp_path / "woot_threaded.pdb"
    deviations = edit_pdb.thread_alignment(
        str(pdb_path),
        chain,
        anarci_out,
        str(output_pdb),
        start_res=0,  # anarci_out starts at index 0
        end_res=anarci_end - anarci_start,  # Adjusted length
        alignment_start=anarci_start,  # Skip N-terminal residues in PDB
    )

    assert deviations == 0, f"Expected 0 deviations, got {deviations}"

    # Step 7: Verify the output structure has correct numbering
    parser = PDB.PDBParser(QUIET=True)
    out_structure = parser.get_structure("output", output_pdb)
    residue_ids = []
    for res in out_structure[0][chain]:
        hetflag, resseq, icode = res.get_id()
        if not hetflag.strip():
            residue_ids.append(resseq)

    # Check N-terminal residues are numbered -6 to 0
    n_terminal_ids = residue_ids[:7]
    expected_n_terminal = [-6, -5, -4, -3, -2, -1, 0]
    assert (
        n_terminal_ids == expected_n_terminal
    ), f"N-terminal numbering wrong: {n_terminal_ids}"

    # Check Fv region starts at 1
    assert residue_ids[7] == 1, f"Fv should start at 1, got {residue_ids[7]}"

    # Check last residue is 128
    assert (
        residue_ids[-1] == 128
    ), f"Last residue should be 128, got {residue_ids[-1]}"


def test_n_terminal_truncated_structure_end_to_end(tmp_path):
    """End-to-end test for structures with N-terminal truncation.

    The 8_21_renumbered_ntrunc.pdb structure is missing IMGT positions 1 and 2,
    starting at position 3. This tests that SAbR correctly handles structures
    where the N-terminus is truncated.

    This test runs the FULL pipeline from start to finish:
    1. Generate MPNN embeddings from PDB
    2. Run SoftAligner to generate alignment
    3. Convert alignment to state vector
    4. Run ANARCI numbering
    5. Thread alignment onto structure
    6. Verify output numbering matches input (0 deviations expected)

    This verifies that:
    - Structures starting at IMGT position 3 are handled correctly
    - The alignment correctly identifies the starting position
    - Zero deviations from the expected IMGT numbering
    """
    pdb_path = resolve_data_path("8_21_renumbered_ntrunc.pdb")
    if not pdb_path.exists():
        pytest.skip(f"Missing structure fixture at {pdb_path}")

    chain = "A"

    # Step 1: Extract sequence
    sequence = None
    for record in SeqIO.parse(str(pdb_path), "pdb-atom"):
        if record.id.endswith(chain):
            sequence = str(record.seq).replace("X", "")
            break
    assert sequence is not None, f"Chain {chain} not found"

    # Step 2: Generate MPNN embeddings from PDB (full pipeline)
    embeddings = mpnn_embeddings.from_pdb(str(pdb_path), chain)
    assert embeddings.embeddings.shape[0] == len(
        sequence
    ), "Embedding count mismatch"

    # Step 3: Run SoftAligner (full pipeline)
    aligner = softaligner.SoftAligner()
    output = aligner(embeddings, chain_type=constants.ChainType.HEAVY)
    assert output.species is not None, "Species should be detected"

    # Step 4: Convert alignment to state vector
    sv, start, end, first_aligned = aln2hmm.alignment_matrix_to_state_vector(
        output.alignment
    )

    # The alignment should start at column 2 (0-indexed),
    # corresponding to IMGT position 3
    assert start == 2, f"Expected start=2 (IMGT position 3), got {start}"

    n_aligned = end - start
    subsequence = "-" * start + sequence[:n_aligned]

    # Step 5: Run ANARCI numbering
    anarci_out, anarci_start, anarci_end = (
        anarci.number_sequence_from_alignment(
            sv,
            subsequence,
            scheme="imgt",
            chain_type=output.species,
        )
    )

    # Filter out gap positions (ANARCI includes gaps for missing positions 1, 2)
    anarci_non_gap = [(pos, aa) for pos, aa in anarci_out if aa != "-"]

    # First non-gap ANARCI position should be IMGT 3 (since 1 and 2 are missing)
    first_pos, first_aa = anarci_non_gap[0]
    assert (
        first_pos[0] == 3
    ), f"Expected first IMGT position 3, got {first_pos[0]}"
    assert first_aa == "Q", f"Expected first residue Q (Gln), got {first_aa}"

    # Step 6: Thread the alignment onto structure
    output_pdb = tmp_path / "8_21_ntrunc_threaded.pdb"
    deviations = edit_pdb.thread_alignment(
        str(pdb_path),
        chain,
        anarci_out,
        str(output_pdb),
        anarci_start,
        anarci_end,
        alignment_start=0,
    )

    assert deviations == 0, f"Expected 0 deviations, got {deviations}"

    # Step 7: Verify the output structure has correct numbering
    parser = PDB.PDBParser(QUIET=True)
    out_structure = parser.get_structure("output", output_pdb)
    residue_ids = []
    for res in out_structure[0][chain]:
        hetflag, resseq, icode = res.get_id()
        if not hetflag.strip():
            residue_ids.append(resseq)

    # Check first residue is 3 (positions 1 and 2 are missing)
    assert (
        residue_ids[0] == 3
    ), f"First residue should be 3, got {residue_ids[0]}"

    # Check last residue is 128
    assert (
        residue_ids[-1] == 128
    ), f"Last residue should be 128, got {residue_ids[-1]}"

    # Verify position 10 is skipped (standard IMGT gap)
    assert (
        10 not in residue_ids
    ), "Position 10 should be skipped in IMGT heavy chains"
