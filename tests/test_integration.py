from pathlib import Path

import numpy as np
import pytest
from ANARCI import anarci
from Bio import PDB, SeqIO
from click.testing import CliRunner

from sabr import aln2hmm, cli, edit_pdb, mpnn_embeddings, renumber, softaligner
from tests.conftest import (
    FIXTURES,
    create_dummy_aligner,
    create_dummy_from_pdb,
    extract_residue_ids_from_pdb,
    load_alignment_fixture,
    resolve_data_path,
)


def run_threading_pipeline(
    pdb_path: Path,
    chain: str,
    alignment: np.ndarray,
    chain_type: str,
    tmp_path: Path,
) -> int:
    sequence = None
    for record in SeqIO.parse(str(pdb_path), "pdb-atom"):
        if record.id.endswith(chain):
            sequence = str(record.seq).replace("X", "")
            break
    if sequence is None:
        raise ValueError(f"Chain {chain} not found in {pdb_path}")

    hmm_output = aln2hmm.alignment_matrix_to_state_vector(alignment)
    n_aligned = hmm_output.imgt_end - hmm_output.imgt_start
    subsequence = "-" * hmm_output.imgt_start + sequence[:n_aligned]

    anarci_alignment, anarci_start, anarci_end = (
        anarci.number_sequence_from_alignment(
            hmm_output.states, subsequence, scheme="imgt", chain_type=chain_type
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
    alignment, chain_type = load_alignment_fixture(data["alignment"])
    deviations = run_threading_pipeline(
        data["pdb"],
        data["chain"],
        alignment,
        chain_type,
        tmp_path,
    )
    min_expected = data.get("min_deviations")
    max_expected = data.get("max_deviations")
    assert min_expected is not None and max_expected is not None
    assert min_expected <= deviations <= max_expected


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
    alignment, chain_type = load_alignment_fixture(data["alignment"])

    DummyAligner = create_dummy_aligner(alignment, chain_type)
    dummy_from_pdb = create_dummy_from_pdb()

    monkeypatch.setattr(mpnn_embeddings, "from_pdb", dummy_from_pdb)
    monkeypatch.setattr(
        renumber.softaligner, "SoftAligner", lambda: DummyAligner()
    )

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

    original_ids = extract_residue_ids_from_pdb(data["pdb"], data["chain"])
    threaded_ids = extract_residue_ids_from_pdb(output_pdb, data["chain"])
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
    alignment, chain_type = load_alignment_fixture(data["alignment"])

    captured_kwargs = {}
    DummyAligner = create_dummy_aligner(alignment, chain_type, captured_kwargs)
    dummy_from_pdb = create_dummy_from_pdb()

    monkeypatch.setattr(mpnn_embeddings, "from_pdb", dummy_from_pdb)
    monkeypatch.setattr(
        renumber.softaligner, "SoftAligner", lambda: DummyAligner()
    )

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


@pytest.mark.parametrize("chain_type", ["H", "L", "auto"])
def test_cli_chain_type_argument(monkeypatch, tmp_path, chain_type):
    """Test that CLI accepts valid --chain-type values (H, L, auto)."""
    data = FIXTURES["8_21"]
    if not data["pdb"].exists():
        pytest.skip(f"Missing structure fixture at {data['pdb']}")
    alignment, fixture_chain_type = load_alignment_fixture(data["alignment"])

    DummyAligner = create_dummy_aligner(alignment, fixture_chain_type)
    dummy_from_pdb = create_dummy_from_pdb()

    monkeypatch.setattr(mpnn_embeddings, "from_pdb", dummy_from_pdb)
    monkeypatch.setattr(
        renumber.softaligner, "SoftAligner", lambda: DummyAligner()
    )

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
            "-t",
            chain_type,
        ],
    )
    assert result.exit_code == 0, result.output


def test_cli_rejects_invalid_chain_type():
    """Test that CLI rejects invalid --chain-type values."""
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
            "-t",
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

    alignment, chain_type = load_alignment_fixture(data["alignment"])

    hmm_output = aln2hmm.alignment_matrix_to_state_vector(alignment)

    assert (
        hmm_output.imgt_start == 1
    ), f"Expected imgt_start=1, got {hmm_output.imgt_start}"

    first_state = hmm_output.states[0]
    assert first_state.residue_number == 2
    assert first_state.insertion_code == "m"
    assert first_state.mapped_residue == 1

    n_aligned = hmm_output.imgt_end - hmm_output.imgt_start
    assert n_aligned > 0

    match_states = [s for s in hmm_output.states if s.insertion_code == "m"]
    assert len(match_states) > 100


@pytest.mark.slow
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
    output = aligner(embeddings)
    assert output.chain_type is not None, "Chain type should be detected"

    # Step 4: Convert alignment to state vector
    hmm_output = aln2hmm.alignment_matrix_to_state_vector(output.alignment)

    # The alignment should start at column 2 (0-indexed),
    # corresponding to IMGT position 3
    assert (
        hmm_output.imgt_start == 2
    ), f"Expected start=2 (IMGT position 3), got {hmm_output.imgt_start}"

    n_aligned = hmm_output.imgt_end - hmm_output.imgt_start
    subsequence = "-" * hmm_output.imgt_start + sequence[:n_aligned]

    # Step 5: Run ANARCI numbering
    anarci_out, anarci_start, anarci_end = (
        anarci.number_sequence_from_alignment(
            hmm_output.states,
            subsequence,
            scheme="imgt",
            chain_type=output.chain_type,
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


@pytest.mark.parametrize(
    ("start", "end", "error_check"),
    [
        ("50", "10", lambda o: "end" in o.lower() and "start" in o.lower()),
        ("50", "50", lambda o: "end" in o.lower() or "greater" in o.lower()),
        (
            "-10",
            "50",
            lambda o: "negative" in o.lower() or "non-negative" in o.lower(),
        ),
    ],
)
def test_cli_rejects_invalid_residue_ranges(start, end, error_check):
    """Test that CLI rejects invalid residue range values."""
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
            "--residue-range",
            start,
            end,
        ],
    )
    assert result.exit_code != 0
    assert error_check(result.output)


@pytest.mark.parametrize(
    ("start", "end"),
    [
        ("1", "128"),  # Full range matching all residues
        ("0", "0"),  # Process all residues
    ],
)
def test_cli_accepts_valid_residue_ranges(monkeypatch, tmp_path, start, end):
    """Test that CLI accepts valid residue range values."""
    data = FIXTURES["8_21"]
    if not data["pdb"].exists():
        pytest.skip(f"Missing structure fixture at {data['pdb']}")
    alignment, chain_type = load_alignment_fixture(data["alignment"])

    DummyAligner = create_dummy_aligner(alignment, chain_type)
    dummy_from_pdb = create_dummy_from_pdb()

    monkeypatch.setattr(mpnn_embeddings, "from_pdb", dummy_from_pdb)
    monkeypatch.setattr(
        renumber.softaligner, "SoftAligner", lambda: DummyAligner()
    )

    runner = CliRunner()
    output_pdb = tmp_path / f"residue_range_{start}_{end}.pdb"
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
            "--residue-range",
            start,
            end,
        ],
    )
    assert result.exit_code == 0, result.output


@pytest.mark.slow
def test_from_chain_produces_same_embeddings_as_from_pdb():
    """Test that from_chain() produces same embeddings as from_pdb().

    This is an integration test that runs the REAL embedding computation
    (no mocking) and verifies that loading a chain from a structure and
    calling from_chain() produces identical results to calling from_pdb()
    directly with the file path.
    """
    data = FIXTURES["5omm"]
    if not data["pdb"].exists():
        pytest.skip(f"Missing structure fixture at {data['pdb']}")

    chain_id = data["chain"]
    pdb_path = str(data["pdb"])

    # Method 1: from_pdb (file-based)
    embeddings_from_pdb = mpnn_embeddings.from_pdb(pdb_path, chain_id)

    # Method 2: from_chain (in-memory)
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("test", pdb_path)
    chain_obj = structure[0][chain_id]
    embeddings_from_chain = mpnn_embeddings.from_chain(chain_obj)

    # Verify both methods produce equivalent results
    assert embeddings_from_pdb.sequence == embeddings_from_chain.sequence, (
        f"Sequence mismatch:\n"
        f"from_pdb: {embeddings_from_pdb.sequence}\n"
        f"from_chain: {embeddings_from_chain.sequence}"
    )

    assert embeddings_from_pdb.idxs == embeddings_from_chain.idxs, (
        f"Residue IDs mismatch:\n"
        f"from_pdb: {embeddings_from_pdb.idxs}\n"
        f"from_chain: {embeddings_from_chain.idxs}"
    )

    assert (
        embeddings_from_pdb.embeddings.shape
        == embeddings_from_chain.embeddings.shape
    ), (
        f"Embeddings shape mismatch: "
        f"{embeddings_from_pdb.embeddings.shape} vs "
        f"{embeddings_from_chain.embeddings.shape}"
    )

    np.testing.assert_allclose(
        embeddings_from_pdb.embeddings,
        embeddings_from_chain.embeddings,
        rtol=1e-5,
        atol=1e-5,
        err_msg="Embeddings values differ between from_pdb and from_chain",
    )


@pytest.mark.slow
def test_renumber_structure_end_to_end():
    """End-to-end test for renumber_structure() BioPython API.

    This test runs the FULL renumbering pipeline without mocking:
    1. Load structure with BioPython
    2. Call renumber_structure()
    3. Verify the returned structure has valid IMGT numbering

    This ensures the BioPython API works correctly for in-memory processing.
    """
    data = FIXTURES["5omm"]
    if not data["pdb"].exists():
        pytest.skip(f"Missing structure fixture at {data['pdb']}")

    chain_id = data["chain"]
    pdb_path = str(data["pdb"])

    # Load structure
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("test", pdb_path)

    # Run renumber_structure (full pipeline, no mocking)
    renumbered = renumber.renumber_structure(
        structure,
        chain=chain_id,
        numbering_scheme="imgt",
    )

    # Verify result is a BioPython Structure
    assert renumbered is not None
    assert hasattr(renumbered, "get_chains")

    # Get residues from the renumbered chain
    renumbered_residues = list(renumbered[0][chain_id].get_residues())
    assert len(renumbered_residues) > 0, "Renumbered chain has no residues"

    # Collect residue numbers (excluding heteroatoms)
    residue_numbers = []
    for res in renumbered_residues:
        hetflag, resseq, icode = res.get_id()
        if not hetflag.strip():
            residue_numbers.append(resseq)

    # Verify IMGT numbering constraints:
    # 1. Residue numbers should generally be in range 1-128 for Fv
    # 2. Numbers should be mostly increasing (with possible gaps)
    assert (
        len(residue_numbers) > 50
    ), f"Expected >50 residues, got {len(residue_numbers)}"

    # Check that the maximum residue number is reasonable for IMGT
    max_resnum = max(residue_numbers)
    assert (
        max_resnum <= 150
    ), f"Maximum residue number {max_resnum} exceeds expected IMGT range"

    # Verify the structure can be saved (basic integrity check)
    from io import StringIO

    from Bio.PDB import PDBIO

    io = PDBIO()
    io.set_structure(renumbered)
    output = StringIO()
    io.save(output)
    pdb_content = output.getvalue()
    assert len(pdb_content) > 0, "Failed to save renumbered structure"
    assert "ATOM" in pdb_content, "Output PDB has no ATOM records"
