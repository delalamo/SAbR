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
def test_thread_alignment_has_expected_deviations(tmp_path, fixture_key):
    """Core test: verify threading pipeline produces expected deviations."""
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
def test_cli_produces_correct_numbering(
    monkeypatch, tmp_path, fixture_key, expect_same
):
    """Core test: CLI renumbering produces expected residue IDs."""
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


def test_alignment_start_position_correct():
    """Regression test: structures starting at IMGT position 2 are handled.

    This tests the fix for the off-by-one bug where sequences starting
    at IMGT position 2 had their first residue incorrectly numbered.
    """
    data = FIXTURES["test_heavy_chain"]
    if not data["pdb"].exists():
        pytest.skip(f"Missing structure fixture at {data['pdb']}")

    alignment, chain_type = load_alignment_fixture(data["alignment"])
    hmm_output = aln2hmm.alignment_matrix_to_state_vector(alignment)

    assert hmm_output.imgt_start == 1
    first_state = hmm_output.states[0]
    assert first_state.residue_number == 2
    assert first_state.insertion_code == "m"
    assert first_state.mapped_residue == 1


@pytest.mark.slow
def test_n_terminal_truncated_structure_end_to_end(tmp_path):
    """E2E test: structures with N-terminal truncation are handled correctly."""
    pdb_path = resolve_data_path("8_21_renumbered_ntrunc.pdb")
    if not pdb_path.exists():
        pytest.skip(f"Missing structure fixture at {pdb_path}")

    chain = "A"

    sequence = None
    for record in SeqIO.parse(str(pdb_path), "pdb-atom"):
        if record.id.endswith(chain):
            sequence = str(record.seq).replace("X", "")
            break
    assert sequence is not None

    embeddings = mpnn_embeddings.from_pdb(str(pdb_path), chain)
    aligner = softaligner.SoftAligner()
    output = aligner(embeddings)

    hmm_output = aln2hmm.alignment_matrix_to_state_vector(output.alignment)
    assert hmm_output.imgt_start == 2  # Structure starts at IMGT position 3

    n_aligned = hmm_output.imgt_end - hmm_output.imgt_start
    subsequence = "-" * hmm_output.imgt_start + sequence[:n_aligned]

    anarci_out, anarci_start, anarci_end = (
        anarci.number_sequence_from_alignment(
            hmm_output.states,
            subsequence,
            scheme="imgt",
            chain_type=output.chain_type,
        )
    )

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

    assert deviations == 0


@pytest.mark.slow
def test_from_chain_produces_same_embeddings_as_from_pdb():
    """E2E test: from_chain() and from_pdb() produce equivalent embeddings."""
    data = FIXTURES["5omm"]
    if not data["pdb"].exists():
        pytest.skip(f"Missing structure fixture at {data['pdb']}")

    chain_id = data["chain"]
    pdb_path = str(data["pdb"])

    embeddings_from_pdb = mpnn_embeddings.from_pdb(pdb_path, chain_id)

    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("test", pdb_path)
    chain_obj = structure[0][chain_id]
    embeddings_from_chain = mpnn_embeddings.from_chain(chain_obj)

    assert embeddings_from_pdb.sequence == embeddings_from_chain.sequence
    assert embeddings_from_pdb.idxs == embeddings_from_chain.idxs
    np.testing.assert_allclose(
        embeddings_from_pdb.embeddings,
        embeddings_from_chain.embeddings,
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.slow
def test_renumber_structure_end_to_end():
    """E2E test: renumber_structure() produces valid IMGT numbering."""
    data = FIXTURES["5omm"]
    if not data["pdb"].exists():
        pytest.skip(f"Missing structure fixture at {data['pdb']}")

    chain_id = data["chain"]
    pdb_path = str(data["pdb"])

    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("test", pdb_path)

    renumbered = renumber.renumber_structure(
        structure,
        chain=chain_id,
        numbering_scheme="imgt",
    )

    assert renumbered is not None
    renumbered_residues = list(renumbered[0][chain_id].get_residues())
    assert len(renumbered_residues) > 50

    residue_numbers = [
        res.get_id()[1]
        for res in renumbered_residues
        if not res.get_id()[0].strip()
    ]
    assert max(residue_numbers) <= 150
