from pathlib import Path

import numpy as np
import pytest
from Bio import PDB, SeqIO
from click.testing import CliRunner

import sabr.renumber as renumber_module
from sabr.alignment.aln2hmm import alignment_matrix_to_state_vector
from sabr.alignment.soft_aligner import SoftAligner
from sabr.cli.main import main as cli_main
from sabr.embeddings.mpnn import from_pdb as mpnn_from_pdb
from sabr.numbering.anarci import build_anarci_subsequence, number_from_alignment
from sabr.options import RenumberOptions
from sabr.renumber import renumber_structure
from sabr.structure.threading import thread_alignment
from sabr.types import NumberingScheme, parse_chain_type
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

    hmm_output = alignment_matrix_to_state_vector(alignment)
    subsequence = build_anarci_subsequence(sequence, hmm_output)
    parsed_chain_type = parse_chain_type(chain_type)
    if parsed_chain_type is None:
        raise AssertionError("Fixture chain type must be concrete")
    anarci_alignment = number_from_alignment(
        hmm_output.states,
        subsequence,
        NumberingScheme.IMGT,
        parsed_chain_type,
    )

    output_pdb = tmp_path / f"{pdb_path.stem}_{chain}_threaded.pdb"
    return thread_alignment(
        str(pdb_path),
        chain,
        anarci_alignment,
        str(output_pdb),
        alignment_start=0,
    )


def _residue_id_strings(ids):
    return [f"{resseq}{icode.strip()}" for _hetflag, resseq, icode in ids]


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

    monkeypatch.setattr(renumber_module, "from_pdb", dummy_from_pdb)
    monkeypatch.setattr(
        renumber_module, "SoftAligner", lambda **_kwargs: DummyAligner()
    )

    runner = CliRunner()
    output_pdb = tmp_path / f"{fixture_key}_cli.pdb"
    result = runner.invoke(
        cli_main,
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


@pytest.mark.golden
def test_heavy_fixture_matches_exact_residue_ids(tmp_path):
    """Golden regression for exact IMGT residue IDs, including insertions."""
    data = FIXTURES["test_heavy_chain"]
    if not data["pdb"].exists():
        pytest.skip(f"Missing structure fixture at {data['pdb']}")

    alignment, chain_type = load_alignment_fixture(data["alignment"])
    run_threading_pipeline(
        data["pdb"],
        data["chain"],
        alignment,
        chain_type,
        tmp_path,
    )

    # TODO: add equivalent exact golden fixtures for kappa and lambda once
    # saved light-chain alignments are available.
    output_pdb = tmp_path / f"{data['pdb'].stem}_{data['chain']}_threaded.pdb"
    expected = [str(pos) for pos in range(2, 10)]
    expected += [str(pos) for pos in range(11, 30)]
    expected += [str(pos) for pos in range(37, 60)]
    expected += [str(pos) for pos in range(62, 73)]
    expected += [str(pos) for pos in range(74, 112)]
    expected += ["111A", "111B", "111C", "112C", "112B", "112A"]
    expected += [str(pos) for pos in range(112, 129)]

    threaded_ids = extract_residue_ids_from_pdb(output_pdb, data["chain"])
    assert _residue_id_strings(threaded_ids) == expected


def test_alignment_start_position_correct():
    """Regression test: structures starting at IMGT position 2 are handled.

    This tests the fix for the off-by-one bug where sequences starting
    at IMGT position 2 had their first residue incorrectly numbered.
    """
    data = FIXTURES["test_heavy_chain"]
    if not data["pdb"].exists():
        pytest.skip(f"Missing structure fixture at {data['pdb']}")

    alignment, chain_type = load_alignment_fixture(data["alignment"])
    hmm_output = alignment_matrix_to_state_vector(alignment)

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

    embeddings = mpnn_from_pdb(str(pdb_path), chain)
    aligner = SoftAligner()
    output = aligner(embeddings)

    hmm_output = alignment_matrix_to_state_vector(output.alignment)
    assert hmm_output.imgt_start == 2  # Structure starts at IMGT position 3

    subsequence = build_anarci_subsequence(sequence, hmm_output)
    anarci_out = number_from_alignment(
        hmm_output.states,
        subsequence,
        NumberingScheme.IMGT,
        output.selected_chain_type,
    )

    output_pdb = tmp_path / "8_21_ntrunc_threaded.pdb"
    deviations = thread_alignment(
        str(pdb_path),
        chain,
        anarci_out,
        str(output_pdb),
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

    embeddings_from_pdb = mpnn_from_pdb(pdb_path, chain_id)

    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("test", pdb_path)
    chain_obj = structure[0][chain_id]
    from sabr.embeddings.mpnn import from_chain as mpnn_from_chain

    embeddings_from_chain = mpnn_from_chain(chain_obj)

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

    renumbered = renumber_structure(
        structure,
        chain_id=chain_id,
        options=RenumberOptions(),
    )

    assert renumbered is not None
    renumbered_residues = list(renumbered[0][chain_id].get_residues())
    assert len(renumbered_residues) > 50

    residue_numbers = [
        res.get_id()[1] for res in renumbered_residues if not res.get_id()[0].strip()
    ]
    assert max(residue_numbers) <= 150
