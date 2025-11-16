from pathlib import Path

import numpy as np
import pytest
from ANARCI import anarci
from Bio import SeqIO

from sabr import aln2hmm, edit_pdb

DATA_DIR = Path(__file__).parent / "data"

FIXTURES = {
    "8_21": {
        "pdb": DATA_DIR / "8_21_renumbered.pdb",
        "chain": "A",
        "alignment": DATA_DIR / "8_21_renumbered_alignment.npz",
        "expected_deviations": 0,
    },
    "5omm": {
        "pdb": DATA_DIR / "5omm_imgt.pdb",
        "chain": "C",
        "alignment": DATA_DIR / "5omm_imgt_alignment.npz",
        "expected_deviations": 104,
    },
}


def fetch_sequence_from_pdb(pdb_file: Path, chain: str) -> str:
    """Return the sequence for ``chain`` from ``pdb_file``."""
    for record in SeqIO.parse(pdb_file, "pdb-atom"):
        if record.id.endswith(chain):
            return str(record.seq).replace("X", "")
    ids = [r.id for r in SeqIO.parse(pdb_file, "pdb-atom")]
    raise ValueError(f"Chain {chain} not found in {pdb_file} (contains {ids})")


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
    sequence = fetch_sequence_from_pdb(pdb_path, chain)
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
    assert deviations == data["expected_deviations"]
