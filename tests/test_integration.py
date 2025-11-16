from pathlib import Path

from ANARCI import anarci
from Bio import SeqIO

from sabr import aln2hmm, edit_pdb, softaligner

DEFAULT_PDB = Path(__file__).parent / "data" / "8_21_renumbered.pdb"
CHAIN = "A"


def fetch_sequence_from_pdb(pdb_file: Path, chain: str) -> str:
    """Return the sequence for ``chain`` from ``pdb_file``."""
    for record in SeqIO.parse(pdb_file, "pdb-atom"):
        if record.id.endswith(chain):
            return str(record.seq).replace("X", "")
    ids = [r.id for r in SeqIO.parse(pdb_file, "pdb-atom")]
    raise ValueError(f"Chain {chain} not found in {pdb_file} (contains {ids})")


def test_thread_alignment_produces_zero_deviations(tmp_path):
    """Full pipeline integration test mirroring ``run_test.py``."""

    sequence = fetch_sequence_from_pdb(DEFAULT_PDB, CHAIN)
    aligner = softaligner.SoftAligner()
    out = aligner(str(DEFAULT_PDB), CHAIN)

    sv, start, end = aln2hmm.alignment_matrix_to_state_vector(out.alignment)
    subsequence = "-" * start + sequence[start:end]

    anarci_out, anarci_start, anarci_end = (
        anarci.number_sequence_from_alignment(
            sv,
            subsequence,
            scheme="imgt",
            chain_type=out.species,
        )
    )

    output_pdb = tmp_path / "threaded.pdb"
    deviations = edit_pdb.thread_alignment(
        str(DEFAULT_PDB),
        CHAIN,
        anarci_out,
        str(output_pdb),
        anarci_start,
        anarci_end,
        alignment_start=start,
    )

    assert deviations == 0
