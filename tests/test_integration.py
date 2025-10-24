import pathlib
import tempfile

from ANARCI import anarci

from sabr import aln2hmm, edit_pdb, softaligner
from sabr.cli import fetch_sequence_from_pdb


def test_pipeline_generates_deviations():
    pdb_path = pathlib.Path(__file__).resolve().parent / "data/5omm_imgt.pdb"
    chain = "C"

    sequence = fetch_sequence_from_pdb(str(pdb_path), chain)
    soft_aligner = softaligner.SoftAligner()
    out = soft_aligner(str(pdb_path), chain)

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

    with tempfile.TomporaryDirectory() as tmpdir:
        deviations = edit_pdb.thread_alignment(
            pdb_path,
            chain,
            anarci_out,
            f"{tmpdir}/test.pdb",
            anarci_start,
            anarci_end,
            start,
        )
    assert len(deviations) > 1
