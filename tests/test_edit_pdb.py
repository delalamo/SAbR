import pytest
from Bio.PDB import Chain, Residue

from sabr import edit_pdb


def build_residue(
    number: int, name: str, hetflag: str = " "
) -> Residue.Residue:
    """Build a test residue with given number, name, and hetflag."""
    resid = (hetflag, number, " ")
    residue = Residue.Residue(resid, name, " ")
    return residue


def test_thread_onto_chain_updates_residue_ids():
    """Core test: thread_onto_chain correctly updates residue IDs."""
    chain = Chain.Chain("A")
    chain.add(build_residue(1, "ALA"))
    chain.add(build_residue(2, "GLY"))

    anarci_out = [
        ((1, " "), "A"),
        ((2, " "), "G"),
    ]

    threaded, deviations = edit_pdb.thread_onto_chain(
        chain=chain,
        anarci_out=anarci_out,
        anarci_start=0,
        anarci_end=2,
        alignment_start=0,
    )

    new_ids = [res.get_id() for res in threaded.get_residues()]
    assert new_ids == [(" ", 1, " "), (" ", 2, " ")]
    assert threaded.id == "A"
    assert deviations == 0


def test_thread_onto_chain_with_insertion_codes():
    """Core test: insertion codes in ANARCI output are handled correctly."""
    chain = Chain.Chain("A")
    chain.add(build_residue(1, "ALA"))
    chain.add(build_residue(2, "GLY"))
    chain.add(build_residue(3, "VAL"))

    anarci_out = [
        ((1, " "), "A"),
        ((1, "A"), "G"),  # Insertion code A
        ((2, " "), "V"),
    ]

    threaded, deviations = edit_pdb.thread_onto_chain(
        chain=chain,
        anarci_out=anarci_out,
        anarci_start=0,
        anarci_end=3,
        alignment_start=0,
    )

    residues = list(threaded.get_residues())
    assert residues[1].get_id()[2] == "A"


def test_thread_onto_chain_with_deletions():
    """Core test: deletions (gaps) in ANARCI output are handled correctly."""
    chain = Chain.Chain("A")
    chain.add(build_residue(1, "ALA"))
    chain.add(build_residue(2, "GLY"))

    anarci_out = [
        ((1, " "), "-"),  # Deletion
        ((2, " "), "A"),
        ((3, " "), "G"),
    ]

    threaded, deviations = edit_pdb.thread_onto_chain(
        chain=chain,
        anarci_out=anarci_out,
        anarci_start=0,
        anarci_end=3,
        alignment_start=0,
    )

    assert len(list(threaded.get_residues())) == 2


def test_thread_onto_chain_counts_deviations():
    """Core test: deviations are counted when numbering changes."""
    chain = Chain.Chain("A")
    chain.add(build_residue(5, "ALA"))  # Originally numbered 5
    chain.add(build_residue(10, "GLY"))  # Originally numbered 10

    anarci_out = [
        ((1, " "), "A"),  # Renumbered to 1
        ((2, " "), "G"),  # Renumbered to 2
    ]

    threaded, deviations = edit_pdb.thread_onto_chain(
        chain=chain,
        anarci_out=anarci_out,
        anarci_start=0,
        anarci_end=2,
        alignment_start=0,
    )

    assert deviations == 2


@pytest.mark.slow
def test_8sve_L_extended_insertion_codes(tmp_path):
    """E2E test: 8SVE_L antibody with extended insertion codes.

    Verifies PDB output fails with extended codes, CIF output succeeds.
    """
    pytest.importorskip("ANARCI")
    from importlib import resources
    from pathlib import Path

    from ANARCI import anarci

    from sabr.alignment import SoftAligner, alignment_matrix_to_state_vector
    from sabr.embeddings import from_pdb

    DATA_PACKAGE = "tests.data"
    pdb_path = Path(resources.files(DATA_PACKAGE) / "8sve_L.pdb")

    if not pdb_path.exists():
        pytest.skip(f"Missing structure fixture at {pdb_path}")

    try:
        input_data = from_pdb(str(pdb_path), "M")
        aligner = SoftAligner()
        result = aligner(input_data)

        sequence = input_data.sequence
        hmm_output = alignment_matrix_to_state_vector(result.alignment)
        n_aligned = hmm_output.imgt_end - hmm_output.imgt_start
        subsequence = "-" * hmm_output.imgt_start + sequence[:n_aligned]

        anarci_out, anarci_start, anarci_end = (
            anarci.number_sequence_from_alignment(
                hmm_output.states,
                subsequence,
                scheme="imgt",
                chain_type=result.chain_type,
            )
        )

        # PDB output should raise ValueError due to extended insertion codes
        output_pdb = tmp_path / "8sve_L_output.pdb"
        with pytest.raises(ValueError, match="Extended insertion codes"):
            edit_pdb.thread_alignment(
                str(pdb_path),
                "M",
                anarci_out,
                str(output_pdb),
                anarci_start,
                anarci_end,
                alignment_start=0,
            )

        # CIF output should succeed
        output_cif = tmp_path / "8sve_L_output.cif"
        edit_pdb.thread_alignment(
            str(pdb_path),
            "M",
            anarci_out,
            str(output_cif),
            anarci_start,
            anarci_end,
            alignment_start=0,
        )

        assert output_cif.exists()

        # Use BioPython to read the CIF (Gemmi can't handle extended icodes)
        from Bio.PDB.MMCIFParser import MMCIFParser

        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure("test", str(output_cif))
        model = structure[0]

        # Find chain M in the structure
        chain_m = model["M"]
        assert chain_m is not None, "Chain M not found in output"

        # Verify residues are present and extended insertion codes work
        residues = [
            res for res in chain_m.get_residues() if not res.id[0].strip()
        ]
        assert len(residues) > 0

        # Check that we have extended insertion codes (multi-char like 'AA')
        icodes = [res.id[2] for res in residues if len(res.id[2].strip()) > 1]
        assert len(icodes) > 0, "Expected extended insertion codes in output"

    except ImportError:
        pytest.skip("SoftAligner dependencies not available")
