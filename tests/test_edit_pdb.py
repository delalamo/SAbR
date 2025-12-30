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
    assert deviations == 0  # No changes in numbering


def test_thread_onto_chain_residue_mismatch():
    """Test ValueError when residue doesn't match ANARCI output."""
    chain = Chain.Chain("A")
    chain.add(build_residue(1, "ALA"))  # ALA = A
    chain.add(build_residue(2, "GLY"))  # GLY = G

    # ANARCI says position 1 should be V (VAL), not A (ALA)
    anarci_out = [
        ((1, " "), "V"),  # Mismatch!
        ((2, " "), "G"),
    ]

    with pytest.raises(ValueError, match="Residue mismatch"):
        edit_pdb.thread_onto_chain(
            chain=chain,
            anarci_out=anarci_out,
            anarci_start=0,
            anarci_end=2,
            alignment_start=0,
        )


def test_thread_onto_chain_with_hetatm():
    """Test that HETATM residues don't cause errors when present."""
    chain = Chain.Chain("A")
    chain.add(build_residue(1, "ALA"))
    chain.add(build_residue(2, "GLY"))
    # HETATM before the ANARCI window starts
    chain.add(build_residue(3, "HOH", hetflag="W"))
    chain.add(build_residue(4, "VAL"))

    # ANARCI window starts at residue 1 (skipping HETATM at position 0)
    # alignment_start=3 means we start processing from the 4th residue (index 3)
    anarci_out = [
        ((1, " "), "V"),  # VAL gets numbered as 1
    ]

    threaded, deviations = edit_pdb.thread_onto_chain(
        chain=chain,
        anarci_out=anarci_out,
        anarci_start=0,
        anarci_end=1,
        alignment_start=3,  # Start alignment from VAL (index 3)
    )

    # Should have all 4 residues
    assert len(list(threaded.get_residues())) == 4
    # VAL should be in the output
    residues = list(threaded.get_residues())
    assert any(res.get_resname() == "VAL" for res in residues)


def test_thread_onto_chain_with_insertion_codes():
    """Test handling of insertion codes in ANARCI output."""
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
    # Second residue should have insertion code "A"
    assert residues[1].get_id()[2] == "A"


def test_thread_onto_chain_with_deletions():
    """Test ANARCI output with deletions (marked as '-')."""
    chain = Chain.Chain("A")
    chain.add(build_residue(1, "ALA"))
    chain.add(build_residue(2, "GLY"))

    # Deletion at position 1
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

    # Should skip the deletion and number correctly
    assert len(list(threaded.get_residues())) == 2


def test_thread_onto_chain_before_anarci_window():
    """Test residues before ANARCI window start."""
    chain = Chain.Chain("A")
    # 5 residues total
    for i in range(1, 6):
        chain.add(build_residue(i, "ALA"))

    # ANARCI window starts at residue 2 (anarci_start=2)
    anarci_out = [
        ((1, " "), "A"),
        ((2, " "), "A"),
        ((3, " "), "A"),
    ]

    threaded, deviations = edit_pdb.thread_onto_chain(
        chain=chain,
        anarci_out=anarci_out,
        anarci_start=2,
        anarci_end=5,
        alignment_start=2,
    )

    residues = list(threaded.get_residues())
    assert len(residues) == 5
    # First 2 residues should be numbered differently (before window)


def test_thread_onto_chain_after_anarci_window():
    """Test residues after ANARCI window end."""
    chain = Chain.Chain("A")
    # 5 residues total
    for i in range(1, 6):
        chain.add(build_residue(i, "ALA"))

    # ANARCI window ends before last residue
    anarci_out = [
        ((1, " "), "A"),
        ((2, " "), "A"),
        ((3, " "), "A"),
    ]

    threaded, deviations = edit_pdb.thread_onto_chain(
        chain=chain,
        anarci_out=anarci_out,
        anarci_start=0,
        anarci_end=3,
        alignment_start=0,
    )

    residues = list(threaded.get_residues())
    assert len(residues) == 5
    # Last 2 residues should be after the ANARCI window


def test_thread_onto_chain_counts_deviations():
    """Test that deviations are counted when numbering changes."""
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

    # Both residues changed numbering
    assert deviations == 2


def test_validate_output_format_pdb_with_single_char():
    """Test that PDB format is allowed with single-character insertion codes."""
    alignment = [
        ((1, " "), "A"),
        ((1, "A"), "G"),
        ((2, " "), "V"),
    ]
    # Should not raise
    edit_pdb.validate_output_format("output.pdb", alignment)


def test_validate_output_format_pdb_with_extended_codes():
    """Test that PDB format raises error with extended insertion codes."""
    alignment = [
        ((1, " "), "A"),
        ((1, "AA"), "G"),  # Extended insertion code
        ((2, " "), "V"),
    ]
    with pytest.raises(ValueError, match="Extended insertion codes"):
        edit_pdb.validate_output_format("output.pdb", alignment)


def test_validate_output_format_cif_with_extended_codes():
    """Test that CIF format is allowed with extended insertion codes."""
    alignment = [
        ((1, " "), "A"),
        ((1, "AA"), "G"),  # Extended insertion code
        ((1, "AB"), "V"),
        ((1, "ZZ"), "L"),
        ((1, "AAA"), "I"),  # Triple-letter code
        ((2, " "), "P"),
    ]
    # Should not raise
    edit_pdb.validate_output_format("output.cif", alignment)


def test_thread_alignment_raises_error_with_pdb_and_extended_insertions(
    tmp_path,
):
    """Test thread_alignment raises ValueError with PDB and extended codes."""
    from importlib import resources
    from pathlib import Path

    DATA_PACKAGE = "tests.data"
    pdb_path = Path(resources.files(DATA_PACKAGE) / "8_21_renumbered.pdb")

    if not pdb_path.exists():
        pytest.skip(f"Missing structure fixture at {pdb_path}")

    # Create alignment with extended insertion codes
    # Sequence starts with QVQLQESGGG
    alignment = [
        ((1, " "), "Q"),
        ((1, "A"), "V"),  # Single-char insertion
        ((1, "AA"), "Q"),  # Extended insertion code
        ((2, " "), "L"),
    ]

    output_pdb = tmp_path / "test_output.pdb"

    # Should raise ValueError with extended codes and .pdb output
    with pytest.raises(ValueError, match="Extended insertion codes"):
        edit_pdb.thread_alignment(
            str(pdb_path),
            "A",
            alignment,
            str(output_pdb),
            start_res=0,
            end_res=4,
            alignment_start=0,
        )


def test_thread_alignment_succeeds_with_cif_and_extended_insertions(tmp_path):
    """Test thread_alignment succeeds with CIF and extended codes."""
    from importlib import resources
    from pathlib import Path

    DATA_PACKAGE = "tests.data"
    pdb_path = Path(resources.files(DATA_PACKAGE) / "8_21_renumbered.pdb")

    if not pdb_path.exists():
        pytest.skip(f"Missing structure fixture at {pdb_path}")

    # Create alignment with extended insertion codes
    # Sequence starts with QVQLQESGGG
    alignment = [
        ((1, " "), "Q"),
        ((1, "A"), "V"),  # Single-char insertion
        ((1, "AA"), "Q"),  # Extended insertion code
        ((2, " "), "L"),
    ]

    output_cif = tmp_path / "test_output.cif"

    # Should NOT raise with extended codes and .cif output
    edit_pdb.thread_alignment(
        str(pdb_path),
        "A",
        alignment,
        str(output_cif),
        start_res=0,
        end_res=4,
        alignment_start=0,
    )

    # Verify the output file was created
    assert output_cif.exists()


def test_8sve_L_raises_error_with_pdb_output(tmp_path):
    """Test 8SVE_L antibody with huge insertions raises error with PDB."""
    pytest.importorskip("ANARCI")
    from importlib import resources
    from pathlib import Path

    from ANARCI import anarci

    from sabr import aln2hmm, mpnn_embeddings, softaligner

    DATA_PACKAGE = "tests.data"
    pdb_path = Path(resources.files(DATA_PACKAGE) / "8sve_L.pdb")

    if not pdb_path.exists():
        pytest.skip(f"Missing structure fixture at {pdb_path}")

    # Use SoftAligner to generate alignment
    try:
        # Generate embeddings first (also extracts sequence)
        input_data = mpnn_embeddings.from_pdb(str(pdb_path), "M")
        aligner = softaligner.SoftAligner()
        result = aligner(input_data)

        # Convert to ANARCI format
        sequence = input_data.sequence
        hmm_output = aln2hmm.alignment_matrix_to_state_vector(result.alignment)
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

        # Try to output to PDB format - should raise ValueError
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
    except ImportError:
        pytest.skip("SoftAligner dependencies not available")


def test_8sve_L_succeeds_with_cif_output_and_correct_numbering(tmp_path):
    """Test 8SVE_L succeeds with CIF output and verify extended codes.

    Verifies extended insertion codes are tolerated and output is created.
    """
    pytest.importorskip("ANARCI")
    from importlib import resources
    from pathlib import Path

    from ANARCI import anarci
    from Bio import PDB

    from sabr import aln2hmm, mpnn_embeddings, softaligner

    DATA_PACKAGE = "tests.data"
    pdb_path = Path(resources.files(DATA_PACKAGE) / "8sve_L.pdb")

    if not pdb_path.exists():
        pytest.skip(f"Missing structure fixture at {pdb_path}")

    # Use SoftAligner to generate alignment
    try:
        # Generate embeddings first (also extracts sequence)
        input_data = mpnn_embeddings.from_pdb(str(pdb_path), "M")
        aligner = softaligner.SoftAligner()
        result = aligner(input_data)

        # Convert to ANARCI format
        sequence = input_data.sequence
        hmm_output = aln2hmm.alignment_matrix_to_state_vector(result.alignment)
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

        # Output to CIF format - should succeed
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

        # Verify the output file was created
        assert output_cif.exists()

        # Parse the output and verify extended insertion codes are present
        parser = PDB.MMCIFParser(QUIET=True)
        structure = parser.get_structure("8sve", str(output_cif))

        residues = list(structure[0]["M"].get_residues())

        # Check first residue is numbered 1
        first_res = residues[0]
        hetflag, resnum, icode = first_res.get_id()
        assert resnum == 1, f"First residue should be numbered 1, got {resnum}"

        # Check last residue numbered in expected range
        # (IMGT light chain ends around 120-128)
        last_res = residues[-1]
        hetflag, resnum, icode = last_res.get_id()
        assert (
            120 <= resnum <= 128
        ), f"Last residue should be around 120-128, got {resnum}"

        # Check that there are extended insertion codes present
        has_extended = False
        extended_codes_found = []
        for res in residues:
            hetflag, resnum, icode = res.get_id()
            if len(icode.strip()) > 1:
                has_extended = True
                extended_codes_found.append((resnum, icode))
                if (
                    len(extended_codes_found) >= 5
                ):  # Just collect a few examples
                    break

        assert (
            has_extended
        ), "Expected to find extended insertion codes in 8sve_L output"

    except ImportError:
        pytest.skip("SoftAligner dependencies not available")
