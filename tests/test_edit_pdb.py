import pytest
from Bio.PDB import Chain, Residue

from sabr import edit_pdb


def build_residue(number, name, hetflag=" "):
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


def test_has_extended_insertion_codes_single_char():
    """Test if single-character insertion codes are flagged as extended"""
    alignment = [
        ((1, " "), "A"),
        ((1, "A"), "G"),
        ((1, "B"), "V"),
        ((2, " "), "L"),
    ]
    assert not edit_pdb.has_extended_insertion_codes(alignment)


def test_has_extended_insertion_codes_multi_char():
    """Test that multi-character insertion codes are flagged as extended."""
    alignment = [
        ((1, " "), "A"),
        ((1, "A"), "G"),
        ((1, "AA"), "V"),  # Extended insertion code
        ((2, " "), "L"),
    ]
    assert edit_pdb.has_extended_insertion_codes(alignment)


def test_has_extended_insertion_codes_empty():
    """Test empty alignment."""
    assert not edit_pdb.has_extended_insertion_codes([])


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
