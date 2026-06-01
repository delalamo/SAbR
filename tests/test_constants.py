from sabr.nn.config import EMBED_DIM, N_MPNN_LAYERS
from sabr.numbering.imgt import IMGT_LOOPS
from sabr.structure.residues import AA_3TO1


def test_imgt_loops_are_within_range():
    for start, end in IMGT_LOOPS.values():
        assert 1 <= start < end <= 128


def test_aa_3to1_has_20_amino_acids():
    """Test that AA_3TO1 contains all 20 standard amino acids."""
    assert len(AA_3TO1) == 20


def test_aa_3to1_all_uppercase():
    """Test that all single-letter codes are uppercase."""
    for single_letter in AA_3TO1.values():
        assert single_letter.isupper()
        assert len(single_letter) == 1


def test_aa_3to1_three_letter_codes():
    """Test that all three-letter codes are correct format."""
    for three_letter in AA_3TO1.keys():
        assert len(three_letter) == 3
        assert three_letter.isupper()


def test_embed_dim_positive():
    """Test that EMBED_DIM is a positive integer."""
    assert isinstance(EMBED_DIM, int)
    assert EMBED_DIM > 0


def test_n_mpnn_layers_positive():
    """Test that N_MPNN_LAYERS is a positive integer."""
    assert isinstance(N_MPNN_LAYERS, int)
    assert N_MPNN_LAYERS > 0


def test_imgt_loops_ordered():
    """Test that CDR loops are in expected order."""
    expected_keys = ["CDR1", "CDR2", "CDR3"]
    assert list(IMGT_LOOPS.keys()) == expected_keys
