from sabr import constants


def test_residue_partitions_cover_full_range():
    combined = set(constants.CDR_RESIDUES) | set(constants.NON_CDR_RESIDUES)
    assert combined == set(range(1, 129))
    assert len(constants.CDR_RESIDUES) + len(constants.NON_CDR_RESIDUES) == 128


def test_imgt_loops_are_within_range():
    for start, end in constants.IMGT_LOOPS.values():
        assert 1 <= start < end <= 128


def test_imgt_frameworks_no_overlap():
    """Test that framework regions don't overlap."""
    all_fw_residues = []
    for fw_residues in constants.IMGT_FRAMEWORKS.values():
        all_fw_residues.extend(fw_residues)

    # Each residue should appear only once
    assert len(all_fw_residues) == len(set(all_fw_residues))


def test_imgt_frameworks_match_non_cdr():
    """Test that IMGT_FRAMEWORKS matches NON_CDR_RESIDUES."""
    fw_residues = sum(constants.IMGT_FRAMEWORKS.values(), [])
    assert set(fw_residues) == set(constants.NON_CDR_RESIDUES)


def test_cdr_residues_match_imgt_loops():
    """Test that CDR_RESIDUES contains all IMGT loop positions."""
    loop_residues = []
    for start, end in constants.IMGT_LOOPS.values():
        loop_residues.extend(range(start, end + 1))

    assert set(loop_residues) == set(constants.CDR_RESIDUES)


def test_aa_3to1_has_20_amino_acids():
    """Test that AA_3TO1 contains all 20 standard amino acids."""
    assert len(constants.AA_3TO1) == 20


def test_aa_3to1_all_uppercase():
    """Test that all single-letter codes are uppercase."""
    for single_letter in constants.AA_3TO1.values():
        assert single_letter.isupper()
        assert len(single_letter) == 1


def test_aa_3to1_three_letter_codes():
    """Test that all three-letter codes are correct format."""
    for three_letter in constants.AA_3TO1.keys():
        assert len(three_letter) == 3
        assert three_letter.isupper()


def test_embed_dim_positive():
    """Test that EMBED_DIM is a positive integer."""
    assert isinstance(constants.EMBED_DIM, int)
    assert constants.EMBED_DIM > 0


def test_n_mpnn_layers_positive():
    """Test that N_MPNN_LAYERS is a positive integer."""
    assert isinstance(constants.N_MPNN_LAYERS, int)
    assert constants.N_MPNN_LAYERS > 0


def test_imgt_loops_ordered():
    """Test that CDR loops are in expected order."""
    expected_keys = ["CDR1", "CDR2", "CDR3"]
    assert list(constants.IMGT_LOOPS.keys()) == expected_keys


def test_imgt_frameworks_ordered():
    """Test that framework regions are in expected order."""
    expected_keys = ["FW1", "FW2", "FW3", "FW4"]
    assert list(constants.IMGT_FRAMEWORKS.keys()) == expected_keys
