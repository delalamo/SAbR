from sabr import constants


def test_residue_partitions_cover_full_range():
    combined = set(constants.CDR_RESIDUES) | set(constants.NON_CDR_RESIDUES)
    assert combined == set(range(1, 129))
    assert len(constants.CDR_RESIDUES) + len(constants.NON_CDR_RESIDUES) == 128


def test_imgt_loops_are_within_range():
    for start, end in constants.IMGT_LOOPS.values():
        assert 1 <= start < end <= 128
