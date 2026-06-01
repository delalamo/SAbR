"""IMGT numbering constants and correction anchors."""

IMGT_MAX_POSITION = 128

# FR1 anchor points.
FR1_ANCHOR_START_COL = 5
FR1_ANCHOR_END_COL = 12
FR1_KAPPA_RESIDUE_COUNT = 8

# DE loop positions 81-84.
FR3_POS81_COL = 80
FR3_POS82_COL = 81
FR3_POS83_COL = 82
FR3_POS84_COL = 83

# C-terminus correction position.
C_TERMINUS_ANCHOR_POSITION = 124

IMGT_REGIONS = {
    "FR1": list(range(1, 27)),
    "CDR1": list(range(27, 39)),
    "FR2": list(range(39, 56)),
    "CDR2": list(range(56, 66)),
    "FR3": list(range(66, 105)),
    "CDR3": list(range(105, 118)),
    "FR4": list(range(118, 129)),
}

IMGT_LOOPS = {
    "CDR1": (IMGT_REGIONS["CDR1"][0], IMGT_REGIONS["CDR1"][-1]),
    "CDR2": (IMGT_REGIONS["CDR2"][0], IMGT_REGIONS["CDR2"][-1]),
    "CDR3": (IMGT_REGIONS["CDR3"][0], IMGT_REGIONS["CDR3"][-1]),
}

# Spatially conserved anchors used by deterministic CDR correction. These are
# not necessarily CDR termini.
CDR_ANCHORS = {
    "CDR1": (23, 40),
    "CDR2": (54, 67),
    "CDR3": (104, 118),
}

# Variable-length positions optionally ignored in benchmark comparisons.
VARIABLE_LENGTH_POSITIONS = set(
    IMGT_REGIONS["CDR1"]
    + IMGT_REGIONS["CDR2"]
    + IMGT_REGIONS["CDR3"]
    + list(range(79, 85))
)
