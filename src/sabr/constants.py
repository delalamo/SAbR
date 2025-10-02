EMBED_DIM = 64
N_MPNN_LAYERS = 3

IMGT_FRAMEWORKS = {
    "FW1": list(range(1, 26)),
    "FW2": list(range(39, 56)),
    "FW3": list(range(66, 105)),
    "FW4": list(range(118, 129)),
}

NON_CDR_RESIDUES = sum(IMGT_FRAMEWORKS.values(), [])
