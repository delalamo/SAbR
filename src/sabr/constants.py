EMBED_DIM = 64
N_MPNN_LAYERS = 3

IMGT_FRAMEWORKS = {
    "FW1": list(range(1, 26)),
    "FW2": list(range(39, 56)),
    "FW3": list(range(66, 105)),
    "FW4": list(range(118, 129)),
}

# Loop definitions are inclusive
IMGT_LOOPS = {
    "CDR1": (27, 38),
    "CDR2": (56, 65),
    "CDR3": (105, 117),
}

NON_CDR_RESIDUES = sum(IMGT_FRAMEWORKS.values(), [])
CDR_RESIDUES = [x for x in range(1, 129) if x not in NON_CDR_RESIDUES]

AA_3TO1 = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}
