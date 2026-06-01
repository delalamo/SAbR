"""Protein-structure geometry constants."""

# Backbone CB reconstruction parameters.
CB_BOND_LENGTH = 1.522
CB_BOND_ANGLE = 1.927
CB_DIHEDRAL = -2.143

# Peptide bond geometry.
PEPTIDE_BOND_LENGTH = 1.33
PEPTIDE_BOND_MAX_DISTANCE = 2 * PEPTIDE_BOND_LENGTH

# Backbone coordinate channel order for MPNN inputs: [N, CA, C, computed CB].
BACKBONE_N_IDX = 0
BACKBONE_CA_IDX = 1
BACKBONE_C_IDX = 2
BACKBONE_CB_IDX = 3
