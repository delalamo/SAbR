# Edit PDB Module

The edit_pdb module provides functions for threading ANARCI alignments onto protein structures, renumbering residues according to antibody numbering schemes.

## Overview

The renumbering process handles three regions:

1. **PRE-Fv**: Residues before the variable region (numbered backwards)
2. **IN-Fv**: Variable region residues (ANARCI-assigned numbers)
3. **POST-Fv**: Residues after the variable region (sequential numbering)

**Supported file formats:**

- Input: PDB (`.pdb`) and mmCIF (`.cif`)
- Output: PDB (`.pdb`) and mmCIF (`.cif`)

## Functions

::: sabr.edit_pdb.thread_alignment
options:
show_root_heading: true

::: sabr.edit_pdb.thread_onto_chain
options:
show_root_heading: true

::: sabr.edit_pdb.validate_output_format
options:
show_root_heading: true
