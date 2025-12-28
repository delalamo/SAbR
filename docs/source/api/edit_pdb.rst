Edit PDB Module
===============

.. module:: sabr.edit_pdb
   :synopsis: Structure file modification and residue renumbering.

The edit_pdb module provides functions for threading ANARCI alignments onto
protein structures, renumbering residues according to antibody numbering
schemes.

Overview
--------

The renumbering process handles three regions:

1. **PRE-Fv**: Residues before the variable region (numbered backwards)
2. **IN-Fv**: Variable region residues (ANARCI-assigned numbers)
3. **POST-Fv**: Residues after the variable region (sequential numbering)

Supported file formats:

- Input: PDB (``.pdb``) and mmCIF (``.cif``)
- Output: PDB (``.pdb``) and mmCIF (``.cif``)

Functions
---------

.. autofunction:: sabr.edit_pdb.thread_alignment

.. autofunction:: sabr.edit_pdb.thread_onto_chain

.. autofunction:: sabr.edit_pdb.validate_output_format

Internal Functions
------------------

.. autofunction:: sabr.edit_pdb._skip_deletions
