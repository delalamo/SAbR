Alignment to HMM Module
=======================

.. module:: sabr.aln2hmm
   :synopsis: Convert alignment matrices to HMMER-style state vectors.

The aln2hmm module converts binary alignment matrices to HMMER-style state
vectors for use with ANARCI numbering.

Overview
--------

The module transforms alignment matrices (where rows are sequence positions
and columns are IMGT positions) into state vectors that can be processed
by ANARCI's numbering functions.

It handles orphan residues (e.g., CDR3 insertions) that don't map directly
to any IMGT column by treating them as insertions after the previous
matched position.

Classes
-------

.. autoclass:: sabr.aln2hmm.State
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__, __iter__, __getitem__

Functions
---------

.. autofunction:: sabr.aln2hmm.alignment_matrix_to_state_vector

.. autofunction:: sabr.aln2hmm.report_output
