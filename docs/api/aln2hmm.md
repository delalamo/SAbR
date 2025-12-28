# Alignment to HMM Module

The aln2hmm module converts binary alignment matrices to HMMER-style state vectors for use with ANARCI numbering schemes.

## Overview

The module transforms alignment matrices (where rows are sequence positions and columns are IMGT positions) into state vectors that can be processed by ANARCI's numbering functions.

It handles orphan residues (e.g., CDR3 insertions) that don't map directly to any IMGT column by treating them as insertions after the previous matched position.

## Classes

::: sabr.aln2hmm.State
options:
show_root_heading: true
members: - to_tuple

## Functions

::: sabr.aln2hmm.alignment_matrix_to_state_vector
options:
show_root_heading: true

::: sabr.aln2hmm.report_output
options:
show_root_heading: true
