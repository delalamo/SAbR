# SoftAligner Module

The softaligner module provides the core alignment functionality for SAbR. It aligns query antibody embeddings against unified reference embeddings to generate IMGT-compatible alignments.

## Overview

The alignment process includes:

1. Embedding comparison against unified reference
2. Deterministic corrections for CDR loops, DE loop, FR1, and C-terminus
3. Expansion to full 128-position IMGT alignment matrix
4. Chain type detection from DE loop occupancy

## Classes

::: sabr.softaligner.SoftAligner
    options:
      show_root_heading: true
      members:
        - __init__
        - __call__
        - normalize
        - read_embeddings
        - fix_aln
        - correct_gap_numbering
        - correct_fr1_alignment
        - correct_fr3_alignment
        - correct_c_terminus

## Functions

::: sabr.softaligner.find_nearest_occupied_column
    options:
      show_root_heading: true
