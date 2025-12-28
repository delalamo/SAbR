SoftAligner Module
==================

.. module:: sabr.softaligner
   :synopsis: SoftAlign-based antibody sequence alignment.

The softaligner module provides the core alignment functionality for SAbR.
It aligns query antibody embeddings against unified reference embeddings
to generate IMGT-compatible alignments.

Overview
--------

The alignment process includes:

1. Embedding comparison against unified reference
2. Deterministic corrections for CDR loops, DE loop, FR1, and C-terminus
3. Expansion to full 128-position IMGT alignment matrix
4. Chain type detection from DE loop occupancy

Classes
-------

.. autoclass:: sabr.softaligner.SoftAligner
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__, __call__

Functions
---------

.. autofunction:: sabr.softaligner.find_nearest_occupied_column

Internal Functions
------------------

.. autofunction:: sabr.softaligner._align_fn
