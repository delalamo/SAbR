Constants Module
================

.. module:: sabr.constants
   :synopsis: Constants and configuration values for SAbR.

The constants module defines constants used throughout the SAbR package.

Type Aliases
------------

.. py:data:: AnarciAlignment
   :type: List[Tuple[Tuple[int, str], str]]

   Type alias for ANARCI alignment output.
   Each element is a tuple of ``((residue_number, insertion_code), amino_acid)``.

Neural Network Configuration
----------------------------

.. py:data:: EMBED_DIM
   :value: 64

   Dimension of MPNN embeddings.

.. py:data:: N_MPNN_LAYERS
   :value: 3

   Number of layers in the MPNN model.

IMGT Numbering
--------------

.. py:data:: IMGT_MAX_POSITION
   :value: 128

   Maximum position in IMGT numbering scheme.

.. py:data:: DEFAULT_TEMPERATURE
   :value: 1e-4

   Default alignment temperature for SoftAlign.

.. py:data:: IMGT_FRAMEWORKS
   :type: dict

   Dictionary mapping framework region names to their IMGT position lists::

       {
           "FW1": [1, 2, ..., 26],
           "FW2": [39, 40, ..., 55],
           "FW3": [66, 67, ..., 104],
           "FW4": [118, 119, ..., 128],
       }

.. py:data:: IMGT_LOOPS
   :type: dict

   Dictionary mapping CDR names to (start, end) position tuples::

       {
           "CDR1": (27, 38),
           "CDR2": (56, 65),
           "CDR3": (105, 117),
       }

.. py:data:: CDR_ANCHORS
   :type: dict

   Framework anchor positions used for CDR renumbering::

       {
           "CDR1": (23, 40),   # Cys23 and position 40
           "CDR2": (54, 66),   # Position 54 and 66
           "CDR3": (104, 118), # Position 104 and 118
       }

FR1 Region Constants
--------------------

.. py:data:: FR1_ANCHOR_START_COL
   :value: 5

   0-indexed column for IMGT position 6.

.. py:data:: FR1_ANCHOR_END_COL
   :value: 11

   0-indexed column for IMGT position 12.

.. py:data:: FR1_POSITION_10_COL
   :value: 9

   0-indexed column for IMGT position 10.

.. py:data:: FR1_KAPPA_RESIDUE_COUNT
   :value: 7

   Kappa chains have 7 residues in positions 6-12.

DE Loop Constants
-----------------

.. py:data:: DE_LOOP_START_COL
   :value: 78

   0-indexed column for IMGT position 79.

.. py:data:: DE_LOOP_END_COL
   :value: 83

   0-indexed column for IMGT position 84.

.. py:data:: DE_LOOP_HEAVY_THRESHOLD
   :value: 5

   Number of residues indicating heavy chain (>= 5).

FR3 Position Constants
----------------------

.. py:data:: FR3_POS81_COL
   :value: 80

.. py:data:: FR3_POS82_COL
   :value: 81

.. py:data:: FR3_POS83_COL
   :value: 82

.. py:data:: FR3_POS84_COL
   :value: 83

C-Terminus Constants
--------------------

.. py:data:: C_TERMINUS_ANCHOR_POSITION
   :value: 124

   0-indexed position for IMGT position 125.

Amino Acid Mapping
------------------

.. py:data:: AA_3TO1
   :type: dict

   Dictionary mapping 3-letter amino acid codes to 1-letter codes::

       {
           "ALA": "A", "CYS": "C", "ASP": "D", ...
       }
