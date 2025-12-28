MPNN Embeddings Module
======================

.. module:: sabr.mpnn_embeddings
   :synopsis: MPNN embedding generation and management.

The mpnn_embeddings module provides functionality for generating and managing
neural network embeddings from protein structures using the MPNN (Message
Passing Neural Network) architecture.

Overview
--------

Embeddings are 64-dimensional vectors computed for each residue, capturing
structural and sequence features for alignment. The module supports both
PDB (``.pdb``) and mmCIF (``.cif``) file formats.

Classes
-------

MPNNInputs
~~~~~~~~~~

.. autoclass:: sabr.mpnn_embeddings.MPNNInputs
   :members:
   :undoc-members:
   :show-inheritance:

MPNNEmbeddings
~~~~~~~~~~~~~~

.. autoclass:: sabr.mpnn_embeddings.MPNNEmbeddings
   :members:
   :undoc-members:
   :show-inheritance:

Functions
---------

.. autofunction:: sabr.mpnn_embeddings.from_pdb

.. autofunction:: sabr.mpnn_embeddings.from_npz

Internal Functions
------------------

.. autofunction:: sabr.mpnn_embeddings._embed_pdb

.. autofunction:: sabr.mpnn_embeddings._get_inputs_mpnn

.. autofunction:: sabr.mpnn_embeddings._get_structure

.. autofunction:: sabr.mpnn_embeddings._compute_cb

.. autofunction:: sabr.mpnn_embeddings._np_extend

.. autofunction:: sabr.mpnn_embeddings._np_norm
