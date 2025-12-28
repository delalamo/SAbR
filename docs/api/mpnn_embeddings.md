# MPNN Embeddings Module

The mpnn_embeddings module provides functionality for generating and managing neural network embeddings from protein structures using the MPNN (Message Passing Neural Network) architecture.

## Overview

Embeddings are 64-dimensional vectors computed for each residue, capturing structural and sequence features for alignment. The module supports both PDB (`.pdb`) and mmCIF (`.cif`) file formats.

## Classes

### MPNNInputs

::: sabr.mpnn_embeddings.MPNNInputs
    options:
      show_root_heading: true

### MPNNEmbeddings

::: sabr.mpnn_embeddings.MPNNEmbeddings
    options:
      show_root_heading: true
      members:
        - save

## Functions

::: sabr.mpnn_embeddings.from_pdb
    options:
      show_root_heading: true

::: sabr.mpnn_embeddings.from_npz
    options:
      show_root_heading: true
