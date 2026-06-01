#!/usr/bin/env python3
"""Embedding generation for SAbR.

This package provides the embedding functionality for SAbR including:
- QueryEmbeddings: Dataclass for per-residue query embeddings
- EmbeddingBackend: JAX/Haiku backend for embedding computation
- Input extraction from PDB/CIF files and BioPython chains
"""
