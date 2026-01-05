#!/usr/bin/env python3
"""Neural network modules for SAbR.

This package contains the neural network components adapted from SoftAlign:
- END_TO_END: Combined embedding and alignment model
- ENC: MPNN encoder for structure embeddings
- Smith-Waterman: Differentiable alignment algorithms

These modules use JAX and Haiku for neural network operations.
"""

from sabr.nn.encoder import ENC
from sabr.nn.end_to_end import END_TO_END
from sabr.nn.smith_waterman import sw, sw_affine

__all__ = ["END_TO_END", "ENC", "sw", "sw_affine"]
