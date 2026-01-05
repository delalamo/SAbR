#!/usr/bin/env python3
"""Type aliases for SAbR.

This module defines type aliases used throughout the SAbR package
for improved code readability and type checking.
"""

from typing import List, Tuple

# Type alias for ANARCI alignment output:
# list of ((residue_number, insertion_code), amino_acid)
AnarciAlignment = List[Tuple[Tuple[int, str], str]]
