#!/usr/bin/env python3
"""Command-line interface for SAbR.

This package provides the CLI entry point and option validation.
"""

from sabr.cli.main import main
from sabr.cli.options import normalize_chain_type, validate_inputs

__all__ = ["main", "normalize_chain_type", "validate_inputs"]
