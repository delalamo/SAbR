#!/usr/bin/env python3
"""Command-line interface for SAbR.

This package provides the CLI entry point, option validation, and
programmatic renumbering functions.
"""

from sabr.cli import renumber
from sabr.cli.main import main
from sabr.cli.options import normalize_chain_type, validate_inputs

__all__ = ["main", "renumber", "normalize_chain_type", "validate_inputs"]
