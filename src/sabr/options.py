#!/usr/bin/env python3
"""CLI option validation for SAbR.

This module provides validation functions for command-line arguments.
"""

import os
from typing import Tuple

import click


def normalize_chain_type(chain_type: str) -> str:
    """Normalize chain type abbreviation to single letter.

    Converts verbose chain type names to their single-letter equivalents
    used by ANARCI. Handles case-insensitive input.

    Args:
        chain_type: Chain type string (e.g., "heavy", "H", "kappa", "auto").

    Returns:
        Normalized chain type: "H", "K", "L", or "auto".
    """
    chain_map = {"heavy": "H", "kappa": "K", "lambda": "L"}
    normalized = chain_map.get(chain_type.lower())
    if normalized:
        return normalized
    upper = chain_type.upper()
    if upper in ("H", "K", "L"):
        return upper
    return chain_type  # Return as-is for "auto" or unknown values


def validate_inputs(
    input_pdb: str,
    input_chain: str,
    output_file: str,
    residue_range: Tuple[int, int],
    extended_insertions: bool,
    overwrite: bool,
) -> None:
    """Validate CLI inputs and raise ClickException on failure.

    Args:
        input_pdb: Path to input structure file.
        input_chain: Chain identifier (single character).
        output_file: Path to output structure file.
        residue_range: Tuple of (start, end) residue numbers in PDB numbering.
            Use (0, 0) to process all residues.
        extended_insertions: Whether extended insertion codes are enabled.
        overwrite: Whether to overwrite existing output file.

    Raises:
        click.ClickException: If any validation fails.
    """
    if not os.path.exists(input_pdb):
        raise click.ClickException(f"Input file '{input_pdb}' does not exist.")

    if not input_pdb.lower().endswith((".pdb", ".cif")):
        raise click.ClickException(
            f"Input file must be a PDB (.pdb) or mmCIF (.cif) file. "
            f"Got: '{input_pdb}'"
        )

    if input_chain and len(input_chain) != 1:
        raise click.ClickException(
            f"Chain identifier must be a single character. Got: '{input_chain}'"
        )

    if not output_file.lower().endswith((".pdb", ".cif")):
        raise click.ClickException(
            f"Output file must have extension .pdb or .cif. "
            f"Got: '{output_file}'"
        )

    if extended_insertions and not output_file.endswith(".cif"):
        raise click.ClickException(
            "The --extended-insertions option requires mmCIF output format. "
            "Please use a .cif file extension for the output file."
        )

    start_res, end_res = residue_range
    if residue_range != (0, 0):
        if end_res <= start_res:
            raise click.ClickException(
                f"Invalid residue range: end ({end_res}) must be greater than "
                f"start ({start_res}). Use '0 0' to process all residues."
            )
        if start_res < 0 or end_res < 0:
            raise click.ClickException(
                f"Residue range values must be non-negative. "
                f"Got: start={start_res}, end={end_res}"
            )

    if os.path.exists(output_file) and not overwrite:
        raise click.ClickException(
            f"{output_file} exists, rerun with --overwrite to replace it"
        )
