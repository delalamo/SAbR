#!/usr/bin/env python3
"""Utility functions for SAbR.

This module provides helper functions for:
- Extracting sequences from structure files (PDB and CIF formats)
- Loading SoftAlign model parameters
"""

import logging
from importlib.resources import files
from pathlib import Path
from typing import Any, Dict

import numpy as np
from Bio import SeqIO
from Bio.PDB import MMCIFParser
from softalign.utils import convert_numpy_to_jax, unflatten_dict

from sabr.constants import AA_3TO1

LOGGER = logging.getLogger(__name__)


def fetch_sequence_from_pdb(pdb_file: str, chain: str) -> str:
    """Return the sequence for chain in pdb_file without X residues.

    Supports both PDB and mmCIF file formats. Format is auto-detected
    based on file extension.

    Args:
        pdb_file: Path to the structure file (.pdb or .cif).
        chain: Chain identifier to extract sequence from.

    Returns:
        Amino acid sequence as a string (without X residues).

    Raises:
        ValueError: If the specified chain is not found.
    """
    path = Path(pdb_file)
    suffix = path.suffix.lower()

    # Use appropriate parser based on file extension
    if suffix == ".cif":
        return _fetch_sequence_from_cif(pdb_file, chain)
    else:
        # Default to PDB format (including .pdb extension)
        return _fetch_sequence_from_pdb_format(pdb_file, chain)


def _fetch_sequence_from_pdb_format(pdb_file: str, chain: str) -> str:
    """Extract sequence from a PDB format file using SeqIO."""
    for record in SeqIO.parse(pdb_file, "pdb-atom"):
        if record.id.endswith(chain):
            return str(record.seq).replace("X", "")
    ids = [r.id for r in SeqIO.parse(pdb_file, "pdb-atom")]
    raise ValueError(f"Chain {chain} not found in {pdb_file} (contains {ids})")


def _fetch_sequence_from_cif(cif_file: str, chain: str) -> str:
    """Extract sequence from a CIF format file using Biopython parser."""
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("structure", cif_file)

    # Find the target chain
    for model in structure:
        for ch in model:
            if ch.id == chain:
                # Build sequence from residues
                seq_parts = []
                for residue in ch.get_residues():
                    # Skip heteroatoms
                    if residue.get_id()[0].strip():
                        continue
                    resname = residue.get_resname()
                    if resname in AA_3TO1:
                        seq_parts.append(AA_3TO1[resname])
                    # Unknown residue types are skipped (like X removal)
                return "".join(seq_parts)

    # Chain not found
    available = []
    for model in structure:
        for ch in model:
            available.append(ch.id)
    raise ValueError(
        f"Chain {chain} not found in {cif_file} (contains {available})"
    )


def read_softalign_params(
    params_name: str = "CONT_SW_05_T_3_1",
    params_path: str = "softalign.models",
) -> Dict[str, Any]:
    """Load SoftAlign parameters from package resources.

    Args:
        params_name: Name of the model parameters file (without extension).
        params_path: Package path containing the parameters file.

    Returns:
        Dictionary containing the model parameters as JAX arrays.
    """
    package_files = files(params_path)
    npz_path = package_files / f"{params_name}.npz"

    with open(npz_path, "rb") as f:
        data = dict(np.load(f, allow_pickle=False))

    # Unflatten the dictionary structure and convert to JAX arrays
    params = unflatten_dict(data)
    params = convert_numpy_to_jax(params)
    LOGGER.info(f"Loaded model parameters from {npz_path}")
    return params
