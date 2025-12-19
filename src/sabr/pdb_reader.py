#!/usr/bin/env python3
"""Custom PDB/CIF file reader using Biopython.

This module provides functions for reading protein structure files (PDB and CIF
formats) and extracting coordinate data for MPNN embedding generation.

Key functions:
- get_inputs_mpnn: Extract coordinates and residue info from PDB/CIF files
- get_structure: Parse a structure file with automatic format detection

The output format is compatible with the SoftAlign MPNN model interface.
"""

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
from Bio.PDB import MMCIFParser, PDBParser
from Bio.PDB.Structure import Structure

LOGGER = logging.getLogger(__name__)

# Backbone atom names to extract (N, CA, C used for CB calculation)
BACKBONE_ATOMS = ["N", "CA", "C"]

# Constants for CB position calculation (standard protein geometry)
CB_BOND_LENGTH = 1.522  # C-CA bond length in Angstroms
CB_BOND_ANGLE = 1.927  # N-CA-CB angle in radians (~110.5 degrees)
CB_DIHEDRAL = -2.143  # N-CA-C-CB dihedral angle in radians


def _np_norm(
    x: np.ndarray, axis: int = -1, keepdims: bool = True
) -> np.ndarray:
    """Compute Euclidean norm of vector with numerical stability.

    Args:
        x: Input array.
        axis: Axis along which to compute the norm.
        keepdims: Whether to keep reduced dimensions.

    Returns:
        Norm of the vector.
    """
    eps = 1e-8
    return np.sqrt(np.square(x).sum(axis=axis, keepdims=keepdims) + eps)


def _np_extend(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    length: float,
    angle: float,
    dihedral: float,
) -> np.ndarray:
    """Compute 4th coordinate given 3 coordinates and internal geometry.

    Given coordinates a-b-c, computes coordinate d where:
    - c-d has the specified length
    - b-c-d has the specified angle
    - a-b-c-d has the specified dihedral

    Args:
        a: First coordinate (N, 3).
        b: Second coordinate (N, 3).
        c: Third coordinate (N, 3).
        length: Bond length c-d in Angstroms.
        angle: Bond angle b-c-d in radians.
        dihedral: Dihedral angle a-b-c-d in radians.

    Returns:
        Fourth coordinate d with shape (N, 3).
    """

    def normalize(x: np.ndarray) -> np.ndarray:
        return x / _np_norm(x)

    bc = normalize(b - c)
    n = normalize(np.cross(b - a, bc))

    # Compute new position using bond geometry
    d = c + (
        length * np.cos(angle) * bc
        + length * np.sin(angle) * np.cos(dihedral) * np.cross(n, bc)
        + length * np.sin(angle) * np.sin(dihedral) * (-n)
    )
    return d


def _compute_cb(
    n_coords: np.ndarray, ca_coords: np.ndarray, c_coords: np.ndarray
) -> np.ndarray:
    """Compute CB (C-beta) coordinates from backbone atoms.

    Uses standard protein geometry to place the CB atom based on
    N, CA, and C positions.

    Args:
        n_coords: N atom coordinates (N, 3).
        ca_coords: CA atom coordinates (N, 3).
        c_coords: C atom coordinates (N, 3).

    Returns:
        CB coordinates with shape (N, 3).
    """
    return _np_extend(
        c_coords,
        n_coords,
        ca_coords,
        CB_BOND_LENGTH,
        CB_BOND_ANGLE,
        CB_DIHEDRAL,
    )


def get_structure(file_path: str) -> Structure:
    """Parse a structure file (PDB or CIF format).

    Automatically detects file format based on extension and uses
    the appropriate Biopython parser.

    Args:
        file_path: Path to the structure file (.pdb or .cif).

    Returns:
        Biopython Structure object.

    Raises:
        ValueError: If file format is not recognized.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".cif":
        parser = MMCIFParser(QUIET=True)
        LOGGER.debug(f"Using MMCIFParser for {file_path}")
    elif suffix == ".pdb":
        parser = PDBParser(QUIET=True)
        LOGGER.debug(f"Using PDBParser for {file_path}")
    else:
        raise ValueError(
            f"Unrecognized file format: {suffix}. Expected .pdb or .cif"
        )

    return parser.get_structure("structure", file_path)


def get_inputs_mpnn(
    file_path: str, chain: str | None = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Extract coordinates and residue info from a PDB or CIF file.

    This function provides the same interface as
    softalign.Input_MPNN.get_inputs_mpnn but uses Biopython for parsing,
    enabling support for both PDB and CIF formats.

    Args:
        file_path: Path to the structure file (.pdb or .cif).
        chain: Chain identifier to extract. If None, uses first chain.

    Returns:
        Tuple of (coords, mask, chain_ids, residue_indices, residue_ids):
        - coords: Backbone coordinates [1, N, 4, 3] (N, CA, C, CB)
        - mask: Binary mask for valid residues [1, N]
        - chain_ids: Chain identifiers (all ones) [1, N]
        - residue_indices: Sequential residue indices [1, N]
        - residue_ids: List of residue ID strings

    Raises:
        ValueError: If the specified chain is not found.
    """
    structure = get_structure(file_path)

    # Get the first model
    model = structure[0]

    # Find the target chain
    target_chain = None
    if chain is not None:
        for ch in model:
            if ch.id == chain:
                target_chain = ch
                break
        if target_chain is None:
            available = [ch.id for ch in model]
            raise ValueError(
                f"Chain '{chain}' not found in {file_path}. "
                f"Available chains: {available}"
            )
    else:
        # Use first chain
        target_chain = list(model.get_chains())[0]
        LOGGER.info(f"No chain specified, using first chain: {target_chain.id}")

    # Extract residue data
    coords_list = []
    ids_list = []

    for residue in target_chain.get_residues():
        # Skip heteroatoms (water, ligands, etc.)
        hetflag = residue.get_id()[0]
        if hetflag.strip():
            continue

        # Check if all backbone atoms are present
        try:
            n_coord = residue["N"].get_coord()
            ca_coord = residue["CA"].get_coord()
            c_coord = residue["C"].get_coord()
        except KeyError:
            # Skip residues missing backbone atoms
            continue

        # Compute CB position
        cb_coord = _compute_cb(
            n_coord.reshape(1, 3),
            ca_coord.reshape(1, 3),
            c_coord.reshape(1, 3),
        ).reshape(3)

        # Store coordinates [N, CA, C, CB]
        residue_coords = np.stack(
            [n_coord, ca_coord, c_coord, cb_coord], axis=0
        )
        coords_list.append(residue_coords)

        # Generate residue ID string
        res_id = residue.get_id()
        resnum = res_id[1]
        icode = res_id[2].strip()
        if icode:
            id_str = f"{resnum}{icode}"
        else:
            id_str = str(resnum)
        ids_list.append(id_str)

    if not coords_list:
        raise ValueError(
            f"No valid residues found in chain '{chain}' of {file_path}"
        )

    # Stack all coordinates
    coords = np.stack(coords_list, axis=0)  # [N, 4, 3]

    # Filter out any residues with NaN coordinates
    valid_mask = ~np.isnan(coords).any(axis=(1, 2))
    coords = coords[valid_mask]
    ids_list = [ids_list[i] for i in range(len(ids_list)) if valid_mask[i]]

    n_residues = coords.shape[0]

    # Create output arrays with batch dimension
    mask = np.ones(n_residues)
    chain_ids = np.ones(n_residues)
    residue_indices = np.arange(n_residues)

    LOGGER.info(
        f"Extracted {n_residues} residues from chain '{target_chain.id}' "
        f"in {file_path}"
    )

    # Add batch dimension to match softalign output format
    return (
        coords[None, :],  # [1, N, 4, 3]
        mask[None, :],  # [1, N]
        chain_ids[None, :],  # [1, N]
        residue_indices[None, :],  # [1, N]
        ids_list,
    )
