#!/usr/bin/env python3
"""Input extraction for MPNN embedding computation.

This module provides functions for extracting backbone coordinates and
residue information from protein structures for MPNN embedding computation.

Supports extraction from:
- PDB/mmCIF files (via Gemmi)
- BioPython Chain objects
"""

import logging
from dataclasses import dataclass
from typing import List, Union

import gemmi
import numpy as np
from Bio.PDB import Chain

from sabr import constants

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class MPNNInputs:
    """Input data for MPNN embedding computation.

    Contains backbone coordinates and residue information extracted
    from a PDB or CIF structure file.

    Attributes:
        coords: Backbone coordinates [1, N, 4, 3] (N, CA, C, CB).
        mask: Binary mask for valid residues [1, N].
        chain_ids: Chain identifiers (all ones) [1, N].
        residue_indices: Sequential residue indices [1, N].
        residue_ids: List of residue ID strings.
        sequence: Amino acid sequence as one-letter codes.
    """

    coords: np.ndarray
    mask: np.ndarray
    chain_ids: np.ndarray
    residue_indices: np.ndarray
    residue_ids: List[str]
    sequence: str


def compute_cb(
    n_coords: np.ndarray, ca_coords: np.ndarray, c_coords: np.ndarray
) -> np.ndarray:
    """Compute CB (C-beta) coordinates from backbone atoms.

    Uses standard protein geometry constants to calculate the CB position
    from N, CA, and C backbone atom coordinates.

    Args:
        n_coords: N atom coordinates [1, 3] or [3].
        ca_coords: CA atom coordinates [1, 3] or [3].
        c_coords: C atom coordinates [1, 3] or [3].

    Returns:
        CB coordinates with same shape as input.
    """
    eps = 1e-8

    def normalize(x: np.ndarray) -> np.ndarray:
        norm = np.sqrt(np.square(x).sum(axis=-1, keepdims=True) + eps)
        return x / norm

    # Compute CB position using internal geometry
    # a=C, b=N, c=CA for the extension calculation
    bc = normalize(n_coords - ca_coords)
    n = normalize(np.cross(n_coords - c_coords, bc))

    length = constants.CB_BOND_LENGTH
    angle = constants.CB_BOND_ANGLE
    dihedral = constants.CB_DIHEDRAL

    cb = ca_coords + (
        length * np.cos(angle) * bc
        + length * np.sin(angle) * np.cos(dihedral) * np.cross(n, bc)
        + length * np.sin(angle) * np.sin(dihedral) * (-n)
    )
    return cb


def extract_from_gemmi_chain(
    chain: gemmi.Chain, source_name: str = ""
) -> MPNNInputs:
    """Extract coordinates, residue info, and sequence from a Gemmi Chain.

    Args:
        chain: Gemmi Chain object to extract from.
        source_name: Source identifier for logging (file path or "structure").

    Returns:
        MPNNInputs containing backbone coordinates and residue information.

    Raises:
        ValueError: If no valid residues are found.
    """
    coords_list = []
    ids_list = []
    seq_list = []

    for residue in chain:
        # Skip heteroatoms (water, ligands, etc.)
        # het_flag: 'A' = amino acid, 'H' = HETATM, 'W' = water
        if residue.het_flag != "A":
            continue

        # Check if all backbone atoms are present
        n_atom = residue.find_atom("N", "*")
        ca_atom = residue.find_atom("CA", "*")
        c_atom = residue.find_atom("C", "*")

        if not (n_atom and ca_atom and c_atom):
            # Skip residues missing backbone atoms
            continue

        n_coord = np.array([n_atom.pos.x, n_atom.pos.y, n_atom.pos.z])
        ca_coord = np.array([ca_atom.pos.x, ca_atom.pos.y, ca_atom.pos.z])
        c_coord = np.array([c_atom.pos.x, c_atom.pos.y, c_atom.pos.z])

        # Extract one-letter amino acid code (X for unknown residues)
        resname = residue.name
        one_letter = constants.AA_3TO1.get(resname, "X")
        seq_list.append(one_letter)

        # Compute CB position
        cb_coord = compute_cb(
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
        resnum = residue.seqid.num
        icode = residue.seqid.icode.strip()
        if icode and icode != " ":
            id_str = f"{resnum}{icode}"
        else:
            id_str = str(resnum)
        ids_list.append(id_str)

    if not coords_list:
        raise ValueError(
            f"No valid residues found in chain '{chain.name}'"
            + (f" of {source_name}" if source_name else "")
        )

    # Stack all coordinates
    coords = np.stack(coords_list, axis=0)  # [N, 4, 3]

    # Filter out any residues with NaN coordinates
    valid_mask = ~np.isnan(coords).any(axis=(1, 2))
    coords = coords[valid_mask]
    ids_list = [ids_list[i] for i in range(len(ids_list)) if valid_mask[i]]
    seq_list = [seq_list[i] for i in range(len(seq_list)) if valid_mask[i]]

    n_residues = coords.shape[0]

    # Create output arrays with batch dimension
    mask = np.ones(n_residues)
    chain_ids = np.ones(n_residues)
    residue_indices = np.arange(n_residues)

    sequence = "".join(seq_list)
    log_msg = f"Extracted {n_residues} residues from chain '{chain.name}'"
    if source_name:
        log_msg += f" in {source_name}"
    LOGGER.info(log_msg)

    # Add batch dimension to match expected format
    return MPNNInputs(
        coords=coords[None, :],  # [1, N, 4, 3]
        mask=mask[None, :],  # [1, N]
        chain_ids=chain_ids[None, :],  # [1, N]
        residue_indices=residue_indices[None, :],  # [1, N]
        residue_ids=ids_list,
        sequence=sequence,
    )


def extract_from_biopython_chain(
    target_chain: Chain.Chain, source_name: str = ""
) -> MPNNInputs:
    """Extract coordinates, residue info, and sequence from a BioPython Chain.

    Args:
        target_chain: BioPython Chain object to extract from.
        source_name: Source identifier for logging (file path or "structure").

    Returns:
        MPNNInputs containing backbone coordinates and residue information.

    Raises:
        ValueError: If no valid residues are found.
    """
    coords_list = []
    ids_list = []
    seq_list = []

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

        # Extract one-letter amino acid code (X for unknown residues)
        resname = residue.get_resname()
        one_letter = constants.AA_3TO1.get(resname, "X")
        seq_list.append(one_letter)

        # Compute CB position
        cb_coord = compute_cb(
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
            f"No valid residues found in chain '{target_chain.id}'"
            + (f" of {source_name}" if source_name else "")
        )

    # Stack all coordinates
    coords = np.stack(coords_list, axis=0)  # [N, 4, 3]

    # Filter out any residues with NaN coordinates
    valid_mask = ~np.isnan(coords).any(axis=(1, 2))
    coords = coords[valid_mask]
    ids_list = [ids_list[i] for i in range(len(ids_list)) if valid_mask[i]]
    seq_list = [seq_list[i] for i in range(len(seq_list)) if valid_mask[i]]

    n_residues = coords.shape[0]

    # Create output arrays with batch dimension
    mask = np.ones(n_residues)
    chain_ids = np.ones(n_residues)
    residue_indices = np.arange(n_residues)

    sequence = "".join(seq_list)
    log_msg = f"Extracted {n_residues} residues from chain '{target_chain.id}'"
    if source_name:
        log_msg += f" in {source_name}"
    LOGGER.info(log_msg)

    # Add batch dimension to match expected format
    return MPNNInputs(
        coords=coords[None, :],  # [1, N, 4, 3]
        mask=mask[None, :],  # [1, N]
        chain_ids=chain_ids[None, :],  # [1, N]
        residue_indices=residue_indices[None, :],  # [1, N]
        residue_ids=ids_list,
        sequence=sequence,
    )


def get_inputs(
    source: Union[str, Chain.Chain], chain: str | None = None
) -> MPNNInputs:
    """Extract MPNN inputs from a file path or Chain object.

    Args:
        source: Either a file path (str) or BioPython Chain object.
        chain: Chain identifier to extract (only used for file paths).

    Returns:
        MPNNInputs containing backbone coordinates and residue information.
    """
    if isinstance(source, str):
        # Use Gemmi for fast file parsing
        structure = gemmi.read_structure(source)
        source_name = source
        model = structure[0]

        if chain is not None:
            for ch in model:
                if ch.name == chain:
                    return extract_from_gemmi_chain(ch, source_name)
            available = [ch.name for ch in model]
            raise ValueError(
                f"Chain '{chain}' not found in {source_name}. "
                f"Available chains: {available}"
            )
        else:
            # Use first chain
            target_chain = list(model)[0]
            LOGGER.info(
                f"No chain specified, using first chain: {target_chain.name}"
            )
            return extract_from_gemmi_chain(target_chain, source_name)
    else:
        # source is a BioPython Chain object
        return extract_from_biopython_chain(source, "")
