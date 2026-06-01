#!/usr/bin/env python3
"""Input extraction for MPNN embedding computation.

This module provides functions for extracting backbone coordinates and
residue information from protein structures for MPNN embedding computation.

Supports extraction from:
- BioPython Chain objects
- PDB/mmCIF files parsed with BioPython
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
from Bio.PDB import Chain

from sabr.errors import ChainNotFoundError
from sabr.structure.io import read_structure
from sabr.structure.residues import AA_3TO1

LOGGER = logging.getLogger(__name__)
CB_BOND_LENGTH = 1.522
CB_BOND_ANGLE = 1.927
CB_DIHEDRAL = -2.143
BACKBONE_N_IDX = 0
BACKBONE_CA_IDX = 1
BACKBONE_C_IDX = 2
BACKBONE_CB_IDX = 3


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

    cb = ca_coords + (
        CB_BOND_LENGTH * np.cos(CB_BOND_ANGLE) * bc
        + CB_BOND_LENGTH * np.sin(CB_BOND_ANGLE) * np.cos(CB_DIHEDRAL) * np.cross(n, bc)
        + CB_BOND_LENGTH * np.sin(CB_BOND_ANGLE) * np.sin(CB_DIHEDRAL) * (-n)
    )
    return cb


def _get_residue_id(residue) -> str:
    res_id = residue.get_id()
    resnum = res_id[1]
    icode = res_id[2].strip()
    return f"{resnum}{icode}" if icode else str(resnum)


def _get_backbone_coords(
    residue,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    try:
        return (
            residue["N"].get_coord(),
            residue["CA"].get_coord(),
            residue["C"].get_coord(),
        )
    except KeyError:
        return None


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
        if residue.get_id()[0].strip():
            continue

        backbone = _get_backbone_coords(residue)
        if backbone is None:
            continue

        n_coord, ca_coord, c_coord = backbone

        resname = residue.get_resname()
        one_letter = AA_3TO1.get(resname, "X")
        seq_list.append(one_letter)

        cb_coord = compute_cb(
            n_coord.reshape(1, 3),
            ca_coord.reshape(1, 3),
            c_coord.reshape(1, 3),
        ).reshape(3)

        residue_coords = np.stack([n_coord, ca_coord, c_coord, cb_coord], axis=0)
        coords_list.append(residue_coords)
        ids_list.append(_get_residue_id(residue))

    chain_name = target_chain.id
    if not coords_list:
        raise ValueError(
            f"No valid residues found in chain '{chain_name}'"
            + (f" of {source_name}" if source_name else "")
        )

    coords = np.stack(coords_list, axis=0)  # [N, 4, 3]

    # Filter out residues with NaN coordinates
    valid_mask = ~np.isnan(coords).any(axis=(1, 2))
    coords = coords[valid_mask]
    ids_list = [ids_list[i] for i in range(len(ids_list)) if valid_mask[i]]
    seq_list = [seq_list[i] for i in range(len(seq_list)) if valid_mask[i]]

    n_residues = coords.shape[0]
    mask = np.ones(n_residues)
    chain_ids = np.ones(n_residues)
    residue_indices = np.arange(n_residues)

    sequence = "".join(seq_list)
    log_msg = f"Extracted {n_residues} residues from chain '{chain_name}'"
    if source_name:
        log_msg += f" in {source_name}"
    LOGGER.info(log_msg)

    return MPNNInputs(
        coords=coords[None, :],  # [1, N, 4, 3]
        mask=mask[None, :],  # [1, N]
        chain_ids=chain_ids[None, :],  # [1, N]
        residue_indices=residue_indices[None, :],  # [1, N]
        residue_ids=ids_list,
        sequence=sequence,
    )


def get_inputs(source: Union[str, Chain.Chain], chain: str | None = None) -> MPNNInputs:
    """Extract MPNN inputs from a file path or Chain object.

    Args:
        source: Either a file path (str) or BioPython Chain object.
        chain: Chain identifier to extract (only used for file paths).

    Returns:
        MPNNInputs containing backbone coordinates and residue information.
    """
    if isinstance(source, str):
        if chain is None:
            raise ValueError("A chain identifier is required for file inputs.")
        structure = read_structure(source)
        source_name = source
        model = structure[0]

        if chain in model:
            return extract_from_biopython_chain(model[chain], source_name)
        available = [ch.id for ch in model]
        raise ChainNotFoundError(
            f"Chain '{chain}' not found in {source_name}. Available chains: {available}"
        )
    return extract_from_biopython_chain(source, "")
