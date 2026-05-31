#!/usr/bin/env python3
"""Input extraction for MPNN embedding computation.

This module provides functions for extracting backbone coordinates and
residue information from protein structures for MPNN embedding computation.

Supports extraction from:
- BioPython Chain objects
- PDB/mmCIF files parsed with BioPython
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple, Union

import numpy as np
from Bio.PDB import Chain

from sabr import constants
from sabr.errors import ChainNotFoundError
from sabr.structure.io import read_structure

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


class ResidueAdapter(ABC):
    """Abstract adapter for accessing residue data from different libraries.

    This adapter provides a uniform interface for extracting backbone
    coordinates and residue information from BioPython chain objects.
    """

    @abstractmethod
    def get_chain_name(self) -> str:
        """Return the chain identifier."""

    @abstractmethod
    def iterate_residues(self) -> Iterator:
        """Iterate over residues in the chain."""

    @abstractmethod
    def is_heteroatom(self, residue) -> bool:
        """Check if residue is a heteroatom (water, ligand, etc.)."""

    @abstractmethod
    def get_backbone_coords(
        self, residue
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Extract N, CA, C coordinates from residue.

        Returns:
            Tuple of (N, CA, C) coordinates, or None if backbone incomplete.
        """

    @abstractmethod
    def get_residue_name(self, residue) -> str:
        """Return 3-letter residue name."""

    @abstractmethod
    def get_residue_id(self, residue) -> str:
        """Return residue ID string (number + optional insertion code)."""


class BioPythonResidueAdapter(ResidueAdapter):
    """Adapter for BioPython Chain objects."""

    def __init__(self, chain: Chain.Chain):
        self._chain = chain

    def get_chain_name(self) -> str:
        return self._chain.id

    def iterate_residues(self) -> Iterator:
        return self._chain.get_residues()

    def is_heteroatom(self, residue) -> bool:
        hetflag = residue.get_id()[0]
        return bool(hetflag.strip())

    def get_backbone_coords(
        self, residue
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        try:
            n_coord = residue["N"].get_coord()
            ca_coord = residue["CA"].get_coord()
            c_coord = residue["C"].get_coord()
            return n_coord, ca_coord, c_coord
        except KeyError:
            return None

    def get_residue_name(self, residue) -> str:
        return residue.get_resname()

    def get_residue_id(self, residue) -> str:
        res_id = residue.get_id()
        resnum = res_id[1]
        icode = res_id[2].strip()
        if icode:
            return f"{resnum}{icode}"
        return str(resnum)


def _extract_from_chain(adapter: ResidueAdapter, source_name: str = "") -> MPNNInputs:
    """Extract backbone coordinates and residue info using a ResidueAdapter.

    This is the shared extraction logic used by BioPython extraction functions.

    Args:
        adapter: ResidueAdapter wrapping the chain to extract from.
        source_name: Source identifier for logging (file path or "structure").

    Returns:
        MPNNInputs containing backbone coordinates and residue information.

    Raises:
        ValueError: If no valid residues are found.
    """
    coords_list = []
    ids_list = []
    seq_list = []

    for residue in adapter.iterate_residues():
        if adapter.is_heteroatom(residue):
            continue

        backbone = adapter.get_backbone_coords(residue)
        if backbone is None:
            continue

        n_coord, ca_coord, c_coord = backbone

        resname = adapter.get_residue_name(residue)
        one_letter = constants.AA_3TO1.get(resname, "X")
        seq_list.append(one_letter)

        cb_coord = compute_cb(
            n_coord.reshape(1, 3),
            ca_coord.reshape(1, 3),
            c_coord.reshape(1, 3),
        ).reshape(3)

        residue_coords = np.stack([n_coord, ca_coord, c_coord, cb_coord], axis=0)
        coords_list.append(residue_coords)
        ids_list.append(adapter.get_residue_id(residue))

    chain_name = adapter.get_chain_name()
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
    adapter = BioPythonResidueAdapter(target_chain)
    return _extract_from_chain(adapter, source_name)


def get_inputs(source: Union[str, Chain.Chain], chain: str | None = None) -> MPNNInputs:
    """Extract MPNN inputs from a file path or Chain object.

    Args:
        source: Either a file path (str) or BioPython Chain object.
        chain: Chain identifier to extract (only used for file paths).

    Returns:
        MPNNInputs containing backbone coordinates and residue information.
    """
    if isinstance(source, str):
        structure = read_structure(source)
        source_name = source
        model = structure[0]

        if chain is not None:
            if chain in model:
                return extract_from_biopython_chain(model[chain], source_name)
            available = [ch.id for ch in model]
            raise ChainNotFoundError(
                f"Chain '{chain}' not found in {source_name}. "
                f"Available chains: {available}"
            )
        else:
            target_chain = next(iter(model))
            LOGGER.info(f"No chain specified, using first chain: {target_chain.id}")
            return extract_from_biopython_chain(target_chain, source_name)
    else:
        return extract_from_biopython_chain(source, "")
