#!/usr/bin/env python3
"""PDB file modification and residue renumbering module.

This module provides functions for threading ANARCI alignments onto PDB
structures, renumbering residues according to antibody numbering schemes.

Key functions:
- thread_alignment: Main entry point for renumbering a PDB chain
- thread_onto_chain: Core renumbering logic for a single chain
- validate_output_format: Ensures mmCIF format for extended insertions

The renumbering process handles three regions:
1. PRE-Fv: Residues before the variable region (numbered backwards)
2. IN-Fv: Variable region residues (ANARCI-assigned numbers)
3. POST-Fv: Residues after the variable region (sequential numbering)
"""

import copy
import logging
from typing import Tuple

from Bio import PDB
from Bio.PDB import Chain, Model, Structure

from sabr.constants import AA_3TO1, AnarciAlignment

LOGGER = logging.getLogger(__name__)


def validate_output_format(
    output_path: str, alignment: AnarciAlignment
) -> None:
    """Validate that the output format supports the insertion codes used."""
    has_extended = any(len(icode.strip()) > 1 for (_, icode), _ in alignment)

    if has_extended and not output_path.endswith(".cif"):
        raise ValueError(
            "Extended insertion codes detected in alignment. "
            "PDB format only supports single-character insertion codes. "
            "Please use mmCIF format (.cif extension) for output."
        )


def _skip_deletions(
    anarci_idx: int,
    anarci_start: int,
    anarci_out: List[Tuple[Tuple[int, str], str]],
) -> int:
    """Advance index past any deletion positions ('-') in ANARCI output.

    Args:
        anarci_idx: Current index in the sequence alignment.
        anarci_start: Starting index of the ANARCI window.
        anarci_out: ANARCI alignment output list.

    Returns:
        Updated index after skipping any deletions.
    """
    while (
        anarci_idx - anarci_start < len(anarci_out)
        and anarci_out[anarci_idx - anarci_start][1] == "-"
    ):
        anarci_idx += 1
    return anarci_idx


def thread_onto_chain(
    chain: Chain.Chain,
    anarci_out: AnarciAlignment,
    anarci_start: int,
    anarci_end: int,
    alignment_start: int,
    max_residues: int = 0,
) -> Tuple[Chain.Chain, int]:
    """Return a deep-copied chain renumbered by the ANARCI window.

    This function handles three regions of the chain:
    1. PRE-Fv: Residues before the antibody variable region
    2. IN-Fv: Residues within the variable region (numbered by ANARCI)
    3. POST-Fv: Residues after the variable region

    Args:
        chain: BioPython Chain object to renumber.
        anarci_out: ANARCI alignment output as list of ((resnum, icode), aa).
        anarci_start: Starting position in the ANARCI window.
        anarci_end: Ending position in the ANARCI window.
        alignment_start: Offset where alignment begins in the sequence.
        max_residues: Maximum residues to process (0 = all).

    Returns:
        Tuple of (new_chain, deviation_count).
    """
    thread_msg = (
        f"Threading chain {chain.id} with ANARCI window "
        f"[{anarci_start}, {anarci_end}) "
        f"(alignment starts at {alignment_start})"
    )
    if max_residues > 0:
        thread_msg += f" (max_residues={max_residues})"
    LOGGER.info(thread_msg)
    new_chain = Chain.Chain(chain.id)

    chain_res = []

    # anarci_idx tracks position in the aligned sequence
    # It starts at -1 and increments for each aligned residue
    anarci_idx = -1
    last_idx = None
    deviations = 0

    for pdb_idx, res in enumerate(chain.get_residues()):
        # Skip residues beyond max_residues limit
        if max_residues > 0:
            res_index = res.id[1]  # Actual residue number from PDB
            if res_index > max_residues:
                LOGGER.info(
                    f"Stopping at residue index {res_index} "
                    f"(max_residues={max_residues})"
                )
                break

        # Determine if we're past the alignment start position
        is_in_aligned_region = pdb_idx >= alignment_start
        is_hetatm = res.get_id()[0].strip() != ""

        # Increment anarci_idx only for aligned, non-HETATM residues
        if is_in_aligned_region and not is_hetatm:
            anarci_idx += 1

        # Skip over deletion positions in ANARCI output
        if anarci_idx >= anarci_start:
            anarci_idx = _skip_deletions(anarci_idx, anarci_start, anarci_out)

        # Determine which region we're in
        is_in_anarci_window = anarci_idx >= anarci_start
        is_before_fv_end = anarci_idx < min(anarci_end, len(anarci_out))

        new_res = copy.deepcopy(res)
        new_res.detach_parent()

        # Compute new residue ID based on region
        if is_in_anarci_window and is_before_fv_end:
            # IN-Fv region: use ANARCI numbering
            (new_idx, icode), aa = anarci_out[anarci_idx - anarci_start]
            last_idx = new_idx

            if aa != AA_3TO1[res.get_resname()]:
                raise ValueError(f"Residue mismatch! {aa} {res.get_resname()}")
            new_id = (res.get_id()[0], new_idx, icode)
        elif anarci_idx < anarci_start:
            # PRE-Fv region: number backwards from first ANARCI position
            first_anarci_pos = anarci_out[0][0][0]
            if is_in_aligned_region:
                # Residue is in aligned sequence but before ANARCI window
                new_idx = first_anarci_pos - (anarci_start - anarci_idx)
            else:
                # Residue is before the aligned sequence entirely
                new_idx = first_anarci_pos - (
                    anarci_start + (alignment_start - pdb_idx)
                )
            new_id = (res.get_id()[0], new_idx, " ")
        else:
            # POST-Fv region: continue from last ANARCI position
            if is_hetatm:
                # Keep original ID for HETATM (water, ligands, etc.)
                new_id = res.get_id()
            else:
                last_idx += 1
                new_id = (" ", last_idx, " ")

        new_res.id = new_id
        LOGGER.info("OLD %s; NEW %s", res.get_id(), new_res.get_id())
        if res.get_id() != new_res.get_id():
            deviations += 1
        new_chain.add(new_res)
        new_res.parent = new_chain
        chain_res.append(res.get_id()[1:])
    return new_chain, deviations


def thread_alignment(
    pdb_file: str,
    chain: str,
    alignment: AnarciAlignment,
    output_pdb: str,
    start_res: int,
    end_res: int,
    alignment_start: int,
    max_residues: int = 0,
) -> int:
    """Write the renumbered chain to ``output_pdb`` and return the structure.

    Args:
        pdb_file: Path to input PDB file.
        chain: Chain identifier to renumber.
        alignment: ANARCI-style alignment list of ((resnum, icode), aa) tuples.
        output_pdb: Path to output file (.pdb or .cif).
        start_res: Start residue index from ANARCI.
        end_res: End residue index from ANARCI.
        alignment_start: Offset where alignment begins in the sequence.
        max_residues: Maximum number of residues to process. If 0,
            process all residues.

    Returns:
        Number of residue ID deviations from original numbering.

    Raises:
        ValueError: If extended insertion codes are used but output is not .cif.
    """
    # Validate output format supports the insertion codes used
    validate_output_format(output_pdb, alignment)

    align_msg = (
        f"Threading alignment for {pdb_file} chain {chain}; "
        f"writing to {output_pdb}"
    )
    LOGGER.info(align_msg)
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("input_structure", pdb_file)
    new_structure = Structure.Structure("threaded_structure")
    new_model = Model.Model(0)

    all_devs = 0

    for ch in structure[0]:
        if ch.id != chain:
            new_model.add(ch)
        else:
            new_chain, deviations = thread_onto_chain(
                ch, alignment, start_res, end_res, alignment_start, max_residues
            )
            new_model.add(new_chain)
            all_devs += deviations

    new_structure.add(new_model)
    io = PDB.PDBIO()
    if output_pdb.endswith(".cif"):
        io = PDB.MMCIFIO()
        LOGGER.debug("Detected CIF output; using MMCIFIO")
    io.set_structure(new_structure)
    io.save(output_pdb)
    LOGGER.info(f"Saved threaded structure to {output_pdb}")
    return all_devs
