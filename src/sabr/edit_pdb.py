#!/usr/bin/env python3
"""Structure file modification and residue renumbering module.

This module provides functions for threading ANARCI alignments onto protein
structures, renumbering residues according to antibody numbering schemes.

Key functions:
- thread_alignment: Main entry point for renumbering a structure chain
- thread_onto_chain: Core renumbering logic for a single chain
- validate_output_format: Ensures mmCIF format for extended insertions

Supported file formats:
- Input: PDB (.pdb) and mmCIF (.cif)
- Output: PDB (.pdb) and mmCIF (.cif)

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
    anarci_out: AnarciAlignment,
) -> int:
    """Advance index past any deletion positions ('-') in ANARCI output.

    Args:
        anarci_idx: Current 0-indexed count of aligned residues.
        anarci_start: First index in anarci_out with actual residue.
        anarci_out: ANARCI alignment output list.

    Returns:
        Updated index after skipping any deletions.
    """
    anarci_array_idx = anarci_idx + anarci_start
    while (
        anarci_array_idx < len(anarci_out)
        and anarci_out[anarci_array_idx][1] == "-"
    ):
        anarci_idx += 1
        anarci_array_idx = anarci_idx + anarci_start
    return anarci_idx


def thread_onto_chain(
    chain: Chain.Chain,
    anarci_out: AnarciAlignment,
    anarci_start: int,
    anarci_end: int,
    alignment_start: int,
    residue_range: Tuple[int, int] = (0, 0),
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
        residue_range: Tuple of (start, end) residue numbers to process
            (inclusive). Use (0, 0) to process all residues.

    Returns:
        Tuple of (new_chain, deviation_count).
    """
    start_res, end_res = residue_range
    range_str = (
        f" (residue_range={start_res}-{end_res})"
        if residue_range != (0, 0)
        else ""
    )
    LOGGER.info(
        f"Threading chain {chain.id} with ANARCI window "
        f"[{anarci_start}, {anarci_end}) (alignment_start={alignment_start})"
        + range_str
    )
    new_chain = Chain.Chain(chain.id)
    aligned_residue_idx = -1
    last_imgt_pos = None
    deviations = 0

    for pdb_idx, res in enumerate(chain.get_residues()):
        res_num = res.id[1]
        # Skip residues outside the specified range
        if residue_range != (0, 0):
            if res_num < start_res:
                continue
            if res_num > end_res:
                LOGGER.info(
                    f"Stopping at residue {res_num} (end of range {end_res})"
                )
                break

        is_in_aligned_region = pdb_idx >= alignment_start
        is_hetatm = res.get_id()[0].strip() != ""

        if is_in_aligned_region and not is_hetatm:
            aligned_residue_idx += 1

        if aligned_residue_idx >= 0:
            aligned_residue_idx = _skip_deletions(
                aligned_residue_idx, anarci_start, anarci_out
            )

        anarci_array_idx = aligned_residue_idx + anarci_start
        is_in_fv_region = aligned_residue_idx >= 0
        is_before_fv_end = anarci_array_idx < len(anarci_out)

        new_res = copy.deepcopy(res)
        new_res.detach_parent()

        if is_in_fv_region and is_before_fv_end and not is_hetatm:
            (new_imgt_pos, icode), aa = anarci_out[anarci_array_idx]
            last_imgt_pos = new_imgt_pos
            if aa != AA_3TO1[res.get_resname()]:
                raise ValueError(f"Residue mismatch! {aa} {res.get_resname()}")
            new_id = (res.get_id()[0], new_imgt_pos, icode)
        elif is_hetatm:
            new_id = res.get_id()
        elif aligned_residue_idx < 0:
            first_anarci_pos = anarci_out[anarci_start][0][0]
            new_imgt_pos = first_anarci_pos - (alignment_start - pdb_idx)
            new_id = (res.get_id()[0], new_imgt_pos, " ")
        else:
            last_imgt_pos += 1
            new_id = (" ", last_imgt_pos, " ")

        new_res.id = new_id
        LOGGER.info("OLD %s; NEW %s", res.get_id(), new_res.get_id())
        if res.get_id() != new_res.get_id():
            deviations += 1
        new_chain.add(new_res)
        new_res.parent = new_chain

    return new_chain, deviations


def thread_alignment(
    pdb_file: str,
    chain: str,
    alignment: AnarciAlignment,
    output_pdb: str,
    start_res: int,
    end_res: int,
    alignment_start: int,
    residue_range: Tuple[int, int] = (0, 0),
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
        residue_range: Tuple of (start, end) residue numbers to process
            (inclusive). Use (0, 0) to process all residues.

    Returns:
        Number of residue ID deviations from original numbering.

    Raises:
        ValueError: If extended insertion codes are used but output is not .cif.
    """
    validate_output_format(output_pdb, alignment)

    LOGGER.info(
        f"Threading alignment for {pdb_file} chain {chain}; "
        f"writing to {output_pdb}"
    )

    parser = (
        PDB.MMCIFParser(QUIET=True)
        if pdb_file.lower().endswith(".cif")
        else PDB.PDBParser(QUIET=True)
    )
    structure = parser.get_structure("input_structure", pdb_file)
    new_structure = Structure.Structure("threaded_structure")
    new_model = Model.Model(0)

    all_devs = 0

    for ch in structure[0]:
        if ch.id != chain:
            new_model.add(ch)
        else:
            new_chain, deviations = thread_onto_chain(
                ch,
                alignment,
                start_res,
                end_res,
                alignment_start,
                residue_range,
            )
            new_model.add(new_chain)
            all_devs += deviations

    new_structure.add(new_model)
    io = PDB.MMCIFIO() if output_pdb.endswith(".cif") else PDB.PDBIO()
    io.set_structure(new_structure)
    io.save(output_pdb)
    LOGGER.info(f"Saved threaded structure to {output_pdb}")
    return all_devs
