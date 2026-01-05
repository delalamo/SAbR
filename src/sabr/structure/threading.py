#!/usr/bin/env python3
"""Structure file modification and residue renumbering module.

This module provides functions for threading ANARCI alignments onto protein
structures, renumbering residues according to antibody numbering schemes.

Key functions:
- thread_alignment: Main entry point for renumbering a structure chain
- thread_onto_chain: Core renumbering logic for a single chain (BioPython)

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

import gemmi
from Bio.PDB import Chain
from Bio.PDB.mmcifio import MMCIFIO

from sabr.core.constants import AA_3TO1
from sabr.core.types import AnarciAlignment
from sabr.structure.io import read_structure_biopython

LOGGER = logging.getLogger(__name__)


def has_extended_insertion_codes(alignment: AnarciAlignment) -> bool:
    """Check if alignment contains extended (multi-char) insertion codes."""
    return any(len(icode.strip()) > 1 for (_, icode), _ in alignment)


def validate_output_format(
    output_path: str, alignment: AnarciAlignment
) -> None:
    """Validate that the output format supports the insertion codes used."""
    if has_extended_insertion_codes(alignment) and not output_path.endswith(
        ".cif"
    ):
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


def _thread_gemmi_chain(
    chain: gemmi.Chain,
    anarci_out: AnarciAlignment,
    anarci_start: int,
    anarci_end: int,
    alignment_start: int,
    residue_range: Tuple[int, int] = (0, 0),
) -> int:
    """Renumber a Gemmi chain in-place using ANARCI alignment.

    Args:
        chain: Gemmi Chain object to renumber (modified in-place).
        anarci_out: ANARCI alignment output as list of ((resnum, icode), aa).
        anarci_start: Starting position in the ANARCI window.
        anarci_end: Ending position in the ANARCI window.
        alignment_start: Offset where alignment begins in the sequence.
        residue_range: Tuple of (start, end) residue numbers to process
            (inclusive). Use (0, 0) to process all residues.

    Returns:
        Number of residue ID deviations from original numbering.
    """
    start_res, end_res = residue_range
    range_str = (
        f" (residue_range={start_res}-{end_res})"
        if residue_range != (0, 0)
        else ""
    )
    LOGGER.info(
        f"Threading chain {chain.name} with ANARCI window "
        f"[{anarci_start}, {anarci_end}) (alignment_start={alignment_start})"
        + range_str
    )

    aligned_residue_idx = -1
    last_imgt_pos = None
    deviations = 0
    pdb_idx = 0

    for res in chain:
        res_num = res.seqid.num

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
        # het_flag: 'A' = amino acid, 'H' = HETATM, 'W' = water
        is_hetatm = res.het_flag != "A"

        if is_in_aligned_region and not is_hetatm:
            aligned_residue_idx += 1

        if aligned_residue_idx >= 0:
            aligned_residue_idx = _skip_deletions(
                aligned_residue_idx, anarci_start, anarci_out
            )

        anarci_array_idx = aligned_residue_idx + anarci_start
        is_in_fv_region = aligned_residue_idx >= 0
        is_before_fv_end = anarci_array_idx < len(anarci_out)

        old_seqid = str(res.seqid)

        if is_in_fv_region and is_before_fv_end and not is_hetatm:
            (new_imgt_pos, icode), aa = anarci_out[anarci_array_idx]
            last_imgt_pos = new_imgt_pos

            # Verify residue matches
            one_letter = AA_3TO1.get(res.name, "X")
            if aa != one_letter:
                raise ValueError(
                    f"Residue mismatch! Expected {aa}, got {one_letter} "
                    f"({res.name})"
                )

            # Set new residue ID
            res.seqid.num = new_imgt_pos
            # Handle insertion code - strip whitespace
            icode_str = icode.strip() if icode else ""
            res.seqid.icode = icode_str[0] if icode_str else " "

        elif is_hetatm:
            # Keep HETATM residues unchanged
            pass

        elif aligned_residue_idx < 0:
            # PRE-Fv region
            first_anarci_pos = anarci_out[anarci_start][0][0]
            new_imgt_pos = first_anarci_pos - (alignment_start - pdb_idx)
            res.seqid.num = new_imgt_pos
            res.seqid.icode = " "

        else:
            # POST-Fv region
            last_imgt_pos += 1
            res.seqid.num = last_imgt_pos
            res.seqid.icode = " "

        new_seqid = str(res.seqid)
        LOGGER.info("OLD %s; NEW %s", old_seqid, new_seqid)
        if old_seqid != new_seqid:
            deviations += 1

        pdb_idx += 1

    return deviations


def _thread_alignment_biopython(
    pdb_file: str,
    chain_id: str,
    alignment: AnarciAlignment,
    output_cif: str,
    start_res: int,
    end_res: int,
    alignment_start: int,
    residue_range: Tuple[int, int] = (0, 0),
) -> int:
    """Thread alignment using BioPython for CIF output with extended icodes.

    This fallback is used when extended insertion codes (multi-character like
    "AA", "AB") are present, which Gemmi cannot handle.

    Args:
        pdb_file: Path to input PDB/CIF file.
        chain_id: Chain identifier to renumber.
        alignment: ANARCI-style alignment list of ((resnum, icode), aa) tuples.
        output_cif: Path to output CIF file.
        start_res: Start residue index from ANARCI.
        end_res: End residue index from ANARCI.
        alignment_start: Offset where alignment begins in the sequence.
        residue_range: Tuple of (start, end) residue numbers to process.

    Returns:
        Number of residue ID deviations from original numbering.
    """
    LOGGER.info(
        f"Using BioPython fallback for extended insertion codes: "
        f"{pdb_file} chain {chain_id}"
    )

    structure = read_structure_biopython(pdb_file)

    # Find and renumber the target chain
    model = structure[0]
    chain = model[chain_id]

    # Thread the chain using BioPython's thread_onto_chain
    threaded_chain, deviations = thread_onto_chain(
        chain,
        alignment,
        start_res,
        end_res,
        alignment_start,
        residue_range,
    )

    # Replace the chain in the model
    model.detach_child(chain_id)
    model.add(threaded_chain)

    # Write output CIF using BioPython's MMCIFIO
    io = MMCIFIO()
    io.set_structure(structure)
    io.save(output_cif)

    LOGGER.info(f"Saved threaded structure to {output_cif}")
    return deviations


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

    Uses Gemmi for fast file I/O operations. Falls back to BioPython when
    extended insertion codes are present (Gemmi only supports single-char).

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

    # Use BioPython fallback for extended insertion codes (Gemmi limitation)
    if has_extended_insertion_codes(alignment) and output_pdb.endswith(".cif"):
        return _thread_alignment_biopython(
            pdb_file,
            chain,
            alignment,
            output_pdb,
            start_res,
            end_res,
            alignment_start,
            residue_range,
        )

    LOGGER.info(
        f"Threading alignment for {pdb_file} chain {chain}; "
        f"writing to {output_pdb}"
    )

    # Read structure with Gemmi
    structure = gemmi.read_structure(pdb_file)

    all_devs = 0

    # Find and renumber the target chain
    model = structure[0]
    for ch in model:
        if ch.name == chain:
            deviations = _thread_gemmi_chain(
                ch,
                alignment,
                start_res,
                end_res,
                alignment_start,
                residue_range,
            )
            all_devs += deviations
            break

    # Write output file
    if output_pdb.endswith(".cif"):
        structure.make_mmcif_document().write_file(output_pdb)
    else:
        structure.write_pdb(output_pdb)

    LOGGER.info(f"Saved threaded structure to {output_pdb}")
    return all_devs
