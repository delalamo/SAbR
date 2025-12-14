#!/usr/bin/env python3

import copy
import logging
from typing import List, Tuple

from Bio import PDB
from Bio.PDB import Chain, Model, Structure

from sabr import constants

LOGGER = logging.getLogger(__name__)


def validate_output_format(
    output_path: str, alignment: List[Tuple[Tuple[int, str], str]]
) -> None:
    """Validate that the output format supports the insertion codes used."""
    has_extended = any(len(icode.strip()) > 1 for (_, icode), _ in alignment)

    if has_extended and not output_path.endswith(".cif"):
        raise ValueError(
            "Extended insertion codes detected in alignment. "
            "PDB format only supports single-character insertion codes. "
            "Please use mmCIF format (.cif extension) for output."
        )


def thread_onto_chain(
    chain: Chain.Chain,
    anarci_out: List[Tuple[Tuple[int, str], str]],
    anarci_start: int,
    anarci_end: int,
    alignment_start: int,
    max_residues: int = 0,
) -> Tuple[Chain.Chain, int]:
    """Return a deep-copied chain renumbered by the ANARCI window."""

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

    i = -1
    last_idx = None
    deviations = 0
    for j, res in enumerate(chain.get_residues()):
        # Skip residues beyond max_residues limit
        # (check actual residue index, not count)
        if max_residues > 0:
            res_index = res.id[1]  # Actual residue number from PDB
            if res_index > max_residues:
                LOGGER.info(
                    f"Stopping at residue index {res_index} "
                    f"(max_residues={max_residues})"
                )
                break
        past_n_pdb = j >= alignment_start  # In Fv, PDB numbering
        hetatm = res.get_id()[0].strip() != ""

        if past_n_pdb and not hetatm:
            i += 1

        if i >= anarci_start:
            # Skip ANARCI positions that correspond to deletions ("-")
            while (
                i - anarci_start < len(anarci_out)
                and anarci_out[i - anarci_start][1] == "-"
            ):
                i += 1

        past_n_anarci = i >= anarci_start  # In Fv, ANARCI numbering
        before_c = i < min(
            anarci_end, len(anarci_out)
        )  # Not yet reached C term of Fv
        new_res = copy.deepcopy(res)
        new_res.detach_parent()
        if past_n_anarci and before_c:
            (new_idx, icode), aa = anarci_out[i - anarci_start]
            last_idx = new_idx

            if aa != constants.AA_3TO1[res.get_resname()]:
                raise ValueError(f"Residue mismatch! {aa} {res.get_resname()}")
            # FIX: Don't add alignment_start - ANARCI already returns
            # correct IMGT positions
            new_id = (res.get_id()[0], new_idx, icode)
        else:
            if i < (anarci_start):
                # PRE-Fv region: number backwards from first ANARCI position
                first_anarci_pos = anarci_out[0][0][0]
                if past_n_pdb:
                    # Residue is in aligned sequence but before ANARCI window
                    # i represents position in aligned sequence,
                    # anarci_start is where ANARCI numbering begins
                    new_idx = first_anarci_pos - (anarci_start - i)
                else:
                    # Residue is before the aligned sequence entirely
                    # Number based on distance from alignment_start
                    new_idx = first_anarci_pos - (
                        anarci_start + (alignment_start - j)
                    )
                new_id = (res.get_id()[0], new_idx, " ")
            else:
                # AFTER Fv region: continue from last ANARCI position
                # Preserve the hetero flag for HETATM residues
                if hetatm:
                    # Keep original ID for HETATM (water, ligands, etc.)
                    new_id = res.get_id()
                else:
                    last_idx += 1
                    new_id = (" ", last_idx, " ")
        new_res.id = new_id
        LOGGER.info(f"OLD {res.get_id()}; NEW {new_res.get_id()}")
        if res.get_id() != new_res.get_id():
            deviations += 1
        new_chain.add(new_res)
        new_res.parent = new_chain
        chain_res.append(res.get_id()[1:])
    return new_chain, deviations


def thread_alignment(
    pdb_file: str,
    chain: str,
    alignment: List[Tuple[Tuple[int, str], str]],
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
