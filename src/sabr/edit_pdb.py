#!/usr/bin/env python3

import copy
import logging
from typing import Tuple

from Bio import PDB
from Bio.PDB import Chain, Model, Structure

from sabr import constants

LOGGER = logging.getLogger(__name__)


def thread_onto_chain(
    chain: Chain.Chain,
    anarci_out: dict[str, str],
    anarci_start: int,
    anarci_end: int,
    alignment_start: int,
) -> Tuple[Chain.Chain, int]:
    """Return a deep-copied chain renumbered by the ANARCI window.

    Raise ValueError on residue mismatches.
    """

    thread_msg = (
        f"Threading chain {chain.id} with ANARCI window "
        f"[{anarci_start}, {anarci_end}) "
        f"(alignment starts at {alignment_start})"
    )
    LOGGER.info(thread_msg)
    new_chain = Chain.Chain(chain.id)

    chain_res = []

    i = -1
    last_idx = None
    deviations = 0
    for j, res in enumerate(chain.get_residues()):
        past_n_pdb = j >= alignment_start  # In Fv, PDB numbering
        past_n_anarci = i >= anarci_start  # In Fv, ANARCI numbering
        before_c = i < min(
            anarci_end, len(anarci_out)
        )  # Not yet reached C term of Fv
        hetatm = res.get_id()[0].strip() != ""

        if not past_n_pdb and not hetatm:
            i += 1
        new_res = copy.deepcopy(res)
        new_res.detach_parent()
        if past_n_anarci and before_c:
            (new_idx, icode), aa = anarci_out[i - anarci_start]
            last_idx = new_idx

            if aa != constants.AA_3TO1[res.get_resname()]:
                raise ValueError(f"Residue mismatch! {aa} {res.get_resname()}")
            new_id = (res.get_id()[0], new_idx + alignment_start, icode)
        else:
            if i < (anarci_start):
                new_idx = (j - (anarci_start + alignment_start)) + anarci_out[
                    0
                ][0][0]
                new_id = (res.get_id()[0], new_idx, " ")
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
    alignment: dict[str, str],
    output_pdb: str,
    start_res: int,
    end_res: int,
    alignment_start: int,
) -> int:
    """Write the renumbered chain to ``output_pdb`` and return the structure."""
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
                ch, alignment, start_res, end_res, alignment_start
            )
            new_model.add(new_chain)
            all_devs += deviations

    new_structure.add(new_model)
    io = PDB.PDBIO()
    if output_pdb.endswith(".cif"):
        io = PDB.MMCIFIO()
        LOGGER.debug("Detected CIF output; using MMCIFIO writer")
    io.set_structure(new_structure)
    io.save(output_pdb)
    LOGGER.info(f"Saved threaded structure to {output_pdb}")
    return all_devs
