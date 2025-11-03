#!/usr/bin/env python3

import copy
import logging

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
) -> Chain.Chain:
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
        new_chain.add(new_res)
        new_res.parent = new_chain
        chain_res.append(res.get_id()[1:])
    return new_chain


def identify_deviations(
    pdb_file: str,
    chain_id: str,
    og_anarci_out: dict[str, str],
    anarci_start: int,
    anarci_end: int,
    alignment_start: int,
) -> Chain.Chain:
    """Return residue-ID changes that ANARCI threading would introduce."""
    LOGGER.info(
        f"Analyzing deviations in {pdb_file} chain {chain_id} "
        f"(ANARCI window [{anarci_start}, {anarci_end}), "
        f"alignment start {alignment_start})"
    )
    anarci_out = [a for a in og_anarci_out if a[1] != "-"]
    parser = PDB.PDBParser(QUIET=True)
    chain = parser.get_structure("input_structure", pdb_file)[0][chain_id]

    deviations = []
    old_ids = []
    new_ids = []
    i = -1
    last_idx = None
    for j, res in enumerate(chain.get_residues()):
        old_ids.append((res.get_id(), res.get_resname()))
        i += 1
        if not (j >= alignment_start and res.get_id()[0].strip() == ""):
            i -= 1
        if (
            i >= anarci_start
            and i < min(anarci_end, len(anarci_out))
            and j >= alignment_start
            and res.get_id()[0].strip() == ""
        ):
            try:
                (new_id, icode), aa = anarci_out[i - anarci_start]
                last_idx = new_id
                # new_id += alignment_start
            except IndexError:
                for j, k in enumerate(anarci_out):
                    LOGGER.debug(f"ANARCI out {j}: {k}")
                raise IndexError(
                    "alignment_start",
                    alignment_start,
                    "anarci_start",
                    anarci_start,
                    "anarci_end",
                    anarci_end,
                    "len(anarci_out)",
                    len(anarci_out),
                    "len(og_anarci_out)",
                    len(og_anarci_out),
                    "i",
                    i,
                )
            if aa == "-":
                i -= 1
                continue
            resname = res.get_resname()
            if aa != constants.AA_3TO1[resname]:
                raise ValueError(
                    f"Residue mismatch {res.get_id()[1]}! {aa} {resname}"
                )
            new_id = (res.get_id()[0], new_id, icode)
        else:
            if i < (anarci_start):
                new_id = (
                    " ",
                    (j - (anarci_start + alignment_start))
                    + anarci_out[0][0][0],
                    " ",
                )
            else:
                last_idx += 1
                new_id = (" ", last_idx, " ")
        index_info = (
            f"Residue index={j} (anarci_pos={i}) "
            f"{res.get_id()}/{res.get_resname()} -> {new_id}"
        )
        LOGGER.debug(index_info)
        new_ids.append((new_id, res.get_resname()))
        if (
            new_id[1] != res.get_id()[1] or new_id[2] != res.get_id()[2]
        ) and res.get_id()[0].strip() == "":
            deviations.append((res.get_id(), new_id))
    if len(deviations) > 0:
        LOGGER.info(f"Found {len(deviations)} residue ID deviations")
        for idx, og in og_anarci_out:
            LOGGER.debug(f"Original ANARCI entry {idx} -> {og}")
        for idx, (old_id, resname) in enumerate(old_ids):
            LOGGER.debug(f"Existing residue {idx}: {old_id} {resname}")
        for idx, (new_id, resname) in enumerate(new_ids):
            LOGGER.debug(f"Proposed residue {idx}: {new_id} {resname}")
    return deviations


def thread_alignment(
    pdb_file: str,
    chain: str,
    alignment: dict[str, str],
    output_pdb: str,
    start_res: int,
    end_res: int,
    alignment_start: int,
) -> PDB.Structure.Structure:
    """Write the renumbered chain to ``output_pdb`` and return the structure."""
    align_msg = (
        f"Threading alignment for {pdb_file} chain {chain}; "
        f"writing to {output_pdb}"
    )
    LOGGER.info(align_msg)
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("input_structure", pdb_file)
    # Create a new structure and model
    new_structure = Structure.Structure("threaded_structure")
    new_model = Model.Model(0)

    for ch in structure[0]:
        if ch.id != chain:
            new_model.add(ch)
        else:
            new_model.add(
                thread_onto_chain(
                    ch, alignment, start_res, end_res, alignment_start
                )
            )

    new_structure.add(new_model)
    io = PDB.PDBIO()
    if output_pdb.endswith(".cif"):
        io = PDB.MMCIFIO()
        LOGGER.debug("Detected CIF output; using MMCIFIO writer")
    io.set_structure(new_structure)
    io.save(output_pdb)
    LOGGER.info(f"Saved threaded structure to {output_pdb}")
