import copy
import logging

from Bio import PDB
from Bio.PDB import Chain, Model, Structure

from sabr import constants

LOGGER = logging.getLogger(__name__)


def thread_onto_chain(
    chain: Chain.Chain,
    anarci_out: dict[str, str],
    start_res: int,
    end_res: int,
    trim: bool = False,
) -> Chain.Chain:
    """
    Thread the alignment onto a given chain object.
    """

    new_chain = Chain.Chain(chain.id)

    chain_res = []

    for i, res in enumerate(chain.get_residues()):
        if res.get_id()[0] != " ":
            continue  # Skip non-standard residues (e.g., heteroatoms, waters)
        new_res = copy.deepcopy(res)
        new_res.detach_parent()
        if i >= start_res and i <= end_res:
            (new_id, icode), aa = anarci_out[i - start_res]
            if aa != constants.AA_3TO1[new_res.get_resname()]:
                for j, ((new_id, icode), aa) in enumerate(anarci_out):
                    print(j, new_id, icode, aa)
                for j, res in enumerate(chain.get_residues()):
                    if res.get_id()[0] != " ":
                        continue
                    print(
                        j,
                        res.get_id(),
                        constants.AA_3TO1[res.get_resname()],
                        res.get_resname(),
                    )
                print("START", start_res)
                print("END", end_res)
                raise ValueError(f"Residue mismatch!")
            new_id = (res.get_id()[0], new_id, icode)
        else:
            if not trim:
                if i < start_res:
                    new_id = (" ", (i - start_res) + 1, " ")
                else:
                    new_id = (" ", (i - end_res) + anarci_out[-1][0][0], " ")
        new_res.id = new_id
        LOGGER.info(f"OLD: {res.get_id()}; NEW: {new_res.get_id()}")
        if not trim or (i >= start_res and i <= end_res):
            new_chain.add(new_res)
            new_res.parent = new_chain
            chain_res.append(res.get_id()[1:])
    return new_chain


def identify_deviations(
    pdb_file: str,
    chain_id: str,
    anarci_out: dict[str, str],
    anarci_start: int,
    anarci_end: int,
    alignment_start: int,
) -> Chain.Chain:
    """
    Thread the alignment onto a given chain object.
    """

    parser = PDB.PDBParser(QUIET=True)
    chain = parser.get_structure("input_structure", pdb_file)[0][chain_id]

    deviations = []
    i = -1
    # for j, ((new_id, icode), aa) in enumerate(anarci_out):
    #     print(j, new_id, icode, aa)
    # print()
    for j, res in enumerate(chain.get_residues()):
        # print(res.get_id(), res.get_resname())
        if res.get_id()[0].strip() != "" or j < alignment_start:
            continue  # Skip non-standard residues (e.g., heteroatoms, waters)
        i += 1
        if i >= anarci_start and i < anarci_end:
            try:
                (new_id, icode), aa = anarci_out[i - anarci_start]
                new_id += alignment_start
                # print(i, aa, new_id, icode, aa, res.get_id()[1])
            except IndexError:
                raise IndexError(anarci_start, anarci_end, len(anarci_out), i)
            if aa == "-":
                i -= 1
                continue
            if aa != constants.AA_3TO1[res.get_resname()]:
                raise ValueError(f"Residue mismatch! {aa} {res.get_resname()}")
            new_id = (res.get_id()[0], new_id, icode)
        else:
            if i < (anarci_start):
                new_id = (" ", (i - (anarci_start + alignment_start)) + 1, " ")
            else:
                new_id = (
                    " ",
                    (i - anarci_end + alignment_start) + anarci_out[-1][0][0],
                    " ",
                )
        if new_id[1] != res.get_id()[1] or new_id[2] != res.get_id()[2]:
            deviations.append((res.get_id(), new_id))
    return deviations


def thread_alignment(
    pdb_file: str,
    chain: str,
    alignment: dict[str, str],
    output_pdb: str,
    start_res: int,
    end_res: int,
    trim: bool = False,
) -> PDB.Structure.Structure:
    """
    Thread the alignment onto the PDB structure.
    """
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
                thread_onto_chain(ch, alignment, start_res, end_res, trim=trim)
            )

    new_structure.add(new_model)
    io = PDB.PDBIO()
    if output_pdb.endswith(".cif"):
        io = PDB.MMCIFIO()
    io.set_structure(new_structure)
    io.save(output_pdb)
