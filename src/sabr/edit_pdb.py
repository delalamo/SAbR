import copy

from Bio import PDB
from Bio.PDB import Chain, Model, Structure

from sabr import constants


def thread_onto_chain(
    chain_obj: Chain.Chain,
    anarci_output: dict[str, str],
    start_res: int,
    end_res: int,
) -> Chain.Chain:
    """
    Thread the alignment onto a given chain object.
    """

    new_chain = Chain.Chain(chain_obj.id)

    chain_res = []
    for i, res in enumerate(chain_obj.get_residues()):
        new_res = copy.deepcopy(res)
        new_res.detach_parent()
        if i >= start_res and i <= end_res:
            (new_id, icode), aa = anarci_output[i - start_res]
            assert aa == constants.AA_3TO1[new_res.get_resname()], print(
                i, start_res, res.get_id(), aa, new_res.get_resname()
            )
            new_id = (res.get_id()[0], new_id, icode)
        else:
            new_id = (" ", (i - start_res) + 1, " ")
        new_res.id = new_id
        print("OLD", res.get_id(), "NEW", new_res.get_id())
        new_chain.add(new_res)
        new_res.parent = new_chain
        chain_res.append(res.get_id()[1:])
    return new_chain


def thread_alignment(
    pdb_file: str,
    chain: str,
    alignment: dict[str, str],
    output_pdb: str,
    start_res: int,
    end_res: int,
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
            new_model.add(thread_onto_chain(ch, alignment, start_res, end_res))

    new_structure.add(new_model)
    io = PDB.PDBIO()
    if output_pdb.endswith(".cif"):
        io = PDB.MMCIFIO()
    io.set_structure(new_structure)
    io.save(output_pdb)
