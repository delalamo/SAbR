# test_edit_pdb.py
import numpy as np
import pytest
from Bio.PDB import Atom, Chain, Model, Residue, Structure
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.PDBParser import PDBParser

from sabr import edit_pdb


def make_residue(resname: str, seqid: int, icode: str = " "):
    """Create a minimal Residue with a single atom so I/O works."""
    res_id = (" ", seqid, icode)  # (' ', sequence number, icode)
    residue = Residue.Residue(res_id, resname, "")
    # Add a single dummy atom
    atom = Atom.Atom(
        name="CA",
        coord=np.array([0.0, 0.0, float(seqid)]),
        bfactor=0.0,
        occupancy=1.0,
        altloc=" ",
        fullname=" CA ",
        serial_number=seqid,
        element="C",
    )
    residue.add(atom)
    return residue


def make_structure_with_chain(chain_id="A", include_water=True):
    """Structure with one chain, 4 amino-acid residues (+ optional HOH)."""
    structure = Structure.Structure("test")
    model = Model.Model(0)
    chain = Chain.Chain(chain_id)

    # 4 residues: ALA(1), GLY(2), SER(3), THR(4)
    chain.add(make_residue("ALA", 1))
    chain.add(make_residue("GLY", 2))
    chain.add(make_residue("SER", 3))
    chain.add(make_residue("THR", 4))

    if include_water:
        # add one HOH which should be ignored by n_res counting
        water = Residue.Residue(("W", 5, " "), "HOH", "")
        chain.add(water)

    model.add(chain)
    structure.add(model)
    return structure, chain


@pytest.fixture
def monkeypatch_AA_3TO1(monkeypatch):
    """Ensure mapping exists for residues used in tests."""
    mapping = {
        "ALA": "A",
        "GLY": "G",
        "SER": "S",
        "THR": "T",
    }
    monkeypatch.setattr(edit_pdb.constants, "AA_3TO1", mapping, raising=True)
    return mapping


def test_thread_onto_chain_length_mismatch_raises(monkeypatch_AA_3TO1):
    # Build chain with 4 amino-acid residues (+ HOH ignored)
    structure, chain = make_structure_with_chain()

    # n_res counts non-water residues -> 4
    # Provide too-short ANARCI output to trigger the error
    anarci_out = [(((10, " "), "A")), (((20, " "), "G"))]  # len=2 != 4

    with pytest.raises(ValueError) as excinfo:
        edit_pdb.thread_onto_chain(
            chain=chain,
            anarci_out=anarci_out,
            start_res=1,
            end_res=2,
        )

    msg = str(excinfo.value)
    assert "ANARCI output does not match sequence length" in msg
    assert "Chain has 4 residues, ANARCI output has 2" in msg


def test_thread_onto_chain_happy_path(monkeypatch_AA_3TO1):
    """
    - start_res=1, end_res=2 means residues at indices 1 and 2 are threaded
    - code uses anarci_out[i - start_res] for those positions
    - outside the range: id -> (' ', (i - start_res) + 1, ' ')
    """
    structure, chain = make_structure_with_chain()

    # Build a full-length ANARCI output
    # (len must equal number of non-HOH residues => 4)
    # Each entry: ((new_seqid, icode), aa_one_letter)
    # Only entries 0 and 1 will be used (for i=1 and i=2),
    # but length must be 4 to pass the len check.
    anarci_out = [
        ((10, " "), "A"),  # for i=1 (ALA -> 'A')
        ((20, " "), "S"),  # for i=2 (SER -> 'S')
        ((999, " "), "X"),  # unused but fills length
        ((999, " "), "X"),  # unused but fills length
    ]

    new_chain = edit_pdb.thread_onto_chain(
        chain=chain,
        anarci_out=anarci_out,
        start_res=1,
        end_res=2,
    )

    # Verify new_chain has same number of residues as original (including HOH)
    assert sum(1 for _ in new_chain.get_residues()) == sum(
        1 for _ in chain.get_residues()
    )

    # Collect new residue IDs (skip waters when checking AA positions)
    # in order of enumeration i=0..N-1
    new_ids = [res.get_id() for res in new_chain.get_residues()]

    # i = 0 (outside range): new_id = (' ', (0 - 1) + 1, ' ') = (' ', 0, ' ')
    assert new_ids[0] == (" ", 0, " ")

    # i = 1 (inside, from anarci_out[0] -> seqid 10)
    assert new_ids[1] == (" ", 10, " ")

    # i = 2 (inside, from anarci_out[1] -> seqid 20)
    assert new_ids[2] == (" ", 20, " ")

    # i = 3 (outside range): new_id = (' ', (3 - 1) + 1, ' ') = (' ', 3, ' ')
    assert new_ids[3] == (" ", 3, " ")


# --- tests for thread_alignment ----------------------------------------------


def test_thread_alignment_writes_output(tmp_path, monkeypatch_AA_3TO1):
    """
    Full E2E check:
      - write a source PDB with chains A and B
      - thread chain A over indices 1..2
      - confirm output file exists and A's residue IDs updated as expected
    """
    # Build a source structure with *two* chains
    # so we also exercise the "other chains copied" branch
    structure, chain_A = make_structure_with_chain(chain_id="A")
    _, chain_B = make_structure_with_chain(chain_id="B", include_water=False)

    # put both chains into the same model of the source structure
    structure[0].add(chain_B)

    # Save the source PDB to disk
    src_pdb = tmp_path / "src.pdb"
    io = PDBIO()
    io.set_structure(structure)
    io.save(src_pdb.as_posix())

    # Prepare alignment
    alignment = [
        ((11, " "), "A"),  # used for i=1
        ((22, " "), "S"),  # used for i=2
        ((999, " "), "X"),
        ((999, " "), "X"),
    ]

    out_pdb = tmp_path / "out.pdb"

    # Run threading
    edit_pdb.thread_alignment(
        pdb_file=src_pdb.as_posix(),
        chain="A",
        alignment=alignment,
        output_pdb=out_pdb.as_posix(),
        start_res=1,
        end_res=2,
    )

    # Output exists
    assert out_pdb.exists() and out_pdb.stat().st_size > 0

    # Parse the output and check chain A residue IDs reflect changes
    parser = PDBParser(QUIET=True)
    out_struct = parser.get_structure("out", out_pdb.as_posix())
    out_chain_A = out_struct[0]["A"]

    out_ids = [res.get_id() for res in out_chain_A.get_residues()]

    # i = 0 (outside): (' ', 0, ' ')
    assert out_ids[0] == (" ", 0, " ")
    # i = 1 (inside, alignment[0] -> 11)
    assert out_ids[1] == (" ", 11, " ")
    # i = 2 (inside, alignment[1] -> 22)
    assert out_ids[2] == (" ", 22, " ")
    # i = 3 (outside): (' ', 3, ' ')
    assert out_ids[3] == (" ", 3, " ")

    # Chain B should also exist and be unmodified in length
    out_chain_B = out_struct[0]["B"]
    assert sum(1 for _ in out_chain_B.get_residues()) == sum(
        1 for _ in chain_B.get_residues()
    )
