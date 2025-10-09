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
    """
    Create renumbered copy of a PDB chain by threading ANARCI indices onto it

    This function deep-copies each residue of the input ``chain`` and assigns a
    new PDB-style residue ID by following a window of ANARCI output. Within the
    ANARCI window, residues are validated against the expected one-letter amino
    acid and assigned the corresponding ANARCI (index, insertion-code). Outside
    the window, residues are numbered by extrapolating before/after the window.

    Parameters
    ----------
    chain : Bio.PDB.Chain.Chain
        The input chain to be renumbered (source of residues).
    anarci_out : Sequence[tuple[tuple[int, str], str]]
        ANARCI results for the *window of interest*, where each element is
        ``((new_index, insertion_code), aa_one_letter)``. Gaps may be present
        in the original ANARCI results (``aa_one_letter == '-'``) but this
        function assumes the provided slice is consistent with the logic below.
    anarci_start : int
        Start offset into the original ANARCI output (inclusive).
    anarci_end : int
        End offset into the original ANARCI output (exclusive).
    alignment_start : int
        Index (0-based) into ``chain.get_residues()`` where the Fv alignment
        begins; residues before this index are treated as N-terminal prefix.

    Returns
    -------
    Bio.PDB.Chain.Chain
        A new Chain instance with residues re-IDed according to the logic
        below. Original residue objects are not modified.

    Raises
    ------
    ValueError
        If an amino-acid identity in the chain disagrees with the ANARCI
        one-letter code at the mapped position.

    Notes
    -----
    - HETATM residues (``hetflag != ''``) are copied but do not advance the
      ANARCI index counter.
    - The resulting residue IDs are assigned as ``(hetflag, index, icode)`` and
      parents are updated to reference the newly created chain.
    """

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
    """
    Compute residue ID changes implied by ANARCI threading, without writing.

    This is a dry-run that reads ``pdb_file``, walks the specified
    ``chain_id``, and determines which standard residues (``hetflag == ''``)
    would change their residue ID if threaded with the provided ANARCI window

    Parameters
    ----------
    pdb_file : str
        Path to an input PDB or mmCIF file that Bio.PDB can parse.
    chain_id : str
        Chain identifier to examine (e.g., ``'H'``, ``'L'``, or ``'A'``).
    og_anarci_out : Sequence[tuple[tuple[int, str], str]]
        Full ANARCI output as ``((index, icode), aa_one_letter)``; gaps may be
        present (``'-'``). Gaps are removed internally for the mapping step.
    anarci_start : int
        Start index into the ANARCI output (inclusive).
    anarci_end : int
        End index into the ANARCI output (exclusive).
    alignment_start : int
        0-based index in the chain residues at which the Fv alignment starts.

    Returns
    -------
    list[tuple[tuple[str, int, str], tuple[str, int, str]]]
        A list of ``(old_id, new_id)`` tuples for residues whose ID would
        change. Each ID is a PDB residue ID ``(hetflag, resseq, icode)``.

    Raises
    ------
    IndexError
        If ANARCI indexing is inconsistent with the chain/alignment window.
    ValueError
        If an amino-acid identity mismatch is detected.

    Notes
    -----
    - This function logs additional debug output when deviations are found.
    - HETATM residues do not contribute to ANARCI index advancement, but they
      are included in the ``old_ids`` / ``new_ids`` bookkeeping.
    """
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
                    print(j, k)
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
        # print(i, j, res.get_id(), res.get_resname(), new_id)
        new_ids.append((new_id, res.get_resname()))
        if (
            new_id[1] != res.get_id()[1] or new_id[2] != res.get_id()[2]
        ) and res.get_id()[0].strip() == "":
            deviations.append((res.get_id(), new_id))
    if len(deviations) > 0:
        for i, og in og_anarci_out:
            print(i, og)
        for i, (old_id, resname) in enumerate(old_ids):
            print(i, old_id, resname)
        for i, (new_id, resname) in enumerate(new_ids):
            print(i, new_id, resname)
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
    """
    Thread an ANARCI numbering window onto a PDB structure and write the file.

    This function parses ``pdb_file``, constructs a new structure containing a
    single model, and for the specified ``chain`` replaces that chain with a
    renumbered copy produced by :func:`thread_onto_chain`. The resulting
    structure is written to ``output_pdb`` (PDB or mmCIF based on extension)
    and returned.

    Parameters
    ----------
    pdb_file : str
        Path to the input coordinate file readable by :mod:`Bio.PDB`.
    chain : str
        Chain identifier to renumber (e.g., ``"H"``, ``"L"``, or ``"A"``).
    alignment : dict[str, str]
        ANARCI window used to drive renumbering.  In practice this is often
        provided as an iterable of entries like
        ``((index, icode), aa_one_letter)``; the value here is typed as a
        ``dict[str, str]`` for historical compatibility with earlier code.
        Whatever is passed is forwarded to :func:`thread_onto_chain` unchanged.
    output_pdb : str
        Output path. If it ends with ``.cif`` the structure is written as
        mmCIF via :class:`Bio.PDB.MMCIFIO`; otherwise PDB via
        :class:`Bio.PDB.PDBIO`.
    start_res : int
        Start (inclusive) of the ANARCI window to apply.
    end_res : int
        End (exclusive) of the ANARCI window to apply.
    alignment_start : int
        Zero-based index along the chain's residue iteration at which the
        alignment to the ANARCI window begins (i.e., N-terminus of the Fv
        region in PDB residue order).

    Returns
    -------
    Bio.PDB.Structure.Structure
        The in-memory structure containing the renumbered chain; it is also
        written to disk at ``output_pdb``.

    Raises
    ------
    FileNotFoundError
        If ``pdb_file`` cannot be opened.
    KeyError / ValueError
        Propagated from downstream parsing or
        :func:`thread_onto_chain` if the alignment content is invalid.

    Notes
    -----
    - All non-target chains are copied verbatim.
    - The writer (PDB vs mmCIF) is selected from the file suffix.
    - See :func:`thread_onto_chain` for details on alignment structure and
      residue-ID assignment semantics.
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
                thread_onto_chain(
                    ch, alignment, start_res, end_res, alignment_start
                )
            )

    new_structure.add(new_model)
    io = PDB.PDBIO()
    if output_pdb.endswith(".cif"):
        io = PDB.MMCIFIO()
    io.set_structure(new_structure)
    io.save(output_pdb)
