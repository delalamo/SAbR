"""Thread ANARCI numbering onto BioPython structures."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Optional

from Bio.PDB import Chain, Model, Structure

from sabr.constants import AA_3TO1
from sabr.errors import ChainNotFoundError, InputStructureError, OutputFormatError
from sabr.numbering.anarci import AnarciAlignment
from sabr.structure.io import read_structure, write_structure
from sabr.structure.residues import ResidueRange, normalize_residue_range

LOGGER = logging.getLogger(__name__)


class ResidueRegion(Enum):
    """Classification of residue regions during threading."""

    PRE_FV = auto()
    IN_FV = auto()
    POST_FV = auto()
    HETATM = auto()


@dataclass
class RenumberingState:
    """State maintained during chain threading."""

    aligned_residue_idx: int = -1
    last_imgt_pos: Optional[int] = None


def _classify_residue_region(
    selected_idx: int,
    aligned_residue_idx: int,
    anarci_array_idx: int,
    alignment_start: int,
    anarci_len: int,
    is_hetatm: bool,
) -> ResidueRegion:
    if is_hetatm:
        return ResidueRegion.HETATM
    if selected_idx < alignment_start or aligned_residue_idx < 0:
        return ResidueRegion.PRE_FV
    if anarci_array_idx < anarci_len:
        return ResidueRegion.IN_FV
    return ResidueRegion.POST_FV


def _skip_deletions(
    anarci_idx: int,
    anarci_start: int,
    anarci_out: AnarciAlignment,
) -> int:
    """Advance past ANARCI deletion rows."""
    anarci_array_idx = anarci_idx + anarci_start
    while anarci_array_idx < len(anarci_out) and anarci_out[anarci_array_idx][1] == "-":
        anarci_idx += 1
        anarci_array_idx = anarci_idx + anarci_start
    return anarci_idx


def _compute_new_residue_id(
    region: ResidueRegion,
    anarci_out: AnarciAlignment,
    anarci_array_idx: int,
    anarci_start: int,
    alignment_start: int,
    selected_idx: int,
    state: RenumberingState,
    original_het_flag: str,
    resname: str,
) -> tuple[str, int, str] | None:
    if region == ResidueRegion.HETATM:
        return None

    if region == ResidueRegion.PRE_FV:
        first_anarci_pos = anarci_out[anarci_start][0][0]
        new_imgt_pos = first_anarci_pos - (alignment_start - selected_idx)
        return (original_het_flag, new_imgt_pos, " ")

    if region == ResidueRegion.IN_FV:
        (new_imgt_pos, icode), expected_aa = anarci_out[anarci_array_idx]
        state.last_imgt_pos = new_imgt_pos

        one_letter = AA_3TO1.get(resname, "X")
        if expected_aa != one_letter:
            raise InputStructureError(
                f"Residue mismatch while threading numbering: expected "
                f"{expected_aa}, got {one_letter} ({resname})."
            )

        return (original_het_flag, new_imgt_pos, icode)

    if state.last_imgt_pos is None:
        raise InputStructureError(
            "Cannot number post-Fv residues before any ANARCI residue was assigned."
        )
    state.last_imgt_pos += 1
    return (" ", state.last_imgt_pos, " ")


def has_extended_insertion_codes(alignment: AnarciAlignment) -> bool:
    """Return whether ANARCI output includes multi-character insertion codes."""
    return any(len(icode.strip()) > 1 for (_, icode), _ in alignment)


def validate_output_format(output_path: str, alignment: AnarciAlignment) -> None:
    """Validate that the output format can represent assigned residue IDs."""
    suffix = Path(output_path).suffix.lower()
    if suffix not in {".pdb", ".cif"}:
        raise OutputFormatError(
            f"Output file must have extension .pdb or .cif. Got: {output_path}"
        )
    if suffix == ".pdb" and has_extended_insertion_codes(alignment):
        raise OutputFormatError(
            "Extended insertion codes detected in alignment. PDB format only "
            "supports single-character insertion codes. Use mmCIF output (.cif)."
        )


def _selected_standard_residue(
    residue,
    residue_range: ResidueRange | None,
) -> bool:
    if residue.get_id()[0].strip():
        return False
    if residue_range is None:
        return True
    return residue_range.contains(residue.get_id())


def thread_onto_chain(
    chain: Chain.Chain,
    anarci_out: AnarciAlignment,
    anarci_start: int,
    anarci_end: int,
    alignment_start: int,
    residue_range: ResidueRange | tuple[int, int] | None = None,
) -> tuple[Chain.Chain, int]:
    """Return a deep-copied chain with selected residues renumbered.

    Residues outside ``residue_range`` are preserved unchanged in the output.
    The range is interpreted against original structure residue numbers.
    """
    del anarci_end
    selected_range = normalize_residue_range(residue_range)
    LOGGER.info(
        "Threading chain %s with ANARCI window starting at %s "
        "(alignment_start=%s, residue_range=%s)",
        chain.id,
        anarci_start,
        alignment_start,
        selected_range or "all",
    )

    new_chain = Chain.Chain(chain.id)
    state = RenumberingState()
    deviations = 0
    selected_idx = -1

    for residue in chain.get_residues():
        new_residue = copy.deepcopy(residue)
        new_residue.detach_parent()

        if not _selected_standard_residue(residue, selected_range):
            new_chain.add(new_residue)
            continue

        selected_idx += 1
        is_in_aligned_region = selected_idx >= alignment_start

        if is_in_aligned_region:
            state.aligned_residue_idx += 1

        if state.aligned_residue_idx >= 0:
            state.aligned_residue_idx = _skip_deletions(
                state.aligned_residue_idx, anarci_start, anarci_out
            )

        anarci_array_idx = state.aligned_residue_idx + anarci_start
        region = _classify_residue_region(
            selected_idx,
            state.aligned_residue_idx,
            anarci_array_idx,
            alignment_start,
            len(anarci_out),
            False,
        )

        new_id = _compute_new_residue_id(
            region,
            anarci_out,
            anarci_array_idx,
            anarci_start,
            alignment_start,
            selected_idx,
            state,
            residue.get_id()[0],
            residue.get_resname(),
        )
        if new_id is not None:
            new_residue.id = new_id

        if residue.get_id() != new_residue.get_id():
            deviations += 1
        new_chain.add(new_residue)

    return new_chain, deviations


def _copy_structure_with_chain_transform(
    structure: Structure.Structure,
    chain_id: str,
    structure_name: str,
    transform_chain,
) -> tuple[Structure.Structure, int]:
    new_structure = Structure.Structure(structure_name)
    new_model = Model.Model(0)
    deviations = 0
    found = False

    for chain in structure[0]:
        if chain.id != chain_id:
            new_chain = copy.deepcopy(chain)
            new_chain.detach_parent()
        else:
            found = True
            new_chain, deviations = transform_chain(chain)
        new_model.add(new_chain)

    if not found:
        available = [chain.id for chain in structure[0]]
        raise ChainNotFoundError(
            f"Chain '{chain_id}' not found in structure. Available chains: {available}"
        )

    new_structure.add(new_model)
    return new_structure, deviations


def thread_numbering_onto_structure(
    structure: Structure.Structure,
    chain_id: str,
    alignment: AnarciAlignment,
    start_res: int,
    end_res: int,
    alignment_start: int,
    residue_range: ResidueRange | tuple[int, int] | None = None,
) -> tuple[Structure.Structure, int]:
    """Return a structure with numbering threaded onto one chain."""

    def transform(chain: Chain.Chain) -> tuple[Chain.Chain, int]:
        return thread_onto_chain(
            chain,
            alignment,
            start_res,
            end_res,
            alignment_start,
            residue_range,
        )

    return _copy_structure_with_chain_transform(
        structure, chain_id, "renumbered_structure", transform
    )


def thread_alignment(
    pdb_file: str,
    chain: str,
    alignment: AnarciAlignment,
    output_pdb: str,
    start_res: int,
    end_res: int,
    alignment_start: int,
    residue_range: ResidueRange | tuple[int, int] | None = None,
) -> int:
    """Write a structure file with numbering threaded onto one chain."""
    validate_output_format(output_pdb, alignment)
    structure = read_structure(pdb_file)
    renumbered_structure, deviations = thread_numbering_onto_structure(
        structure,
        chain,
        alignment,
        start_res,
        end_res,
        alignment_start,
        residue_range,
    )
    write_structure(renumbered_structure, output_pdb)
    return deviations
