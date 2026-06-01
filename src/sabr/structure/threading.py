"""Thread ANARCI numbering onto BioPython structures."""

from __future__ import annotations

import copy
import logging
from pathlib import Path

from Bio.PDB import Chain, Model, Structure

from sabr.errors import ChainNotFoundError, InputStructureError, OutputFormatError
from sabr.numbering.anarci import AnarciAlignment
from sabr.structure.io import read_structure, write_structure
from sabr.structure.residues import AA_3TO1, ResidueId, ResidueRange

LOGGER = logging.getLogger(__name__)


def _residue_id_from_biopython(residue_id: tuple[str, int, str]) -> ResidueId:
    return ResidueId(residue_id[1], residue_id[2].strip())


def has_extended_insertion_codes(alignment: AnarciAlignment) -> bool:
    """Return whether ANARCI output includes multi-character insertion codes."""
    return any(len(residue.insertion_code.strip()) > 1 for residue in alignment)


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
    return _residue_id_from_biopython(residue.get_id()).in_range(residue_range)


def _check_amino_acid(residue, expected_aa: str) -> None:
    one_letter = AA_3TO1.get(residue.get_resname(), "X")
    if expected_aa != one_letter:
        raise InputStructureError(
            f"Residue mismatch while threading numbering: expected "
            f"{expected_aa}, got {one_letter} ({residue.get_resname()})."
        )


def _check_duplicate_ids(
    proposed: list[tuple[object, tuple[str, int, str]]],
    chain_id: str,
) -> None:
    seen = set()
    for _residue, residue_id in proposed:
        if residue_id[0].strip():
            continue
        if residue_id in seen:
            raise OutputFormatError(
                f"Renumbering would create duplicate residue id {residue_id!r} "
                f"in chain {chain_id}. This usually happens when --residue-range "
                "preserves residues whose original numbers overlap the target "
                "numbering. Use a narrower input file or output only the variable "
                "domain."
            )
        seen.add(residue_id)


def thread_onto_chain(
    chain: Chain.Chain,
    anarci_out: AnarciAlignment,
    alignment_start: int,
    residue_range: ResidueRange | None = None,
) -> tuple[Chain.Chain, int]:
    """Return a deep-copied chain with selected residues renumbered.

    Residues outside ``residue_range`` are preserved unchanged in the output.
    The range is interpreted against original structure residue numbers.
    """
    LOGGER.info(
        "Threading chain %s (alignment_start=%s, residue_range=%s)",
        chain.id,
        alignment_start,
        residue_range or "all",
    )

    deviations = 0
    selected_idx = -1
    numbering_idx = 0
    last_position = None
    proposed = []

    for residue in chain.get_residues():
        new_residue = copy.deepcopy(residue)
        new_residue.detach_parent()

        if not _selected_standard_residue(residue, residue_range):
            proposed.append((new_residue, new_residue.get_id()))
            continue

        selected_idx += 1

        if selected_idx < alignment_start:
            if not anarci_out:
                raise InputStructureError(
                    "Cannot number residues without ANARCI output."
                )
            first_position = anarci_out[0].position
            new_id = (" ", first_position - (alignment_start - selected_idx), " ")
        elif numbering_idx < len(anarci_out):
            numbered_residue = anarci_out[numbering_idx]
            _check_amino_acid(residue, numbered_residue.amino_acid)
            new_id = (
                " ",
                numbered_residue.position,
                numbered_residue.insertion_code or " ",
            )
            last_position = numbered_residue.position
            numbering_idx += 1
        else:
            if last_position is None:
                raise InputStructureError(
                    "Cannot number post-Fv residues before any ANARCI residue "
                    "was assigned."
                )
            last_position += 1
            new_id = (" ", last_position, " ")

        if residue.get_id() != new_id:
            deviations += 1
        proposed.append((new_residue, new_id))

    _check_duplicate_ids(proposed, chain.id)

    new_chain = Chain.Chain(chain.id)
    for new_residue, new_id in proposed:
        new_residue.id = new_id
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
    alignment_start: int,
    residue_range: ResidueRange | None = None,
) -> tuple[Structure.Structure, int]:
    """Return a structure with numbering threaded onto one chain."""

    def transform(chain: Chain.Chain) -> tuple[Chain.Chain, int]:
        return thread_onto_chain(
            chain,
            alignment,
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
    alignment_start: int,
    residue_range: ResidueRange | None = None,
) -> int:
    """Write a structure file with numbering threaded onto one chain."""
    validate_output_format(output_pdb, alignment)
    structure = read_structure(pdb_file)
    renumbered_structure, deviations = thread_numbering_onto_structure(
        structure,
        chain,
        alignment,
        alignment_start,
        residue_range,
    )
    write_structure(renumbered_structure, output_pdb)
    return deviations
