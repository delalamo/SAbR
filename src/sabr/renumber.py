"""Public renumbering API for SAbR."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from Bio.PDB import Structure

from sabr.alignment.aln2hmm import alignment_matrix_to_state_vector
from sabr.alignment.soft_aligner import SoftAligner
from sabr.embeddings.mpnn import QueryEmbeddings, from_chain, from_pdb
from sabr.embeddings.references import DEFAULT_REFERENCE_EMBEDDINGS
from sabr.errors import (
    AlignmentError,
    ChainNotFoundError,
    InputStructureError,
    OutputFormatError,
)
from sabr.numbering.anarci import (
    AnarciAlignment,
    build_anarci_subsequence,
    number_from_alignment,
)
from sabr.options import RenumberOptions
from sabr.structure.threading import (
    thread_alignment,
    thread_numbering_onto_structure,
)
from sabr.types import ChainType

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class _NumberingPlan:
    alignment: AnarciAlignment
    chain_type: ChainType
    first_aligned_row: int
    score: float


@dataclass(frozen=True)
class RenumberResult:
    """Result metadata returned by ``renumber_file``."""

    output_path: Path
    chain_type: ChainType
    residue_count: int
    changed_residue_count: int


def _validate_chain_id(chain_id: str) -> None:
    if len(chain_id) != 1:
        raise InputStructureError("Chain identifier must be exactly one character.")


def _validate_file_paths(
    input_path: Path,
    output_path: Path,
    options: RenumberOptions,
) -> None:
    if not input_path.exists():
        raise InputStructureError(f"Input file '{input_path}' does not exist.")
    if input_path.suffix.lower() not in {".pdb", ".cif"}:
        raise InputStructureError(
            f"Input file must be a PDB (.pdb) or mmCIF (.cif) file. Got: '{input_path}'"
        )
    if output_path.suffix.lower() not in {".pdb", ".cif"}:
        raise OutputFormatError(
            f"Output file must have extension .pdb or .cif. Got: '{output_path}'"
        )
    if output_path.exists() and not options.overwrite:
        raise OutputFormatError(
            f"{output_path} exists, rerun with overwrite enabled to replace it"
        )


def _create_numbering_plan(
    embeddings: QueryEmbeddings,
    options: RenumberOptions,
    reference_embeddings_name: str = DEFAULT_REFERENCE_EMBEDDINGS,
) -> _NumberingPlan:
    """Run alignment and ANARCI numbering for precomputed embeddings."""
    aligner = SoftAligner(
        embeddings_name=reference_embeddings_name,
        random_seed=options.random_seed,
    )

    try:
        alignment_result = aligner(
            embeddings,
            deterministic_loop_renumbering=options.deterministic_corrections,
            use_custom_gap_penalties=options.custom_gap_penalties,
            chain_type=options.chain_type,
        )
        hmm_output = alignment_matrix_to_state_vector(alignment_result.alignment)
    except Exception as exc:
        if isinstance(exc, AlignmentError):
            raise
        raise AlignmentError(f"Renumbering alignment failed: {exc}") from exc

    subsequence = build_anarci_subsequence(embeddings.sequence or "", hmm_output)
    anarci_alignment = number_from_alignment(
        hmm_output.states,
        subsequence,
        options.numbering_scheme,
        alignment_result.selected_chain_type,
    )

    return _NumberingPlan(
        alignment=anarci_alignment,
        chain_type=alignment_result.selected_chain_type,
        first_aligned_row=hmm_output.first_aligned_row,
        score=alignment_result.score,
    )


def renumber_file(
    input_path: str | Path,
    chain_id: str,
    output_path: str | Path,
    options: RenumberOptions | None = None,
) -> RenumberResult:
    """Renumber one chain in a structure file and write the result."""
    options = options or RenumberOptions()
    input_path = Path(input_path)
    output_path = Path(output_path)

    _validate_chain_id(chain_id)
    _validate_file_paths(input_path, output_path, options)

    try:
        embeddings = from_pdb(
            str(input_path),
            chain_id,
            residue_range=options.residue_range,
            random_seed=options.random_seed,
        )
    except (NotImplementedError, ValueError) as exc:
        raise InputStructureError(str(exc)) from exc

    plan = _create_numbering_plan(embeddings, options)
    changed_count = thread_alignment(
        str(input_path),
        chain_id,
        plan.alignment,
        str(output_path),
        alignment_start=plan.first_aligned_row,
        residue_range=options.residue_range,
    )

    return RenumberResult(
        output_path=output_path,
        chain_type=plan.chain_type,
        residue_count=len(embeddings.idxs),
        changed_residue_count=changed_count,
    )


def renumber_structure(
    structure: Structure.Structure,
    chain_id: str,
    options: RenumberOptions | None = None,
) -> Structure.Structure:
    """Renumber one chain in an in-memory BioPython structure."""
    options = options or RenumberOptions()
    _validate_chain_id(chain_id)

    try:
        chain = structure[0][chain_id]
    except KeyError as exc:
        available = [chain.id for chain in structure[0]]
        raise ChainNotFoundError(
            f"Chain '{chain_id}' not found in structure. Available chains: {available}"
        ) from exc

    try:
        embeddings = from_chain(
            chain,
            residue_range=options.residue_range,
            random_seed=options.random_seed,
        )
    except ValueError as exc:
        raise InputStructureError(str(exc)) from exc

    plan = _create_numbering_plan(embeddings, options)
    renumbered_structure, _changed_count = thread_numbering_onto_structure(
        structure,
        chain_id,
        plan.alignment,
        plan.first_aligned_row,
        residue_range=options.residue_range,
    )
    return renumbered_structure
