"""Public renumbering API for SAbR."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from Bio.PDB import Structure

from sabr.alignment.aln2hmm import alignment_matrix_to_state_vector
from sabr.alignment.backend import AlignmentBackend
from sabr.alignment.soft_aligner import SoftAligner
from sabr.embeddings.backend import EmbeddingBackend
from sabr.embeddings.mpnn import QueryEmbeddings, from_chain, from_pdb
from sabr.embeddings.references import ReferenceEmbeddings
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
from sabr.structure.io import read_structure
from sabr.structure.threading import (
    thread_alignment,
    thread_numbering_onto_structure,
)
from sabr.types import ChainType, parse_chain_type

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class NumberingPlan:
    """Internal result of alignment and numbering."""

    anarci_alignment: AnarciAlignment
    detected_chain_type: ChainType
    selected_reference: str
    first_aligned_row: int
    alignment_score: float


@dataclass(frozen=True)
class RenumberResult:
    """Result metadata returned by ``renumber_file``."""

    detected_chain_type: ChainType
    selected_reference: str
    first_aligned_row: int
    residue_count: int
    renumbered_count: int
    output_path: Path | None = None


ReferenceLoader = Callable[[str], dict[str, ReferenceEmbeddings]]
NumberingBackend = Callable[..., AnarciAlignment]


class Renumberer:
    """Orchestrates SAbR embedding, alignment, numbering, and threading."""

    def __init__(
        self,
        embedding_backend: EmbeddingBackend | None = None,
        alignment_backend: AlignmentBackend | None = None,
        reference_loader: ReferenceLoader | None = None,
        numbering_backend: NumberingBackend | None = None,
    ) -> None:
        self.embedding_backend = embedding_backend
        self.alignment_backend = alignment_backend
        self.reference_loader = reference_loader
        self.numbering_backend = numbering_backend or number_from_alignment

    def create_numbering_plan(
        self,
        embeddings: QueryEmbeddings,
        options: RenumberOptions,
    ) -> NumberingPlan:
        """Run alignment and ANARCI numbering for precomputed embeddings."""
        reference_embeddings = None
        if self.reference_loader is not None:
            reference_embeddings = self.reference_loader(options.reference_embeddings)

        aligner = SoftAligner(
            embeddings_name=options.reference_embeddings,
            random_seed=options.random_seed,
            backend=self.alignment_backend,
            reference_embeddings=reference_embeddings,
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

        detected_chain_type = parse_chain_type(alignment_result.chain_type)
        if detected_chain_type == "auto":
            raise AlignmentError("Alignment did not produce a concrete chain type.")

        subsequence = build_anarci_subsequence(embeddings.sequence or "", hmm_output)
        anarci_alignment = self.numbering_backend(
            hmm_output.states,
            subsequence,
            options.numbering_scheme,
            detected_chain_type,
        )

        return NumberingPlan(
            anarci_alignment=anarci_alignment,
            detected_chain_type=detected_chain_type,
            selected_reference=alignment_result.selected_reference,
            first_aligned_row=hmm_output.first_aligned_row,
            alignment_score=alignment_result.score,
        )

    def renumber_file(
        self,
        input_path: str | Path,
        chain_id: str,
        output_path: str | Path,
        options: RenumberOptions | None = None,
    ) -> RenumberResult:
        """Renumber one chain in a structure file and write the result."""
        options = options or RenumberOptions()
        input_path = Path(input_path)
        output_path = Path(output_path)

        self._validate_file_paths(input_path, output_path, options)

        try:
            embeddings = from_pdb(
                str(input_path),
                chain_id,
                residue_range=options.residue_range,
                random_seed=options.random_seed,
                backend=self.embedding_backend,
            )
        except ValueError as exc:
            raise InputStructureError(str(exc)) from exc

        numbering_plan = self.create_numbering_plan(embeddings, options)
        thread_alignment(
            str(input_path),
            chain_id,
            numbering_plan.anarci_alignment,
            str(output_path),
            0,
            len(numbering_plan.anarci_alignment),
            alignment_start=numbering_plan.first_aligned_row,
            residue_range=options.residue_range,
        )

        return RenumberResult(
            detected_chain_type=numbering_plan.detected_chain_type,
            selected_reference=numbering_plan.selected_reference,
            first_aligned_row=numbering_plan.first_aligned_row,
            residue_count=len(embeddings.idxs),
            renumbered_count=len(numbering_plan.anarci_alignment),
            output_path=output_path,
        )

    def renumber_structure(
        self,
        structure: Structure.Structure,
        chain_id: str,
        options: RenumberOptions | None = None,
    ) -> Structure.Structure:
        """Renumber one chain in an in-memory BioPython structure."""
        options = options or RenumberOptions()
        if len(chain_id) != 1:
            raise InputStructureError("Chain identifier must be exactly one character.")

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
                backend=self.embedding_backend,
            )
        except ValueError as exc:
            raise InputStructureError(str(exc)) from exc

        numbering_plan = self.create_numbering_plan(embeddings, options)
        renumbered_structure, _deviations = thread_numbering_onto_structure(
            structure,
            chain_id,
            numbering_plan.anarci_alignment,
            0,
            len(numbering_plan.anarci_alignment),
            numbering_plan.first_aligned_row,
            residue_range=options.residue_range,
        )
        return renumbered_structure

    @staticmethod
    def _validate_file_paths(
        input_path: Path,
        output_path: Path,
        options: RenumberOptions,
    ) -> None:
        if not input_path.exists():
            raise InputStructureError(f"Input file '{input_path}' does not exist.")
        if input_path.suffix.lower() not in {".pdb", ".cif"}:
            raise InputStructureError(
                f"Input file must be a PDB (.pdb) or mmCIF (.cif) file. "
                f"Got: '{input_path}'"
            )
        if output_path.suffix.lower() not in {".pdb", ".cif"}:
            raise OutputFormatError(
                f"Output file must have extension .pdb or .cif. Got: '{output_path}'"
            )
        if output_path.exists() and not options.overwrite:
            raise OutputFormatError(
                f"{output_path} exists, rerun with overwrite enabled to replace it"
            )


def renumber_file(
    input_path: str | Path,
    chain_id: str,
    output_path: str | Path,
    options: RenumberOptions | None = None,
) -> RenumberResult:
    """Renumber one chain in a structure file and write the result."""
    return Renumberer().renumber_file(input_path, chain_id, output_path, options)


def renumber_structure(
    structure: Structure.Structure,
    chain_id: str,
    options: RenumberOptions | None = None,
) -> Structure.Structure:
    """Renumber one chain in an in-memory BioPython structure."""
    return Renumberer().renumber_structure(structure, chain_id, options)


def load_structure(input_path: str | Path) -> Structure.Structure:
    """Read a structure file for callers that need BioPython objects."""
    return read_structure(str(input_path))
