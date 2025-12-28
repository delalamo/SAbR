#!/usr/bin/env python3
"""Structure renumbering module for programmatic access.

This module provides functions for renumbering antibody structures using
the SAbR pipeline. Unlike the CLI, these functions work directly with
BioPython Structure objects and return renumbered structures in memory.

Key functions:
- renumber_structure: Main entry point for renumbering a BioPython structure
- run_renumbering_pipeline: Core pipeline logic shared with CLI

Example usage:
    from Bio.PDB import PDBParser
    from sabr import renumber

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("antibody", "input.pdb")

    renumbered = renumber.renumber_structure(
        structure,
        chain="A",
        numbering_scheme="imgt",
    )
"""

import copy
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from ANARCI import anarci
from Bio.PDB import Chain, Model, Structure

from sabr import aln2hmm, edit_pdb, mpnn_embeddings, softaligner, util
from sabr.constants import AA_3TO1, AnarciAlignment

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RenumberingResult:
    """Result of the renumbering pipeline.

    Attributes:
        structure: The renumbered BioPython Structure object.
        chain_type: Detected or specified chain type ("H", "K", or "L").
        deviations: Number of residue IDs that changed during renumbering.
        sequence: The amino acid sequence of the renumbered chain.
        anarci_alignment: The ANARCI alignment used for renumbering.
    """

    structure: Structure.Structure
    chain_type: str
    deviations: int
    sequence: str
    anarci_alignment: AnarciAlignment


def _extract_sequence_from_chain(chain: Chain.Chain, max_residues: int = 0) -> str:
    """Extract amino acid sequence from a BioPython chain.

    Args:
        chain: BioPython Chain object.
        max_residues: Maximum residues to extract (0 = all).

    Returns:
        Single-letter amino acid sequence.
    """
    residues = []
    for idx, res in enumerate(chain.get_residues()):
        if max_residues > 0 and idx >= max_residues:
            break
        # Skip HETATM records
        if res.get_id()[0].strip() != "":
            continue
        resname = res.get_resname()
        if resname in AA_3TO1:
            residues.append(AA_3TO1[resname])
    return "".join(residues)


def run_renumbering_pipeline(
    embeddings: mpnn_embeddings.MPNNEmbeddings,
    numbering_scheme: str = "imgt",
    chain_type: str = "auto",
    deterministic_loop_renumbering: bool = True,
) -> Tuple[AnarciAlignment, str, int]:
    """Run the core renumbering pipeline.

    This function encapsulates the alignment and ANARCI numbering steps
    that are shared between the CLI and programmatic API.

    Args:
        embeddings: MPNN embeddings for the structure chain.
        numbering_scheme: Numbering scheme (imgt, chothia, kabat, etc.).
        chain_type: Chain type ("H", "K", "L", or "auto").
        deterministic_loop_renumbering: Apply deterministic corrections.

    Returns:
        Tuple of (anarci_alignment, detected_chain_type, first_aligned_row).
    """
    sequence = embeddings.sequence

    aligner = softaligner.SoftAligner()
    alignment_result = aligner(
        embeddings,
        deterministic_loop_renumbering=deterministic_loop_renumbering,
    )

    state_vector, imgt_start, imgt_end, first_aligned_row = (
        aln2hmm.alignment_matrix_to_state_vector(alignment_result.alignment)
    )

    n_aligned = imgt_end - imgt_start
    subsequence = "-" * imgt_start + sequence[:n_aligned]

    # Detect chain type from alignment if not specified
    if chain_type == "auto":
        chain_type = util.detect_chain_type(alignment_result.alignment)
    else:
        LOGGER.info(f"Using specified chain type: {chain_type}")

    anarci_out, start_res, end_res = anarci.number_sequence_from_alignment(
        state_vector,
        subsequence,
        scheme=numbering_scheme,
        chain_type=chain_type,
    )

    # Remove gap positions
    anarci_out = [a for a in anarci_out if a[1] != "-"]

    return anarci_out, chain_type, first_aligned_row


def _thread_structure(
    structure: Structure.Structure,
    chain_id: str,
    anarci_alignment: AnarciAlignment,
    alignment_start: int,
    max_residues: int = 0,
) -> Tuple[Structure.Structure, int]:
    """Thread ANARCI alignment onto a BioPython structure.

    This function creates a new structure with renumbered residues,
    without writing to disk.

    Args:
        structure: BioPython Structure object.
        chain_id: Chain identifier to renumber.
        anarci_alignment: ANARCI alignment output.
        alignment_start: Offset where alignment begins.
        max_residues: Maximum residues to process (0 = all).

    Returns:
        Tuple of (new_structure, total_deviations).
    """
    new_structure = Structure.Structure("renumbered_structure")
    new_model = Model.Model(0)
    total_deviations = 0

    for chain in structure[0]:
        if chain.id != chain_id:
            # Copy non-target chains as-is
            new_chain = copy.deepcopy(chain)
            new_chain.detach_parent()
            new_model.add(new_chain)
        else:
            # Renumber the target chain
            new_chain, deviations = edit_pdb.thread_onto_chain(
                chain,
                anarci_alignment,
                0,  # start_res
                len(anarci_alignment),  # end_res
                alignment_start,
                max_residues,
            )
            new_model.add(new_chain)
            total_deviations += deviations

    new_structure.add(new_model)
    return new_structure, total_deviations


def renumber_structure(
    structure: Structure.Structure,
    chain: str,
    numbering_scheme: str = "imgt",
    chain_type: str = "auto",
    max_residues: int = 0,
    deterministic_loop_renumbering: bool = True,
) -> RenumberingResult:
    """Renumber an antibody structure using SAbR.

    This is the main entry point for programmatic renumbering. It takes
    a BioPython Structure object and returns a renumbered structure
    without writing to disk.

    Args:
        structure: BioPython Structure object to renumber.
        chain: Chain identifier to renumber (single character).
        numbering_scheme: Numbering scheme to apply. Options:
            "imgt", "chothia", "kabat", "martin", "aho", "wolfguy".
        chain_type: Expected chain type for ANARCI. Options:
            "H" (heavy), "K" (kappa), "L" (lambda), "auto" (detect).
        max_residues: Maximum residues to process. If 0, process all.
        deterministic_loop_renumbering: Apply deterministic corrections
            for loop regions (FR1, DE loop, CDRs). Default True.

    Returns:
        RenumberingResult containing the renumbered structure and metadata.

    Raises:
        ValueError: If chain is not found or is not a single character.

    Example:
        >>> from Bio.PDB import PDBParser
        >>> from sabr import renumber
        >>> parser = PDBParser(QUIET=True)
        >>> structure = parser.get_structure("ab", "antibody.pdb")
        >>> result = renumber.renumber_structure(structure, chain="H")
        >>> renumbered_structure = result.structure
        >>> print(f"Chain type: {result.chain_type}")
        >>> print(f"Deviations: {result.deviations}")
    """
    if len(chain) != 1:
        raise ValueError("Chain identifier must be exactly one character.")

    # Verify chain exists
    chain_obj = None
    for ch in structure[0]:
        if ch.id == chain:
            chain_obj = ch
            break

    if chain_obj is None:
        available = [ch.id for ch in structure[0]]
        raise ValueError(
            f"Chain '{chain}' not found in structure. "
            f"Available chains: {available}"
        )

    # Extract sequence for embedding generation
    sequence = _extract_sequence_from_chain(chain_obj, max_residues)
    LOGGER.info(f"Extracted sequence (len {len(sequence)}): {sequence}")

    # Generate embeddings from structure
    embeddings = mpnn_embeddings.from_structure(structure, chain, max_residues)

    # Run the renumbering pipeline
    anarci_alignment, detected_chain_type, first_aligned_row = run_renumbering_pipeline(
        embeddings,
        numbering_scheme=numbering_scheme,
        chain_type=chain_type,
        deterministic_loop_renumbering=deterministic_loop_renumbering,
    )

    # Thread the alignment onto the structure
    renumbered_structure, deviations = _thread_structure(
        structure,
        chain,
        anarci_alignment,
        first_aligned_row,
        max_residues,
    )

    return RenumberingResult(
        structure=renumbered_structure,
        chain_type=detected_chain_type,
        deviations=deviations,
        sequence=sequence,
        anarci_alignment=anarci_alignment,
    )
