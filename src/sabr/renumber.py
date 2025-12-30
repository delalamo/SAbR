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
from typing import Optional, Tuple

from ANARCI import anarci
from Bio.PDB import Chain, Model, Structure

from sabr import aln2hmm, edit_pdb, mpnn_embeddings, softaligner, util
from sabr.constants import AnarciAlignment

LOGGER = logging.getLogger(__name__)


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

    hmm_output = aln2hmm.alignment_matrix_to_state_vector(
        alignment_result.alignment
    )

    n_aligned = hmm_output.imgt_end - hmm_output.imgt_start
    subsequence = "-" * hmm_output.imgt_start + sequence[:n_aligned]

    # Detect chain type from alignment if not specified
    if chain_type == "auto":
        chain_type = util.detect_chain_type(alignment_result.alignment)
    else:
        LOGGER.info(f"Using specified chain type: {chain_type}")

    anarci_out, start_res, end_res = anarci.number_sequence_from_alignment(
        hmm_output.states,
        subsequence,
        scheme=numbering_scheme,
        chain_type=chain_type,
    )

    # Remove gap positions
    anarci_out = [a for a in anarci_out if a[1] != "-"]

    return anarci_out, chain_type, hmm_output.first_aligned_row


def _thread_structure(
    structure: Structure.Structure,
    chain_id: str,
    anarci_alignment: AnarciAlignment,
    alignment_start: int,
) -> Structure.Structure:
    """Thread ANARCI alignment onto a BioPython structure.

    This function creates a new structure with renumbered residues,
    without writing to disk.

    Args:
        structure: BioPython Structure object.
        chain_id: Chain identifier to renumber.
        anarci_alignment: ANARCI alignment output.
        alignment_start: Offset where alignment begins.

    Returns:
        New structure with renumbered residues.
    """
    new_structure = Structure.Structure("renumbered_structure")
    new_model = Model.Model(0)

    for chain in structure[0]:
        if chain.id != chain_id:
            # Copy non-target chains as-is
            new_chain = copy.deepcopy(chain)
            new_chain.detach_parent()
            new_model.add(new_chain)
        else:
            # Renumber the target chain
            new_chain, _ = edit_pdb.thread_onto_chain(
                chain,
                anarci_alignment,
                0,  # start_res
                len(anarci_alignment),  # end_res
                alignment_start,
            )
            new_model.add(new_chain)

    new_structure.add(new_model)
    return new_structure


def _extract_chain_subset(
    structure: Structure.Structure,
    chain_id: str,
    res_start: Optional[int],
    res_end: Optional[int],
) -> Structure.Structure:
    """Extract a subset of residues from a structure based on range.

    Args:
        structure: BioPython Structure object.
        chain_id: Chain identifier to extract from.
        res_start: Starting residue number (inclusive). None = from beginning.
        res_end: Ending residue number (inclusive). None = to end.

    Returns:
        New structure with only the specified residue range.
    """
    new_structure = Structure.Structure("subset_structure")
    new_model = Model.Model(0)

    for chain in structure[0]:
        if chain.id != chain_id:
            # Copy non-target chains as-is
            new_chain = copy.deepcopy(chain)
            new_chain.detach_parent()
            new_model.add(new_chain)
        else:
            # Extract subset of target chain
            new_chain = Chain.Chain(chain.id)
            for res in chain.get_residues():
                res_num = res.get_id()[1]
                # Check if residue is in range
                if res_start is not None and res_num < res_start:
                    continue
                if res_end is not None and res_num > res_end:
                    continue
                new_res = copy.deepcopy(res)
                new_res.detach_parent()
                new_chain.add(new_res)
                new_res.parent = new_chain
            new_model.add(new_chain)

    new_structure.add(new_model)
    return new_structure


def renumber_structure(
    structure: Structure.Structure,
    chain: str,
    numbering_scheme: str = "imgt",
    chain_type: str = "auto",
    res_start: Optional[int] = None,
    res_end: Optional[int] = None,
    deterministic_loop_renumbering: bool = True,
) -> Structure.Structure:
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
        res_start: Starting residue number to renumber (inclusive).
            If None, starts from the first residue.
        res_end: Ending residue number to renumber (inclusive).
            If None, processes to the last residue.
        deterministic_loop_renumbering: Apply deterministic corrections
            for loop regions (FR1, DE loop, CDRs). Default True.

    Returns:
        Renumbered BioPython Structure object.

    Raises:
        ValueError: If chain is not found or is not a single character.

    Example:
        >>> from Bio.PDB import PDBParser
        >>> from sabr import renumber
        >>> parser = PDBParser(QUIET=True)
        >>> structure = parser.get_structure("ab", "antibody.pdb")
        >>> renumbered = renumber.renumber_structure(structure, chain="H")
        >>> # Or with a specific residue range:
        >>> renumbered = renumber.renumber_structure(
        ...     structure, chain="H", res_start=1, res_end=128
        ... )
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

    # Extract subset if range specified
    if res_start is not None or res_end is not None:
        working_structure = _extract_chain_subset(
            structure, chain, res_start, res_end
        )
    else:
        working_structure = structure

    # Get the chain object and generate embeddings
    chain_obj = working_structure[0][chain]
    embeddings = mpnn_embeddings.from_chain(chain_obj)

    LOGGER.info(
        f"Processing chain {chain} with {len(embeddings.idxs)} residues"
    )

    # Run the renumbering pipeline
    anarci_alignment, detected_chain_type, first_aligned_row = (
        run_renumbering_pipeline(
            embeddings,
            numbering_scheme=numbering_scheme,
            chain_type=chain_type,
            deterministic_loop_renumbering=deterministic_loop_renumbering,
        )
    )

    # Thread the alignment onto the structure
    renumbered_structure = _thread_structure(
        working_structure,
        chain,
        anarci_alignment,
        first_aligned_row,
    )

    return renumbered_structure
