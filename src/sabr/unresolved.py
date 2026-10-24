"""Use explicit unresolved-residue metadata between structural anchors.

Virtual rows survive scheme conversion so missing central insertions still
occupy their insertion codes. They never enter the encoder or output atoms.
Author-number jumps alone are not evidence of unresolved sequence.
"""

import logging

import numpy as np

from sabr import constants
from sabr.corrections import _aligned_row, correct_cdr_loop, correct_de_loop
from sabr.structure import _parent_residue_name

LOGGER = logging.getLogger(__name__)


def missing_residues_from_mmcif(data):
    """Normalize public MMCIF2Dict data to PDB missing-residue records.

    Author IDs must match the structure (MMCIFParser defaults). Only absent
    polymer residues are included; zero-occupancy coordinates are excluded.
    """
    prefix = "_pdbx_unobs_or_zero_occ_residues."
    names = ("auth_asym_id", "auth_seq_id", "auth_comp_id")
    records = []
    if all(prefix + name in data for name in names):
        count = len(data[prefix + names[0]])

        def value(name, index, default):
            return data.get(prefix + name, [default] * count)[index]

        for index in range(count):
            if value("polymer_flag", index, "Y") != "Y":
                continue
            # Zero occupancy is distinct from absent coordinates.
            if value("occupancy_flag", index, "1") != "1":
                continue
            try:
                number = int(value("auth_seq_id", index, "?"))
                model = value("PDB_model_num", index, "?")
                model = None if model in ("?", ".") else int(model)
            except ValueError:
                continue
            insertion = value("PDB_ins_code", index, "?")
            records.append(
                {
                    "chain": value("auth_asym_id", index, "?"),
                    "ssseq": number,
                    "res_name": value("auth_comp_id", index, "UNK"),
                    "insertion": (
                        None if insertion in ("?", ".") else insertion
                    ),
                    "model": model,
                }
            )
    return records


def missing_between(structure, chain, residue_ids, mmcif_dict=None):
    """Return amino acids explicitly declared missing between selected rows.

    PDBParser supplies REMARK 465 records in ``header['missing_residues']``.
    Optional MMCIF2Dict data supplies the equivalent mmCIF category.
    Ignore other models/chains and ambiguous author IDs.
    """
    model = next(iter(structure))
    missing = {}
    invalid = set()
    observed = set(residue_ids)
    records = (
        missing_residues_from_mmcif(mmcif_dict)
        if mmcif_dict is not None
        else structure.header.get("missing_residues", [])
    )
    for record in records:
        if record.get("chain") != chain or record.get("model") not in (
            None,
            model.serial_num,
        ):
            continue
        try:
            key = (
                int(record["ssseq"]),
                (record.get("insertion") or "").strip(),
            )
        except (KeyError, TypeError, ValueError):
            continue
        parent = _parent_residue_name(record.get("res_name", ""))
        amino_acid = constants.AA_3TO1.get(parent, "X")
        if key in observed or (key in missing and missing[key] != amino_acid):
            invalid.add(key)
        missing[key] = amino_acid
    result = {}
    for row, (left, right) in enumerate(zip(residue_ids, residue_ids[1:])):
        if left >= right or any(left <= key <= right for key in invalid):
            continue
        sequence = "".join(
            missing[key] for key in sorted(missing) if left < key < right
        )
        if sequence:
            result[row] = sequence
    return result


def expand_loop_alignment(alignment, sequence, gaps, missing):
    """Correct eligible loops on virtual sequence, returning observed row map.

    Both exact anchors must exist. Every detected break in a region must be
    explained by metadata; otherwise preserve that region's learned alignment.
    Only anchored regions are expanded, so terminal metadata and linkers do
    not extend domain spans or alter candidate selection.
    """
    regions = []
    used = set()
    for start in range(0, alignment.shape[1], constants.IMGT_MAX_POSITION):
        domain = alignment[:, start : start + constants.IMGT_MAX_POSITION]
        anchors = {**constants.CDR_ANCHORS, "DE": (79, 85)}
        for name, (left, right) in anchors.items():
            first = _aligned_row(domain, left - 1)
            last = _aligned_row(domain, right - 1)
            if first is None or last is None or first >= last:
                continue
            boundaries = set(range(first, last))
            relevant = boundaries.intersection(missing)
            if not relevant or (boundaries.intersection(gaps) - missing.keys()):
                continue
            regions.append((start, name))
            used.update(relevant)
    if not used:
        return alignment, sequence, tuple(range(len(sequence)))
    rows = []
    letters = []
    observed_rows = []
    for row, letter in enumerate(sequence):
        rows.append(alignment[row])
        letters.append(letter)
        observed_rows.append(row)
        if row in used:
            for amino_acid in missing[row]:
                rows.append(np.zeros(alignment.shape[1], dtype=alignment.dtype))
                letters.append(amino_acid)
                observed_rows.append(None)
    expanded = np.asarray(rows)
    for start, name in regions:
        LOGGER.info(
            "Applying %s correction with unresolved-residue metadata "
            "in domain %d.",
            name,
            start // constants.IMGT_MAX_POSITION + 1,
        )
        domain = expanded[:, start : start + constants.IMGT_MAX_POSITION]
        if name == "DE":
            correct_de_loop(domain)
        else:
            correct_cdr_loop(domain, name, *constants.IMGT_LOOPS[name])
    return expanded, "".join(letters), tuple(observed_rows)
