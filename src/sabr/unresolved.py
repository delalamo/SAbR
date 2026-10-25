"""Use explicit unresolved-residue metadata between structural anchors.

Zero-embedding ghost rows survive scheme conversion so central insertions still
occupy their insertion codes. They never enter the encoder or output atoms.
Author-number jumps alone are not evidence of unresolved sequence.
"""

import numpy as np

from sabr import constants
from sabr.structure import _parent_residue_name


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


def insert_ghost_embeddings(embeddings, sequence, gaps, missing):
    """Insert zero vectors after encoding observed coordinates.

    Metadata-backed breaks become ordinary sequence positions for alignment
    and deterministic corrections. Unexplained breaks retain their opening
    discount and correction fallback, with indices shifted into the expanded
    query. Ghosts survive scheme conversion and are then discarded using the
    returned expanded-row-to-observed-row map. No coordinates are fabricated.
    """
    if not missing:
        return embeddings, sequence, gaps, tuple(range(len(sequence)))
    size = len(sequence) + sum(len(letters) for letters in missing.values())
    expanded = np.zeros((size, embeddings.shape[1]), dtype=embeddings.dtype)
    letters = []
    observed_rows = []
    remaining_gaps = set()
    for row, amino_acid in enumerate(sequence):
        expanded_row = len(letters)
        expanded[expanded_row] = embeddings[row]
        letters.append(amino_acid)
        observed_rows.append(row)
        if row in gaps and row not in missing:
            remaining_gaps.add(expanded_row)
        unresolved = missing.get(row, "")
        letters.extend(unresolved)
        observed_rows.extend([None] * len(unresolved))
    return (
        expanded,
        "".join(letters),
        frozenset(remaining_gaps),
        tuple(observed_rows),
    )
