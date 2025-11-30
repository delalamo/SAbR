#!/usr/bin/env python3

from Bio import SeqIO


def fetch_sequence_from_pdb(pdb_file: str, chain: str) -> str:
    """Return the sequence for chain in pdb_file without X residues."""
    for record in SeqIO.parse(pdb_file, "pdb-atom"):
        if record.id.endswith(chain):
            return str(record.seq).replace("X", "")
    ids = [r.id for r in SeqIO.parse(pdb_file, "pdb-atom")]
    raise ValueError(f"Chain {chain} not found in {pdb_file} (contains {ids})")
