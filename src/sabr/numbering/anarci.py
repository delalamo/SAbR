"""Small ANARCI boundary for SAbR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ANARCI import anarci

from sabr.alignment.aln2hmm import Aln2HmmOutput, State
from sabr.errors import NumberingError
from sabr.types import ChainType, NumberingScheme, chain_type_value


@dataclass(frozen=True)
class NumberedResidue:
    """A single residue numbered by ANARCI."""

    position: int
    insertion_code: str
    amino_acid: str


AnarciAlignment = list[tuple[tuple[int, str], str]]


def build_anarci_subsequence(sequence: str, hmm: Aln2HmmOutput) -> str:
    """Build the padded subsequence consumed by ANARCI."""
    return "-" * hmm.imgt_start + sequence[hmm.first_aligned_row :]


def number_from_alignment(
    states: Iterable[State],
    subsequence: str,
    scheme: NumberingScheme,
    chain_type: ChainType,
) -> AnarciAlignment:
    """Number a sequence from HMM states through ANARCI."""
    try:
        anarci_out, _start_res, _end_res = anarci.number_sequence_from_alignment(
            list(states),
            subsequence,
            scheme=scheme.value,
            chain_type=chain_type_value(chain_type),
        )
    except Exception as exc:  # pragma: no cover - ANARCI owns internals
        raise NumberingError(f"ANARCI numbering failed: {exc}") from exc

    return [item for item in anarci_out if item[1] != "-"]
