"""Residue identifiers and range helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass

_RESIDUE_ID_RE = re.compile(r"^\s*(-?\d+)([A-Za-z]*)\s*$")

AA_3TO1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


@dataclass(frozen=True, order=True)
class ResidueId:
    """Structure residue number plus optional insertion code."""

    number: int
    insertion_code: str = ""

    @classmethod
    def parse(cls, value: str | int) -> "ResidueId":
        """Parse residue IDs like ``52`` and ``52A``."""
        if isinstance(value, int):
            return cls(value)

        match = _RESIDUE_ID_RE.match(str(value))
        if match is None:
            raise ValueError(f"Invalid residue id: {value!r}")
        return cls(int(match.group(1)), match.group(2))

    def in_range(self, residue_range: "ResidueRange | None") -> bool:
        """Return whether this residue is selected by an inclusive range."""
        if residue_range is None:
            return True
        return residue_range.start <= self.number <= residue_range.end

    def __str__(self) -> str:
        return f"{self.number}{self.insertion_code}"


@dataclass(frozen=True)
class ResidueRange:
    """Inclusive numeric residue range in original structure numbering."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(
                f"Residue range end ({self.end}) must be >= start ({self.start})."
            )

    def contains(self, residue_id: ResidueId | str | int) -> bool:
        """Return whether a residue id is included by this range."""
        return ResidueId.parse(residue_id).in_range(self)

    def __str__(self) -> str:
        return f"{self.start}-{self.end}"
