"""Domain-level renumbering options."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from sabr.structure.residues import ResidueRange, normalize_residue_range
from sabr.types import (
    ChainType,
    NumberingScheme,
    parse_chain_type,
    parse_numbering_scheme,
    parse_reference_chain_type,
)


@dataclass(frozen=True)
class RenumberOptions:
    """Options for file and in-memory renumbering."""

    numbering_scheme: NumberingScheme = NumberingScheme.IMGT
    chain_type: ChainType | Literal["auto"] = "auto"
    reference_chain_type: ChainType | Literal["auto"] = "auto"
    deterministic_corrections: bool = True
    custom_gap_penalties: bool = True
    residue_range: ResidueRange | None = None
    random_seed: int = 0
    reference_embeddings: str = "embeddings.npz"
    overwrite: bool = False

    @classmethod
    def from_values(
        cls,
        numbering_scheme: str | NumberingScheme = NumberingScheme.IMGT,
        chain_type: str | ChainType = "auto",
        reference_chain_type: str | ChainType = "auto",
        deterministic_corrections: bool = True,
        custom_gap_penalties: bool = True,
        residue_range: ResidueRange | tuple[int, int] | None = None,
        random_seed: int = 0,
        reference_embeddings: str = "embeddings.npz",
        overwrite: bool = False,
    ) -> "RenumberOptions":
        """Build options from CLI/API primitive values."""
        return cls(
            numbering_scheme=parse_numbering_scheme(numbering_scheme),
            chain_type=parse_chain_type(chain_type),
            reference_chain_type=parse_reference_chain_type(reference_chain_type),
            deterministic_corrections=deterministic_corrections,
            custom_gap_penalties=custom_gap_penalties,
            residue_range=normalize_residue_range(residue_range),
            random_seed=random_seed,
            reference_embeddings=reference_embeddings,
            overwrite=overwrite,
        )
