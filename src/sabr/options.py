"""Domain-level renumbering options."""

from __future__ import annotations

from dataclasses import dataclass

from sabr.structure.residues import ResidueRange
from sabr.types import (
    ChainType,
    NumberingScheme,
    parse_chain_type,
    parse_numbering_scheme,
)


@dataclass(frozen=True)
class RenumberOptions:
    """Options for file and in-memory renumbering."""

    numbering_scheme: NumberingScheme = NumberingScheme.IMGT
    chain_type: ChainType | None = None
    deterministic_corrections: bool = True
    custom_gap_penalties: bool = True
    residue_range: ResidueRange | None = None
    random_seed: int = 0
    overwrite: bool = False

    def __post_init__(self) -> None:
        if self.residue_range is not None and not isinstance(
            self.residue_range, ResidueRange
        ):
            raise ValueError("residue_range must be a ResidueRange or None.")
        if self.chain_type is not None and not isinstance(self.chain_type, ChainType):
            raise ValueError("chain_type must be ChainType or None.")

    @classmethod
    def from_values(
        cls,
        numbering_scheme: str | NumberingScheme = NumberingScheme.IMGT,
        chain_type: str | ChainType | None = None,
        deterministic_corrections: bool = True,
        custom_gap_penalties: bool = True,
        residue_range: ResidueRange | None = None,
        random_seed: int = 0,
        overwrite: bool = False,
    ) -> "RenumberOptions":
        """Build options from CLI/API primitive values."""
        if residue_range is not None and not isinstance(residue_range, ResidueRange):
            raise ValueError("residue_range must be a ResidueRange or None.")
        return cls(
            numbering_scheme=parse_numbering_scheme(numbering_scheme),
            chain_type=parse_chain_type(chain_type),
            deterministic_corrections=deterministic_corrections,
            custom_gap_penalties=custom_gap_penalties,
            residue_range=residue_range,
            random_seed=random_seed,
            overwrite=overwrite,
        )
