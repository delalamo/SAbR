"""Shared domain types for SAbR."""

from __future__ import annotations

from enum import Enum


class ChainType(Enum):
    """Antibody chain type accepted by ANARCI."""

    HEAVY = "H"
    KAPPA = "K"
    LAMBDA = "L"


class NumberingScheme(Enum):
    """Supported antibody numbering schemes."""

    IMGT = "imgt"
    CHOTHIA = "chothia"
    KABAT = "kabat"
    MARTIN = "martin"
    AHO = "aho"
    WOLF_GUY = "wolfguy"


def parse_chain_type(value: str | ChainType | None) -> ChainType | None:
    """Parse user input into a chain type, with ``None`` meaning auto."""
    if value is None:
        return None
    if isinstance(value, ChainType):
        return value

    normalized = value.strip().lower()
    if normalized == "auto":
        return None

    aliases = {
        "h": ChainType.HEAVY,
        "k": ChainType.KAPPA,
        "l": ChainType.LAMBDA,
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError("Chain type must be one of H, K, L, or auto.") from exc


def parse_numbering_scheme(value: str | NumberingScheme) -> NumberingScheme:
    """Parse user input into a numbering scheme enum."""
    if isinstance(value, NumberingScheme):
        return value

    normalized = value.strip().lower()
    for scheme in NumberingScheme:
        if scheme.value == normalized:
            return scheme

    valid = ", ".join(scheme.value for scheme in NumberingScheme)
    raise ValueError(f"Numbering scheme must be one of: {valid}.")


def chain_type_value(value: ChainType) -> str:
    """Return the ANARCI/reference string representation for a chain type."""
    return value.value
