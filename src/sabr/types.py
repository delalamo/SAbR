"""Shared domain types for SAbR."""

from __future__ import annotations

from enum import Enum
from typing import Literal

AutoChainType = Literal["auto"]


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


def parse_chain_type(value: str | ChainType) -> ChainType | AutoChainType:
    """Parse user input into a chain type or ``"auto"``."""
    if isinstance(value, ChainType):
        return value

    normalized = value.strip().lower()
    if normalized == "auto":
        return "auto"
    if normalized.endswith("_h"):
        return ChainType.HEAVY
    if normalized.endswith("_k"):
        return ChainType.KAPPA
    if normalized.endswith("_l"):
        return ChainType.LAMBDA

    aliases = {
        "h": ChainType.HEAVY,
        "heavy": ChainType.HEAVY,
        "k": ChainType.KAPPA,
        "kappa": ChainType.KAPPA,
        "l": ChainType.LAMBDA,
        "lambda": ChainType.LAMBDA,
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(
            "Chain type must be one of H, K, L, heavy, kappa, lambda, or auto."
        ) from exc


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


def chain_type_value(value: ChainType | AutoChainType) -> str:
    """Return the ANARCI/reference string representation for a chain type."""
    if value == "auto":
        return "auto"
    return value.value


def numbering_scheme_value(value: NumberingScheme) -> str:
    """Return the string representation expected by ANARCI."""
    return value.value
