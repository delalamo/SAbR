"""Exception hierarchy for SAbR."""


class SAbRError(Exception):
    """Base class for user-facing SAbR errors."""


class InputStructureError(SAbRError):
    """Raised when an input structure cannot be read or interpreted."""


class ChainNotFoundError(InputStructureError):
    """Raised when the requested chain is absent from a structure."""


class AlignmentError(SAbRError):
    """Raised when alignment fails or produces invalid output."""


class NumberingError(SAbRError):
    """Raised when ANARCI numbering fails."""


class OutputFormatError(SAbRError):
    """Raised when output format cannot represent the numbering result."""
