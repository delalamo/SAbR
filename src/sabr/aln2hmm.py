#!/usr/bin/env python3

import logging
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class State:
    """Represents an HMM state with residue number and insertion code.

    This dataclass can be used like a tuple for backward compatibility.
    """

    residue_number: int
    insertion_code: str
    mapped_residue: Optional[int] = None

    def to_tuple(self) -> Tuple[Tuple[int, str], Optional[int]]:
        """Convert to ANARCI-compatible tuple format."""
        return ((self.residue_number, self.insertion_code), self.mapped_residue)

    def __iter__(self) -> Iterator:
        """Allow unpacking like a tuple for backward compatibility."""
        yield (self.residue_number, self.insertion_code)
        yield self.mapped_residue

    def __getitem__(self, index: int):
        """Allow indexing like a tuple for backward compatibility."""
        if index == 0:
            return (self.residue_number, self.insertion_code)
        elif index == 1:
            return self.mapped_residue
        else:
            raise IndexError(f"State index out of range: {index}")


def alignment_matrix_to_state_vector(
    matrix: np.ndarray,
) -> Tuple[List[State], int, int]:
    """Return an HMMER-style state vector from a binary alignment matrix."""

    if matrix.ndim != 2:
        raise ValueError("matrix must be 2D")
    LOGGER.info(f"Converting alignment matrix with shape {matrix.shape}")

    path = sorted(np.argwhere(np.transpose(matrix) == 1).tolist())
    assert len(path) > 0, "Alignment matrix contains no path"

    out = []

    for (b, a), (b2, a2) in zip(path[:-1], path[1:]):
        db, da = b2 - b, a2 - a

        # 1) Diagonal steps -> matches
        while db > 0 and da > 0:
            b += 1
            a += 1
            db -= 1
            da -= 1
            out.append(State(b, "m", a - 1))  # report pre-move A index

        # 2) A-only steps -> inserts (emit current A, then advance A)
        while da > 0:
            if b == max(bp for bp, _ in path):
                out.append(State(path[-1][0] + 1, "m", a))
                report_output(out)
                return out, path[0][0], a + 1 + path[0][0]
            out.append(State(b + 1, "i", a))
            a += 1
            da -= 1

        # 3) B-only steps -> deletes
        while db > 0:
            b += 1
            db -= 1
            out.append(State(b, "d", None))

    report_output(out)
    return out, path[0][0], path[-1][1] + path[0][0]


def report_output(out: List[State]) -> None:
    """Log each HMM state in ``out`` at INFO level."""
    LOGGER.info(f"Reporting {len(out)} HMM states")
    for idx, st in enumerate(out):
        if st.mapped_residue is None:
            LOGGER.info(
                f"{idx} (({st.residue_number}, '{st.insertion_code}'), None)"
            )
        else:
            LOGGER.info(
                f"{idx} (({st.residue_number}, '{st.insertion_code}'), "
                f"{st.mapped_residue})"
            )
