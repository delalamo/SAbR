"""Reference embedding loading boundary."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import as_file, files
from pathlib import Path

import numpy as np

from sabr.nn.config import EMBED_DIM
from sabr.validation import validate_array_shape

VALID_REFERENCE_LABELS = {"H", "K", "L"}
DEFAULT_REFERENCE_EMBEDDINGS = "embeddings.npz"
NOISE_REFERENCE_EMBEDDINGS = {
    "0.0": "embeddings_noise_0.0.npz",
    "0.2": "embeddings_noise_0.2.npz",
    "0.5": "embeddings_noise_0.5.npz",
    "1.0": "embeddings_noise_1.0.npz",
    "2.0": "embeddings_noise_2.0.npz",
}


@dataclass(frozen=True)
class ReferenceEmbeddings:
    """Reference embeddings keyed by IMGT position rather than residue id."""

    name: str
    embeddings: np.ndarray
    positions: list[int]

    def __post_init__(self) -> None:
        validate_array_shape(
            self.embeddings,
            0,
            len(self.positions),
            "embeddings",
            "len(positions)",
            f"Error raised for {self.name}",
        )
        validate_array_shape(
            self.embeddings,
            1,
            EMBED_DIM,
            "embeddings",
            "EMBED_DIM",
            f"Error raised for {self.name}",
        )

    @property
    def idxs(self) -> list[str]:
        """Return IMGT positions as strings for legacy internal callers."""
        return [str(position) for position in self.positions]


def resolve_reference_embeddings_name(noise_level: str | None) -> str:
    """Resolve a CLI noise-level option to a packaged reference embedding file."""
    if noise_level is None:
        return DEFAULT_REFERENCE_EMBEDDINGS
    try:
        return NOISE_REFERENCE_EMBEDDINGS[noise_level]
    except KeyError as exc:
        valid = ", ".join(sorted(NOISE_REFERENCE_EMBEDDINGS))
        raise ValueError(f"Noise level must be one of: {valid}.") from exc


def _load_split_reference_npz(path: Path) -> dict[str, ReferenceEmbeddings]:
    data = np.load(path, allow_pickle=True)
    if "arr_0" not in data:
        raise ValueError(
            f"Reference embeddings must use the split H/K/L schema: {path}"
        )

    split_data = data["arr_0"].item()
    labels = set(split_data)
    if labels != VALID_REFERENCE_LABELS:
        raise ValueError(
            "Reference embeddings must be labelled exactly H, K, and L. "
            f"Got labels: {sorted(labels)}."
        )

    return {
        label: ReferenceEmbeddings(
            name=label,
            embeddings=split_data[label]["array"],
            positions=[int(position) for position in split_data[label]["idxs"]],
        )
        for label in sorted(VALID_REFERENCE_LABELS)
    }


def load_reference_embeddings_from_npz(
    npz_path: str | Path,
) -> dict[str, ReferenceEmbeddings]:
    """Load split H/K/L reference embeddings from a concrete NPZ path."""
    return _load_split_reference_npz(Path(npz_path))


@lru_cache(maxsize=None)
def _load_packaged_reference_embeddings(
    embeddings_name: str,
    embeddings_path: str,
) -> tuple[tuple[str, ReferenceEmbeddings], ...]:
    path = files(embeddings_path) / embeddings_name
    with as_file(path) as concrete_path:
        embeddings = _load_split_reference_npz(concrete_path)
    return tuple(embeddings.items())


def load_reference_embeddings(
    embeddings_name: str = DEFAULT_REFERENCE_EMBEDDINGS,
    embeddings_path: str = "sabr.assets",
) -> dict[str, ReferenceEmbeddings]:
    """Load cached packaged split reference embeddings labelled H/K/L."""
    return dict(_load_packaged_reference_embeddings(embeddings_name, embeddings_path))
