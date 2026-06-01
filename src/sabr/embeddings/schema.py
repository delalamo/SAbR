"""NPZ schemas for query embeddings."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

QUERY_SCHEMA = "query-v1"

if TYPE_CHECKING:
    from sabr.embeddings.mpnn import QueryEmbeddings


def save_query_embeddings(embeddings: "QueryEmbeddings", output_path: str) -> None:
    """Save query embeddings using the documented schema."""
    np.savez(
        Path(output_path),
        schema=QUERY_SCHEMA,
        name=embeddings.name,
        embeddings=embeddings.embeddings,
        idxs=np.array(embeddings.idxs),
        sequence=embeddings.sequence if embeddings.sequence else "",
    )


def load_query_embeddings(npz_file: str) -> "QueryEmbeddings":
    """Load query embeddings, including older SAbR query files."""
    from sabr.embeddings.mpnn import QueryEmbeddings

    input_path = Path(npz_file)
    data = np.load(input_path, allow_pickle=True)

    if "embeddings" in data:
        vectors = data["embeddings"]
    elif "array" in data:
        vectors = data["array"]
    else:
        raise ValueError(
            f"Embedding file {input_path} has no 'embeddings' or 'array' key."
        )

    sequence = str(data["sequence"]) if "sequence" in data else ""
    return QueryEmbeddings(
        name=str(data["name"]) if "name" in data else input_path.stem,
        embeddings=vectors,
        idxs=[str(x) for x in data["idxs"]],
        sequence=sequence or None,
    )
