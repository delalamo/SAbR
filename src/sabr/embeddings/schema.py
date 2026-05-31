"""NPZ schemas for query and reference embeddings."""

from __future__ import annotations

from enum import Enum
from importlib.resources import as_file, files
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sabr.embeddings.mpnn import MPNNEmbeddings


class EmbeddingSchemaVersion(Enum):
    """Known embedding NPZ schema variants."""

    QUERY_V1 = "query-v1"
    REFERENCE_UNIFIED_V1 = "reference-unified-v1"
    REFERENCE_SPLIT_V1 = "reference-split-v1"


def save_query_embeddings(embeddings: "MPNNEmbeddings", output_path: str) -> None:
    """Save query embeddings using the documented schema."""
    np.savez(
        Path(output_path),
        schema=EmbeddingSchemaVersion.QUERY_V1.value,
        name=embeddings.name,
        embeddings=embeddings.embeddings,
        idxs=np.array(embeddings.idxs),
        sequence=embeddings.sequence if embeddings.sequence else "",
    )


def load_query_embeddings(npz_file: str) -> "MPNNEmbeddings":
    """Load query embeddings, including older SAbR query files."""
    from sabr.embeddings.mpnn import MPNNEmbeddings

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
    return MPNNEmbeddings(
        name=str(data["name"]) if "name" in data else input_path.stem,
        embeddings=vectors,
        idxs=[str(x) for x in data["idxs"]],
        sequence=sequence or None,
    )


def load_reference_embeddings(
    embeddings_name: str = "embeddings.npz",
    embeddings_path: str = "sabr.assets",
) -> dict[str, "MPNNEmbeddings"]:
    """Load packaged reference embeddings in all supported schemas."""
    from sabr.embeddings.mpnn import MPNNEmbeddings

    path = files(embeddings_path) / embeddings_name
    with as_file(path) as concrete_path:
        data = np.load(concrete_path, allow_pickle=True)

        if "array" in data:
            embedding = MPNNEmbeddings(
                name=str(data["name"]) if "name" in data else "unified",
                embeddings=data["array"],
                idxs=[str(x) for x in data["idxs"]],
            )
            return {"unified": embedding}

        if "arr_0" in data:
            split_data = data["arr_0"].item()
            embeddings = {}
            for chain_type in ["H", "K", "L"]:
                chain_data = split_data[chain_type]
                embeddings[chain_type] = MPNNEmbeddings(
                    name=chain_type,
                    embeddings=chain_data["array"],
                    idxs=[str(x) for x in chain_data["idxs"]],
                )
            return embeddings

    raise ValueError(f"Unsupported reference embedding schema: {path}")
