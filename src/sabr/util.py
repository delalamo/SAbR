#!/usr/bin/env python3

import logging
import pickle
from importlib.resources import files
from typing import Any, Dict

from Bio import SeqIO

LOGGER = logging.getLogger(__name__)


class JaxBackwardsCompatUnpickler(pickle.Unpickler):
    """Custom unpickler to handle deprecated JAX named_shape attribute."""

    def find_class(self, module: str, name: str) -> Any:
        """Override to wrap ShapedArray with a backwards-compatible version."""
        cls = super().find_class(module, name)
        if module == "jax.core" and name == "ShapedArray":
            return self._make_shaped_array_wrapper(cls)
        return cls

    @staticmethod
    def _make_shaped_array_wrapper(cls: type) -> type:
        """Create a wrapper that strips the deprecated named_shape kwarg."""

        class ShapedArrayWrapper(cls):
            def __new__(
                cls_inner, *args: Any, named_shape: Any = None, **kwargs: Any
            ) -> "ShapedArrayWrapper":
                del named_shape  # Discard deprecated attribute
                return super().__new__(cls_inner, *args, **kwargs)

        return ShapedArrayWrapper


def fetch_sequence_from_pdb(pdb_file: str, chain: str) -> str:
    """Return the sequence for chain in pdb_file without X residues."""
    for record in SeqIO.parse(pdb_file, "pdb-atom"):
        if record.id.endswith(chain):
            return str(record.seq).replace("X", "")
    ids = [r.id for r in SeqIO.parse(pdb_file, "pdb-atom")]
    raise ValueError(f"Chain {chain} not found in {pdb_file} (contains {ids})")


def read_softalign_params(
    params_name: str = "CONT_SW_05_T_3_1",
    params_path: str = "softalign.models",
) -> Dict[str, Any]:
    """Load SoftAlign parameters from package resources.

    Uses JaxBackwardsCompatUnpickler to handle deprecated JAX attributes.

    Args:
        params_name: Name of the model parameters file.
        params_path: Package path containing the parameters file.

    Returns:
        Dictionary containing the model parameters.
    """
    path = files(params_path) / params_name
    with open(path, "rb") as f:
        params = JaxBackwardsCompatUnpickler(f).load()
    LOGGER.info(f"Loaded model parameters from {path}")
    return params
