#!/usr/bin/env python3
"""Convert pickled JAX model files to numpy npz format.

This script converts model parameter files from pickle format (which contains
JAX arrays) to numpy npz format (which is version-agnostic). This allows
the models to be loaded with any version of JAX.

Usage:
    # Must be run with JAX 0.4.x installed (the version models were saved with)
    pip install "jax>=0.4,<0.5" "jaxlib>=0.4,<0.5"
    python scripts/convert_models_to_npz.py

The script will:
1. Load each pickle file with the old JAX version
2. Convert all JAX arrays to numpy arrays
3. Save as .npz files alongside the originals
"""

import pickle
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np


def convert_to_numpy(obj: Any) -> Any:
    """Recursively convert JAX arrays to numpy arrays."""
    if isinstance(obj, dict):
        return {k: convert_to_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        converted = [convert_to_numpy(item) for item in obj]
        return type(obj)(converted)
    elif hasattr(obj, "__array__"):
        # JAX arrays, numpy arrays, etc.
        return np.asarray(obj)
    else:
        return obj


def convert_pickle_to_npz(pickle_path: Path) -> Path:
    """Convert a pickle file to npz format.

    Args:
        pickle_path: Path to the pickle file.

    Returns:
        Path to the created npz file.
    """
    print(f"Loading {pickle_path}...")
    with open(pickle_path, "rb") as f:
        params = pickle.load(f)

    print(f"  Converting to numpy...")
    params_np = convert_to_numpy(params)

    # Flatten nested dicts for npz storage
    flat_params = flatten_dict(params_np)

    npz_path = pickle_path.with_suffix(".npz")
    print(f"  Saving to {npz_path}...")
    np.savez(npz_path, **flat_params)

    # Verify the save
    loaded = dict(np.load(npz_path, allow_pickle=False))
    print(f"  Verified: {len(loaded)} arrays saved")

    return npz_path


def flatten_dict(d: Dict, parent_key: str = "", sep: str = "/") -> Dict:
    """Flatten a nested dictionary using separator in keys."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d: Dict, sep: str = "/") -> Dict:
    """Unflatten a dictionary with separator-joined keys."""
    result = {}
    for key, value in d.items():
        parts = key.split(sep)
        current = result
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
    return result


def main():
    # Find model files
    models_dir = Path(__file__).parent.parent / "external" / "SoftAlign" / "softalign" / "models"

    if not models_dir.exists():
        print(f"Error: Models directory not found: {models_dir}")
        sys.exit(1)

    # Files to convert
    pickle_files = [
        "CONT_SW_05_T_3_1",
        "CONT_SFT_06_T_3_1",
    ]

    # Also convert joblib pkl files (they use pickle internally)
    pkl_files = list(models_dir.glob("*.pkl"))

    print(f"Found {len(pkl_files)} .pkl files and {len(pickle_files)} parameter files")
    print()

    converted = []

    # Convert parameter files (no extension)
    for name in pickle_files:
        path = models_dir / name
        if path.exists():
            try:
                npz_path = convert_pickle_to_npz(path)
                converted.append(npz_path)
            except Exception as e:
                print(f"  Error converting {path}: {e}")
        else:
            print(f"  Skipping {name}: file not found")

    # Convert .pkl files
    for path in pkl_files:
        try:
            # These use joblib, need special handling
            import joblib
            print(f"Loading {path} (joblib)...")
            data = joblib.load(path)

            print(f"  Converting to numpy...")
            data_np = convert_to_numpy(data)

            flat_data = flatten_dict(data_np)

            npz_path = path.with_suffix(".npz")
            print(f"  Saving to {npz_path}...")
            np.savez(npz_path, **flat_data)

            loaded = dict(np.load(npz_path, allow_pickle=False))
            print(f"  Verified: {len(loaded)} arrays saved")
            converted.append(npz_path)
        except Exception as e:
            print(f"  Error converting {path}: {e}")

    print()
    print(f"Converted {len(converted)} files:")
    for p in converted:
        print(f"  {p}")


if __name__ == "__main__":
    main()
