from importlib.resources import as_file, files

import numpy as np


def load_data(name: str) -> np.ndarray:
    """Load a data file from the package's data directory."""
    with as_file(files("sabr.assets.embeddings") / name) as path:
        return np.load(path, allow_pickle=True)
