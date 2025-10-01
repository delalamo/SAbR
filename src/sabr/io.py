from importlib.resources import as_file, files
from typing import Optional

import numpy as np


def load_data(name: str, package: Optional[str] = "sabr.assets") -> np.ndarray:
    """Load a numpy array from the package's assets."""
    root = files(package) if isinstance(package, str) else package
    with as_file(root / name) as path:
        return np.load(path, allow_pickle=True)
