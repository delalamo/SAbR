"""Structure-based Antibody Renumbering (SAbR)."""

from pathlib import Path as _Path

# Dynamically load README as module docstring for pdoc
_readme = _Path(__file__).parent.parent.parent / "README.md"
if _readme.exists():
    __doc__ = _readme.read_text()
