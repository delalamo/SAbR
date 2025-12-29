"""Structure-based Antibody Renumbering (SAbR)."""

from pathlib import Path as _Path

# Load README as module docstring for pdoc homepage
_readme_path = _Path(__file__).resolve().parent.parent.parent / "README.md"
if _readme_path.exists():
    __doc__ = _readme_path.read_text(encoding="utf-8")
