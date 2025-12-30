from pathlib import Path

# Load README as module docstring for pdoc homepage
# README.md is copied to package dir during RTD build (see .readthedocs.yaml)
_readme = Path(__file__).resolve().parent / "README.md"
if _readme.exists():
    __doc__ = _readme.read_text(encoding="utf-8")
else:
    __doc__ = """Structure-based Antibody Renumbering (SAbR)."""
