# Structure-based Antibody Renumbering

[![Tests](https://github.com/delalamo/SAbR/actions/workflows/test.yml/badge.svg)](https://github.com/delalamo/SAbR/actions/workflows/test.yml)
[![Code Formatting](https://github.com/delalamo/SAbR/actions/workflows/format.yml/badge.svg)](https://github.com/delalamo/SAbR/actions/workflows/format.yml)
[![Documentation](https://readthedocs.org/projects/sabr/badge/?version=latest)](https://sabr.readthedocs.io/en/latest/)
[![PyPI version](https://img.shields.io/pypi/v/sabr-kit.svg)](https://pypi.org/project/sabr-kit/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

SAbR renumbers antibody structure files from 3D coordinates. It computes
structure embeddings, aligns them to antibody reference embeddings, converts the
alignment to ANARCI-compatible states, and writes a renumbered PDB or mmCIF
structure.

Full documentation is available at [sabr.readthedocs.io](https://sabr.readthedocs.io/).

## Installation

```bash
pip install sabr-kit
```

For source installs:

```bash
git clone --recursive https://github.com/delalamo/SAbR.git
cd SAbR
pip install -e .[test,docs]
```

## CLI

```bash
sabr -i input.pdb -c H -o output.cif --chain-type H
```

Use `.cif` output when long CDR insertions may require multi-character
insertion codes. Custom gap penalties are CDR-only: by default SAbR sets
gap-open penalties to zero for IMGT CDR positions and keeps gap-extension
penalties normal.

By default, `--chain-type auto` tries the labelled `H`, `K`, and `L` reference
embeddings and passes the best-scoring label to ANARCI. Passing `--chain-type H`,
`--chain-type K`, or `--chain-type L` restricts alignment to that labelled
embedding and passes the same label to ANARCI.

## Python API

```python
from pathlib import Path

from sabr import RenumberOptions, renumber_file

result = renumber_file(
    input_path=Path("input.pdb"),
    chain_id="H",
    output_path=Path("output.cif"),
    options=RenumberOptions.from_values(chain_type="H", overwrite=True),
)

print(result.detected_chain_type.value)
```

## Development

```bash
pytest
JAX_PLATFORMS=cpu pytest -m slow
sphinx-build -W -b html docs docs/_build/html
ruff check .
ruff format --check .
```
