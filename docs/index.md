# SAbR Documentation

**Structure-based Antibody Renumbering**

[![PyPI version](https://img.shields.io/pypi/v/sabr-kit.svg)](https://pypi.org/project/sabr-kit/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

SAbR (**S**tructure-based **A**nti**b**ody **R**enumbering) renumbers antibody PDB files using the 3D coordinates of backbone atoms. It uses neural network embeddings from [SoftAlign](https://github.com/delalamo/SoftAlign) and antibody numbering schemes from [ANARCI](https://github.com/delalamo/ANARCI) to align structures to consensus embeddings and apply standard numbering schemes.

!!! note
    This project is currently in development. If you encounter any bugs, please report them at [GitHub Issues](https://github.com/delalamo/SAbR/issues).

## Key Features

- **Structure-based alignment**: Uses 3D backbone coordinates instead of sequence
- **Multiple numbering schemes**: IMGT, Chothia, Kabat, Martin, Aho, Wolfguy
- **Automatic chain detection**: Distinguishes heavy, kappa, and lambda chains
- **PDB and mmCIF support**: Input and output in both formats
- **Extended insertions**: Support for very long CDR loops (mmCIF only)

## Quick Start

**Installation:**

```bash
pip install sabr-kit
```

**Basic usage:**

```bash
sabr -i input.pdb -c A -o output.pdb -n imgt
```

## How It Works

SAbR uses a 6-step pipeline:

1. **Load structure**: Parse PDB or mmCIF file and extract the target chain
2. **Generate embeddings**: Compute 64-dimensional MPNN embeddings for each residue
3. **Align to reference**: Use SoftAlign to align against unified consensus embeddings
4. **Convert to states**: Transform alignment matrix to HMMER-style state vector
5. **Apply numbering**: Use ANARCI to apply the selected numbering scheme
6. **Write output**: Save the renumbered structure

## Navigation

| Section | Description |
|---------|-------------|
| [Installation](installation.md) | Install SAbR via pip, from source, or using Docker |
| [Quick Start](quickstart.md) | Get up and running with SAbR in minutes |
| [Usage Guide](usage.md) | Detailed usage instructions and examples |
| [Numbering Schemes](numbering_schemes.md) | Learn about IMGT, Chothia, Kabat, etc. |
| [API Reference](api/cli.md) | Complete API documentation for developers |
| [Contributing](contributing.md) | How to contribute to SAbR |
