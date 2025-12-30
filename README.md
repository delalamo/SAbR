# Structure-based Antibody Renumbering

[![Tests](https://github.com/delalamo/SAbR/actions/workflows/test.yml/badge.svg)](https://github.com/delalamo/SAbR/actions/workflows/test.yml)
[![Code Formatting](https://github.com/delalamo/SAbR/actions/workflows/format.yml/badge.svg)](https://github.com/delalamo/SAbR/actions/workflows/format.yml)
[![Documentation](https://readthedocs.org/projects/sabr/badge/?version=latest)](https://sabr.readthedocs.io/en/latest/)
[![PyPI version](https://img.shields.io/pypi/v/sabr-kit.svg)](https://pypi.org/project/sabr-kit/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

SAbR (<u>S</U>tructure-based <u>A</u>nti<u>b</u>ody <u>R</u>enumbering) renumbers antibody PDB files using the 3D coordinate of backbone atoms. It uses custom forked versions of [SoftAlign](https://github.com/delalamo/SoftAlign) and [ANARCI](https://github.com/delalamo/ANARCI/tree/master) to align structures to SAbDaB-derived consensus embeddings and renumber to various antibody schemes, respectively.

## Documentation

Full API documentation is available at [sabr.readthedocs.io](https://sabr.readthedocs.io/).

## Installation and use

**Requirements:** Python 3.11 or higher

1. SAbR can be installed into a virtual environment via pip:

```bash
# Latest release
pip install sabr-kit

# Most recent version from Github
git clone --recursive https://github.com/delalamo/SAbR.git
cd SAbR/
pip install -e .
```

It can then be run using the `sabr` command (see below).

2. Alternatively, SAbR can be directly run with the latest docker container:

```bash
docker run --rm ghcr.io/delalamo/sabr:latest -i input.pdb -o output.pdb -c CHAIN_ID
```

## Running SAbR

Practical considerations:

- Heavy and light chain structures are similar enough that chain type should be manually declared with `--chain-type` if possible (leave blank if uncertain).
- It is recommended for now to truncate the query structure to contain only the Fv when running SAbR, as it will sometimes align variable region beta-strands to those in the constant region.
- When running scFvs, it is recommended to run each variable domain independently.

If running on a Mac with apple silicon, set the environmental variable `JAX_PLATFORMS` to `cpu`.

```bash
Usage: sabr [OPTIONS]

  Structure-based Antibody Renumbering (SAbR) renumbers antibody structure
  files using the 3D coordinates of backbone atoms. Supports both PDB and
  mmCIF input formats.

Options:
  -i, --input-pdb FILE            Input structure file (PDB or mmCIF format).
                                  [required]
  -c, --input-chain TEXT          Chain identifier to renumber (single
                                  character).  [required]
  -o, --output FILE               Destination structure file. Use .pdb
                                  extension for PDB format or .cif extension
                                  for mmCIF format. mmCIF is required when
                                  using --extended-insertions.  [required]
  -n, --numbering-scheme [imgt|chothia|kabat|martin|aho|wolfguy]
                                  Numbering scheme.  [default: IMGT]
  --overwrite                     Overwrite the output file if it already
                                  exists.
  -v, --verbose                   Enable verbose logging.
  --residue-range START END       Range of residues to process in PDB
                                  numbering (inclusive). Use '0 0' (default)
                                  to process all residues. Example:
                                  --residue-range 1 120 processes residues
                                  1-120.
  --extended-insertions           Enable extended insertion codes (AA, AB,
                                  ..., ZZ, AAA, etc.) for antibodies with very
                                  long CDR loops. Requires mmCIF output format
                                  (.cif extension). Standard PDB format only
                                  supports single-character insertion codes
                                  (A-Z, max 26 insertions per position).
  --disable-deterministic-renumbering
                                  Disable deterministic renumbering corrections
                                  for loop regions. By default, corrections are
                                  applied for FR1, DE loop, and CDR loops.
  -t, --chain-type [H|K|L|heavy|kappa|lambda|auto]
                                  Chain type for ANARCI numbering.
                                  H/heavy=heavy chain, K/kappa=kappa light,
                                  L/lambda=lambda light. Use 'auto' (default)
                                  to detect from DE loop occupancy.
                                  [default: auto]
  -h, --help                      Show this message and exit.
```

## Python API

SAbR can also be used programmatically to renumber BioPython Structure objects directly in memory:

```python
from Bio.PDB import PDBParser, PDBIO
from sabr import renumber

# Load a structure
parser = PDBParser(QUIET=True)
structure = parser.get_structure("antibody", "input.pdb")

# Renumber the structure (returns a new BioPython Structure)
renumbered = renumber.renumber_structure(
    structure,
    chain="H",                      # Chain identifier
    numbering_scheme="imgt",        # imgt, chothia, kabat, martin, aho, wolfguy
    chain_type="auto",              # H, K, L, or auto
)

# Optionally specify a residue range
renumbered = renumber.renumber_structure(
    structure,
    chain="H",
    res_start=1,                    # Start at residue 1
    res_end=128,                    # End at residue 128
)

# Save the renumbered structure
io = PDBIO()
io.set_structure(renumbered)
io.save("output.pdb")
```

## Known issues

- SAbR currently struggles with scFvs for two reasons. First, it is unclear how to assign canonical numbering to multiple domains within a single chain, unless we accept a spacer (e.g., starting chain #2 at 201 instead of 1). Second, it will sometimes align across both chains, introducing a massive insertion in between. It is unclear how to prevent this; please see [issue #2](https://github.com/delalamo/SAbR/issues/2) for details.
- SAbR sometimes mistakenly includes sheets from the Fab in the VH.
- The algorithm for renumbering CDRs, which is the same as the one for IMGT, does not account for unassigned residues. So if a residue is missing due to heterogeneity, the CDR numbering algorithm will misnumber other residues in the CDR.
