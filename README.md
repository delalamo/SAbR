# Structure-based Antibody Renumbering

[![PyPI](https://img.shields.io/pypi/v/sabr-kit)](https://pypi.org/project/sabr-kit/)
[![Python ≥3.11](https://img.shields.io/badge/python-%E2%89%A53.11-blue?logo=python&logoColor=white)](https://pypi.org/project/sabr-kit/)
[![Python checks](https://github.com/delalamo/SAbR/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/delalamo/SAbR/actions/workflows/test.yml)
[![Documentation](https://readthedocs.org/projects/sabr/badge/?version=latest)](https://sabr.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Docker build](https://github.com/delalamo/SAbR/actions/workflows/docker-build.yml/badge.svg)](https://github.com/delalamo/SAbR/actions/workflows/docker-build.yml)
[![PyPI downloads](https://img.shields.io/pypi/dm/sabr-kit)](https://pypi.org/project/sabr-kit/)

SAbR (Structure-based Antibody Renumbering) assigns antibody residue numbers
from backbone coordinates. It combines a fine-tuned version of the ProteinMPNN
encoder from [SoftAlign](https://github.com/jtrinquier/SoftAlign) with the
original affine Smith–Waterman alignment and ANARCI numbering rules.

The complete usage guide is available in the
[SAbR documentation](https://sabr.readthedocs.io/).

## Installation

SAbR requires Python 3.11 or newer. Both `sabr` and `softalign` modes run
on CPU using NumPy and SciPy

```bash
pip install sabr-kit
```

## Command line

```bash
sabr -i antibody.pdb -c H -o numbered.pdb
```

The complete interface is:

```text
sabr -i INPUT -c CHAIN -o OUTPUT
     [-n imgt|chothia|kabat|martin|aho|wolfguy]
     [-t auto|TYPES]
     [--noise-level 0.0|0.2|0.5|1.0|2.0]
     [-m sabr|softalign]
     [--residue-range START END]
     [--scfv]
     [--dangerously-allow-structural-gaps]
     [--no-mmcif]
     [--overwrite] [-v]
```

Defaults are IMGT numbering, automatic H/K/L selection, noise level `0.0`,
`sabr` mode, and the entire selected chain. `--chain-type` accepts a
comma-separated set of candidates. Each candidate is a sequence of `H`, `K`,
and `L` domains: `H,K` tries heavy and kappa single domains, `HK,HL` tries
heavy-kappa and heavy-lambda two-domain chains, and `HHK,HHL` tries the
corresponding three-domain chains. Existing outputs are never replaced unless
`--overwrite` is given. Normal output contains only warnings and errors; `-v`
reports reference scores and pipeline decisions.

Use `--mode softalign` to select the original SoftAlign encoder weights,
reference embeddings, and affine gap penalties together. SoftAlign references
do not vary with `--noise-level`, so that option is ignored in this mode.

Use `--scfv` to search only the scFv candidate set. It is equivalent to
`--chain-type HK,HL,KH,LH` and still requires the default automatic chain type
when used as a flag. Multi-domain results place successive domains in separate
1000-number residue blocks: `1–128`, `1001–1128`, `2001–2128`, and so on.
Linker residues continue sequentially from the preceding domain's last number.
Multi-domain references use the selected parameter mode, so either form can be
combined with `--mode softalign`.

Input and output may be PDB (`.pdb`) or mmCIF (`.cif` or `.mmcif`). When a
requested PDB output needs multi-character insertion codes, SAbR warns and
automatically writes it to the corresponding `.cif` path instead. Pass
`--no-mmcif` to forbid this conversion and fail. Other values that exceed PDB
field limits still require an explicitly named mmCIF output. Writes are atomic,
so a failed run does not leave a partial output.

CLI conversion guarantees preservation of atomic structure content, not
arbitrary non-atomic mmCIF categories. It warns for every mmCIF input.

## Python API

```python
from Bio.PDB import PDBParser
from sabr import renumber_structure

structure = PDBParser(QUIET=True).get_structure("antibody", "antibody.pdb")
numbered = renumber_structure(structure, chain="H")
```

`renumber_structure` accepts a Biopython `Structure`, never mutates its input,
and returns a new Biopython `Structure`. Non-target chains, hetero residues,
waters, and residues outside an inclusive `residue_range` are preserved. SAbR
rejects multi-model structures rather than silently modifying only one model.
If a partial range would create duplicate residue IDs with unchanged residues,
the operation fails with an explanation.

For unusually long loops that need extended insertion codes, use mmCIF output.

## Scientific behavior

- The default `sabr` mode preserves the trained SAbR encoder weights,
  references, and gap penalties unchanged.
- The optional `softalign` mode uses `softalign_encoder.npz`,
  `softalign_embeddings.npz`, and the exact penalties in
  `softalign_gap.npz` as one parameter set.
- Alignment uses the original affine Smith–Waterman recurrence, with soft
  assignments computed by an explicit NumPy reverse pass.
- Deterministic CDR gap distribution and DE-loop correction are always
  applied. Between IMGT anchors 79 and 85, DE-loop residues fill 80 first,
  then 84 back through 81; additional residues are inserted after 82.
- Automatic chain selection aligns against H, K, and L references and uses
  the highest score, with deterministic H/K/L tie order.
- `chain_type` candidate order is deterministic and resolves score ties.
- scFv mode searches only the `HK,HL,KH,LH` candidate list in that order.
- Multi-domain candidates do not apply gap-open or gap-extension costs to
  query linker residues aligned at any boundary between domain references.
- Multi-domain candidates receive normal affine gap-open and gap-extension
  costs for unaligned query and reference termini when their selection scores
  are compared; the underlying alignments and raw alignment scores are
  unchanged.

A structural gap is detected when the C–N distance between consecutive
residues exceeds 2.66 Å. SAbR refuses to run when a structural gap is
detected. To override this safety check, pass
`--dangerously-allow-structural-gaps`; a warning is printed before any other
runtime output. Python API callers can set
`dangerously_allow_structural_gaps=True`. When overridden, a gap skips only
the affected CDR or DE-loop correction and other regions continue normally,
but performance degrades when structural gaps are included in the input.

T-cell receptors are not an officially supported SAbR target. For
experimental use, pass the actual TCR chain type (`A`, `B`, `G`, or `D`) with
`--chain-type` or `chain_type`. SAbR aligns only against the K reference,
which includes IMGT position 10 as TCRs do, while retaining the TCR type for
ANARCI conversion. This workflow is limited to IMGT and AHo numbering; the
other bundled schemes are antibody-specific.

## Development

```bash
pip install -c constraints.txt -e '.[test]'
pytest
pre-commit run --all-files
```

`constraints.txt` records the exact canonical development and CI environment.
Package metadata remains ranged for normal installation.

The committed tests are self-contained and never download data. They verify
the fixed asset hashes, encoder and alignment baselines, all numbering schemes,
H/K/L selection, regional corrections, structure-object behavior, and CLI
failure handling.

## License and attribution

SAbR is distributed under the repository license. The vendored ANARCI
numbering code retains its original license in `src/sabr/_anarci/LICENSE`.
