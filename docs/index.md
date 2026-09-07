# SAbR

SAbR assigns antibody residue numbers from backbone coordinates. It accepts
PDB and mmCIF structures, preserves the input object, and supports IMGT,
Chothia, Kabat, Martin, AHo, and Wolfguy numbering. Its encoder is a fine-tuned
version of the ProteinMPNN encoder from
[SoftAlign](https://github.com/jtrinquier/SoftAlign).

## Installation

SAbR requires Python 3.11 or newer.

```bash
pip install sabr-kit
```

## Command line

Renumber chain `H` using the default IMGT scheme and automatic heavy,
kappa, or lambda reference selection:

```bash
sabr -i antibody.pdb -c H -o numbered.pdb
```

Select a numbering scheme and chain type explicitly:

```bash
sabr -i antibody.cif -c light_chain -o numbered.cif \
  --scheme chothia --chain-type K
```

Pass comma-separated candidates to restrict automatic selection. Every
candidate is an ordered sequence of `H`, `K`, and `L` domains:

```bash
sabr -i multispecific.cif -c A -o numbered.cif \
  --chain-type HHK,HHL
```

For example, `H,K` tries only heavy and kappa single domains, while `HK,HL`
tries heavy-kappa and heavy-lambda two-domain chains. `HHK,HHL` can represent
a VHH followed by an scFv whose heavy domain precedes its light domain.

Use the complete SoftAlign parameter set (encoder, reference embeddings, and
gap penalties):

```bash
sabr -i antibody.pdb -c H -o numbered.pdb --mode softalign
```

Renumber only residues whose source PDB numbers are 1 through 130:

```bash
sabr -i antibody.pdb -c H -o numbered.pdb \
  --residue-range 1 130
```

Insertion-coded residues are included when their numeric component falls in
the range. Existing files are protected unless `--overwrite` is supplied.

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

The defaults are IMGT, automatic chain selection, noise level `0.0`, and
`sabr` mode. SoftAlign mode uses its own fixed references, so `noise_level` is
ignored in that mode. Normal output contains only warnings and errors. Use
`--verbose` to show the numerical backend, chain-selection scores, and a traceback
on failure.

To search only the scFv candidate set, pass `--scfv`. This is equivalent to
`--chain-type HK,HL,KH,LH`; as a compatibility flag it requires the default
automatic chain type. SAbR places successive domains in separate 1000-number
residue blocks: `1–128`, `1001–1128`, `2001–2128`, and so on. Every linker
continues sequentially from the preceding domain's last number. Multi-domain
references use the selected parameter mode, so either form can be combined
with `--mode softalign`. Gap-open and gap-extension penalties are disabled for
query linker residues aligned at every boundary between domain references.
All other internal gap transitions retain the selected parameter mode's
normal penalties.

When candidates are compared, SAbR applies the normal affine gap-open and
gap-extension costs to unaligned query and reference termini of multi-domain
representations. This post-hoc selection penalty does not change the computed
alignments or their raw scores.

## Python API

`renumber_structure` accepts a Biopython `Structure`, returns a new Biopython
`Structure`, and does not mutate its input.

```python
from Bio.PDB import PDBParser
from sabr import renumber_structure

structure = PDBParser(QUIET=True).get_structure("antibody", "antibody.pdb")
numbered = renumber_structure(
    structure,
    chain="H",
    scheme="imgt",
    chain_type="auto",
)
```

The function signature is:

```python
renumber_structure(
    structure,
    chain: str,
    scheme: str = "imgt",
    chain_type: str = "auto",
    noise_level: float = 0.0,
    residue_range: tuple[int, int] | None = None,
    mode: str = "sabr",
    scfv: bool = False,
    dangerously_allow_structural_gaps: bool = False,
)
```

`mode="softalign"` selects the SoftAlign encoder weights, reference
embeddings, and exact gap penalties stored in `softalign_gap.npz`. The default
`mode="sabr"` preserves existing behavior.

Non-target chains, waters, ligands, metadata represented by the input object,
and residues outside the selected range are preserved in the returned clone.
Only single-model structures are supported.

## PDB and mmCIF output

Use mmCIF when a structure has:

- a chain ID longer than one character;
- multi-character insertion codes;
- residue numbers outside the PDB range `-999` through `9999`; or
- more than 99,999 atoms.

For exceptionally long CDR insertions, use mmCIF output.

CLI conversion preserves atomic structure content but does not preserve every
non-atomic mmCIF category.

## Structural gaps and modified residues

A C–N distance above 2.66 Å is treated as a structural gap. SAbR refuses to
run if the selected chain contains one. CLI users can explicitly override the
safety check with `--dangerously-allow-structural-gaps`, which prints a warning
before any other runtime output. Python API users can set
`dangerously_allow_structural_gaps=True`. With the override enabled, a gap
crossing a CDR or the DE loop between IMGT anchors 79 and 85 retains the
learned alignment for that region while corrections elsewhere continue.
Even with the override, performance degrades when structural gaps are
included. Without a gap crossing that region, DE-loop residues fill 80 first,
then 84 back through 81; additional residues are inserted after 82.

Supported modified peptide residues are translated to their canonical parent
only for sequence generation. Original residue names and atoms are preserved.
Unknown or ambiguous polymer chemistry is rejected rather than converted to
`X`.

## Experimental TCR use

T-cell receptors are not an officially supported SAbR target. For
experimental use, pass the actual TCR chain type (`A`, `B`, `G`, or `D`) as
`--chain-type` or `chain_type`. SAbR aligns only against the K reference,
which includes IMGT position 10 as TCRs do, while retaining the TCR type for
ANARCI conversion. This workflow supports IMGT and AHo numbering only;
Chothia, Kabat, Martin, and Wolfguy are antibody-specific.

## Errors

Common errors include:

- selecting a missing chain;
- selecting more than one structural model;
- residues missing N, CA, or C atoms;
- a range containing multiple or discontinuous domains;
- output that cannot be represented in PDB format; and
- selections above the 1,024-residue safety limit.

For a long chain containing domains outside the desired candidate, pass
`--residue-range` on the command line or `residue_range=(start, end)` in
Python.
