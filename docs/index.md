# SAbR

SAbR assigns antibody residue numbers from backbone coordinates. It accepts
PDB and mmCIF structures, preserves the input object, and supports IMGT,
Chothia, Kabat, Martin, AHo, and Wolfguy numbering. Its encoder is a fine-tuned
version of the ProteinMPNN encoder from
[SoftAlign](https://github.com/jtrinquier/SoftAlign).

## Installation

SAbR requires Python 3.11 or newer. Both `sabr` and `softalign` modes run
on CPU using NumPy and SciPy.

```bash
pip install sabr-kit
```

### Docker

With Docker installed and running, pull the image and run SAbR from the
directory containing `antibody.pdb`:

```bash
docker pull ghcr.io/delalamo/sabr:latest
docker run --rm -v "$PWD:/workspace" ghcr.io/delalamo/sabr:latest \
  -i antibody.pdb -c H -o numbered.pdb
```

The mounted directory makes the input available inside the container and
saves `numbered.pdb` back to the same directory on your computer. The image
runs `sabr` directly; append any of the command-line options below, or use
`docker run --rm ghcr.io/delalamo/sabr:latest --help` for help. To pin a
release, replace `latest` with its version tag (without the leading `v`).

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
     [--no-mmcif]
     [--overwrite] [-v]
```

The defaults are IMGT, automatic chain selection, noise level `0.0`, and
`sabr` mode. SoftAlign mode uses its own fixed references, so `noise_level` is
ignored in that mode. Normal output contains only warnings and errors. Use
`--verbose` to show `Numerical backend: NumPy/SciPy (CPU)`, chain-selection
scores, and a traceback on failure.

To search only the scFv candidate set, pass `--scfv`. This is equivalent to
`--chain-type HK,HL,KH,LH`; as a compatibility flag it requires the default
automatic chain type. Each domain is numbered in the selected scheme, then
successive domains receive offsets of `0`, `1000`, `2000`, and so on. For
IMGT, this gives domain ranges `1–128`, `1001–1128`, `2001–2128`, and so on.
Every linker continues sequentially from the preceding domain's last number.
Multi-domain references use the selected parameter mode, so either form can be
combined with `--mode softalign`. Gap-open and gap-extension penalties are
disabled for query linker residues aligned at every boundary between domain
references. All other internal gap transitions retain the selected parameter
mode's normal penalties.

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

structure = PDBParser(PERMISSIVE=False, QUIET=True).get_structure(
    "antibody", "antibody.pdb"
)
numbered = renumber_structure(
    structure,
    chain="H",
    scheme="imgt",
    chain_type="auto",
)
```

Strict PDB parsing matches the CLI and rejects malformed duplicate atom
records before renumbering.

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

## Long-CDR numbering support

SAbR extends the bundled ANARCI insertion codes in IMGT, Kabat, Chothia,
Martin, and AHo: `A`–`Z`, `AA`–`ZZ`, then three-letter codes, up to 5,000
codes per insertion anchor. These are SAbR extensions of the numbering
implementation, not claims that the original scheme specifications define
every such loop. Existing insertion locations and short-loop gap rules are
preserved. Linear insertions proceed from `Z` to `AA`; IMGT distributes them
between two central anchors, with the right anchor's codes in reverse
sequence order.

Supported lengths in residues for the regions renumbered by each scheme:

| Scheme | Chain | CDR1 | CDR2 | CDR3 |
| --- | --- | ---: | ---: | ---: |
| IMGT | All supported chains | 10,012 | 10,010 | 10,013 |
| Kabat | Heavy | 5,013 | 5,008 | 5,010 |
| Chothia / Martin | Heavy | 5,011 | 5,008 | 5,010 |
| Kabat / Chothia / Martin | Kappa / lambda | 5,011 | 5,004 | 5,009 |
| AHo | All supported chains | 5,018 | 5,020 | 5,032 |
| Wolfguy | Heavy | 49 | 49 | 51 |
| Wolfguy | Kappa / lambda | 49 | 49 | 49 |

Lengths count residues, excluding alignment deletions, in ANARCI's
scheme-specific renumbering regions. They are not interchangeable biological
CDR definitions. All schemes take CDR3 from IMGT alignment positions
105–117 and their insertions. The CDR1/CDR2 input spans are IMGT 27–38/56–65
for IMGT, 25–40/56–75 for AHo, 24–40/55–65 for Kabat heavy,
24–38/55–65 for Chothia and Martin heavy, 24–40/57–67 for their light-chain
implementations (including Kabat), and 27–40/55–74 (heavy) or 24–40/56–69
(light) for Wolfguy. Framework-boundary insertions are smoothed before these
regions are counted. AHo's renumbered FW3 region (IMGT 76–91) also supports
extended codes, up to 5,016 residues.

Wolfguy uses fixed numbered positions for its loops; SAbR does not add
insertion anchors to those loops. An unsupported length raises `ValueError`,
for example: `Wolfguy CDR3 length 52 exceeds the supported limit of 51 residues.`
The same error format applies when an extended-code scheme exhausts its
supported codes.

These limits apply to numbering an existing alignment. Structure selections
remain limited to 1,024 polymer residues before model alignment, and these
numbering extensions do not guarantee that the model can align every long
loop. BioPython and mmCIF can retain multi-character codes; switching output
formats does not bypass numbering or structure-selection limits.

## Structural gaps and modified residues

A C–N distance above 2.66 Å is treated as a structural gap. After encoding
observed coordinates, internal missing residues declared in PDB REMARK 465
or mmCIF metadata receive zero embedding vectors. The expanded sequence
passes through alignment, deterministic CDR/DE corrections, and scheme
conversion before ghost rows are removed. Ghosts contribute zero similarity
to every reference position, and metadata-filled breaks use normal gap
opening costs. Unexplained breaks retain the opening-cost waiver and
correction fallback, with their indices shifted to account for ghost rows.
No atoms are created; terminal unresolved residues are not added.

Ghost residues are the default when usable missing-residue annotations are
available. Without them, SAbR automatically uses detected structural gaps
with zero gap-opening costs. For partially annotated inputs, unexplained
breaks still use this fallback. To ignore all missing-residue annotations
and force geometry-only handling, pass `--auto-detect-gaps` (alias
`--auto_detect_gaps`) on the CLI or `auto_detect_gaps=True` to
`renumber_structure`. This disables both ghosts and metadata-based loop
correction; extension costs remain unchanged.

The CLI reads metadata automatically using Biopython’s `MMCIF2Dict`. For
the Python API, `Bio.PDB.PDBParser` retains PDB missing-residue records.
For mmCIF, use the standard `Bio.PDB.MMCIFParser` with its default author
chain/residue IDs and pass `mmcif_dict=MMCIF2Dict("input.cif")` to
`renumber_structure`. This dictionary takes precedence over header records
and must describe the input structure’s original author IDs. Missing
records are matched by author chain, model, residue number, and insertion
code. Number jumps alone do not establish missing sequence. The returned
structure drops original missing-residue records for the renumbered chain
to avoid reusing obsolete author IDs; the input structure is unchanged.

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
