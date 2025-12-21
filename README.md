# Structure-based Antibody Renumbering

_This repo is currently in development. If you encounter any bugs, please report the issue [here](https://github.com/delalamo/SAbR/issues)._

SAbR (<u>S</U>tructure-based <u>A</u>nti<u>b</u>ody <u>R</u>enumbering) renumbers antibody PDB files using the 3D coordinate of backbone atoms. It uses custom forked versions of [SoftAlign](https://github.com/delalamo/SoftAlign) and [ANARCI](https://github.com/delalamo/ANARCI/tree/master) to align structures to SAbDaB-derived consensus embeddings and renumber to various antibody schemes, respectively.

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

  Structure-based Antibody Renumbering (SAbR) renumbers antibody PDB files
  using the 3D coordinates of backbone atoms.

Options:
  -i, --input-pdb FILE            Input PDB file.  [required]
  -c, --input-chain TEXT          Chain identifier to renumber.
  -o, --output FILE               Destination structure file. Use .pdb
                                  extension for PDB format or .cif extension
                                  for mmCIF format. mmCIF is required when
                                  using --extended-insertions.  [required]
  -n, --numbering-scheme [imgt|chothia|kabat|martin|aho|wolfguy]
                                  Numbering scheme.  [default: (IMGT)]
  --overwrite                     Overwrite the output PDB if it already
                                  exists.
  -v, --verbose                   Enable verbose logging.
  --max-residues INTEGER          Maximum number of residues to process from
                                  the chain. If 0 (default), process all
                                  residues.
  -t, --chain-type [heavy|light|auto]
                                  Restrict alignment to specific chain type
                                  embeddings. 'heavy' searches only heavy
                                  chain (H) embeddings, 'light' searches only
                                  light chain (K and L) embeddings, 'auto'
                                  searches all embeddings and picks the best
                                  match.  [default: auto]
  --extended-insertions           Enable extended insertion codes (AA, AB,
                                  ..., ZZ, AAA, etc.) for antibodies with very
                                  long CDR loops. Requires mmCIF output format
                                  (.cif extension). Standard PDB format only
                                  supports single-character insertion codes
                                  (A-Z, max 26 insertions per position)
  -v, --verbose         Verbose output
```

## Known issues

- SAbR currently struggles with scFvs for two reasons. First, it is unclear how to assign canonical numbering to multiple domains within a single chain, unless we accept a spacer (e.g., starting chain #2 at 201 instead of 1). Second, it will sometimes align across both chains, introducing a massive insertion in between. It is unclear how to prevent this; please see [issue #2](https://github.com/delalamo/SAbR/issues/2) for details.
- SAbR sometimes mistakenly includes sheets from the Fab in the VH.
- The algorithm for renumbering CDRs, which is the same as the one for IMGT, does not account for unassigned residues. So if a residue is missing due to heterogeneity, the CDR numbering algorithm will misnumber other residues in the CDR.
