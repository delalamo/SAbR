# Structure-based Antibody Renumbering

_This repo is currently in development. If you encounter any bugs, please report the issue [here](https://github.com/delalamo/SAbR/issues)._

SAbR (<u>S</U>tructure-based <u>A</u>nti<u>b</u>ody <u>R</u>enumbering) renumbers antibody PDB files using the 3D coordinate of backbone atoms. It uses custom forked versions of [SoftAlign](https://github.com/delalamo/SoftAlign) and [ANARCI](https://github.com/delalamo/ANARCI/tree/master) to align structures to SAbDaB-derived consensus embeddings and renumber to various antibody schemes, respectively.

## Installation and use

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

2. ~~Alternatively, SAbR can be directly run with the latest docker container:~~

❌ _This doesn't currently work. Please check back soon!_ ❌

```bash
docker run --rm ghcr.io/delalamo/sabr:latest -i input.pdb -o output.pdb -c CHAIN_ID
```

## Running SAbR

```bash
usage: sabr [-h] -i INPUT_PDB -c INPUT_CHAIN -o OUTPUT_PDB [-n NUMBERING_SCHEME] [-t] [--overwrite] [-v]

Structure-based Antibody Renumbering (SAbR) renumbers antibody PDB files using the 3D coordinate of backbone atoms.

options:
  -h, --help            show this help message and exit
  -i INPUT_PDB, --input_pdb INPUT_PDB
                        Input pdb file
  -c INPUT_CHAIN, --input_chain INPUT_CHAIN
                        Input chain
  -o OUTPUT_PDB, --output_pdb OUTPUT_PDB
                        Output pdb file
  -n NUMBERING_SCHEME, --numbering_scheme NUMBERING_SCHEME
                        Numbering scheme, default is IMGT. Supports IMGT, Chothia, Kabat, Martin, AHo, and Wolfguy.
  --overwrite           Overwrite PDB
  -v, --verbose         Verbose output
```

## Known issues

- SAbR currently struggles with scFvs for two reasons. First, it is unclear how to assign canonical numbering to multiple domains within a single chain, unless we accept a spacer (e.g., starting chain #2 at 201 instead of 1). Second, it will sometimes align across both chains, introducing a massive insertion in between. It is unclear how to prevent this; please see [issue #2](https://github.com/delalamo/SAbR/issues/2) for details.
- SAbR sometimes mistakenly includes sheets from the Fab in the VH.
- The algorithm for renumbering CDRs, which is the same as the one for IMGT, does not account for unassigned residues. So if a residue is missing due to heterogeneity, the CDR numbering algorithm will misnumber other residues in the CDR.
