# Structure-based Antibody Renumbering

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

2. Alternatively, SAbR can be directly run with the latest docker container:

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
  -t, --trim            Remove regions outside aligned V-region
  --overwrite           Overwrite PDB
  -v, --verbose         Verbose output
```
