# Structure-based Antibody Renumbering

SAbR (<u>S</U>tructure-based <u>A</u>nti<u>b</u>ody <u>R</u>enumbering) renumbers antibody PDB files using the 3D coordinate of backbone atoms. It uses custom forked versions of [SoftAlign](https://github.com/delalamo/SoftAlign) and [ANARCI](https://github.com/delalamo/ANARCI/tree/master) to align structures to SAbDaB-derived consensus embeddings and renumber to various antibody schemes, respectively.

To install from PyPI:

```bash
pip install sabr-kit
```

Or to install the latest version:

```bash
git clone --recursive https://github.com/delalamo/SAbR.git
cd SAbR/
pip install -e .
```

Running SAbR:

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
