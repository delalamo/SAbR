# CLI Module

::: sabr.cli
    options:
      show_root_heading: true
      show_source: false
      members:
        - main

## Command Reference

```
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
                                  for mmCIF format.  [required]
  -n, --numbering-scheme [imgt|chothia|kabat|martin|aho|wolfguy]
                                  Numbering scheme.  [default: IMGT]
  --overwrite                     Overwrite the output file if it exists.
  -v, --verbose                   Enable verbose logging.
  --max-residues INTEGER          Maximum number of residues to process.
  --extended-insertions           Enable extended insertion codes for very
                                  long CDR loops (requires mmCIF output).
  --disable-deterministic-renumbering
                                  Disable deterministic renumbering corrections.
  -t, --chain-type [H|K|L|heavy|kappa|lambda|auto]
                                  Chain type for ANARCI numbering.
  -h, --help                      Show this message and exit.
```
