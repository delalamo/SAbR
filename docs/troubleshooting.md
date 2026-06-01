# Troubleshooting

## Missing Chain

Check the chain identifier in the input structure. SAbR reports the available
chains when the requested chain is absent.

## Missing Backbone Atoms

Embedding extraction uses residues with complete `N`, `CA`, and `C` atoms.
Residues missing those atoms are skipped for embedding generation.

## JAX Install Errors

Use a Python environment whose architecture matches the installed `jaxlib`
wheel. On Apple Silicon, prefer an ARM Python environment. For CPU-only runs,
set:

```bash
JAX_PLATFORMS=cpu
```

## Extended Insertion Codes

Use `.cif` output if SAbR reports extended insertion codes. PDB output cannot
store multi-character insertion codes.

## Unexpected Chain-Type Selection

`--chain-type auto` aligns against the labelled `H`, `K`, and `L` reference
embeddings and uses the best-scoring label for ANARCI. Pass `--chain-type H`,
`--chain-type K`, or `--chain-type L` when the chain type is known and SAbR
should align only against that labelled embedding.
