# Mismatch Cases for Debugging

These PDB structures produce "Residue mismatch" errors during alignment,
where the alignment assigns a residue to an IMGT position that expects
a different amino acid.

## Cases

| File       | Chain | Error                      |
| ---------- | ----- | -------------------------- |
| 7fbk_C.pdb | C     | Residue mismatch: S vs ASN |
| 7q1u_B.pdb | B     | Residue mismatch: F vs LYS |
| 7qbe_F.pdb | F     | Residue mismatch: S vs GLY |
| 7s83_A.pdb | A     | Residue mismatch: R vs ALA |

## Reproduction

These errors occur when running the SAbDab evaluation pipeline with
the soft CDR boundary detection feature enabled.

```python
from sabr import mpnn_embeddings, softaligner, constants

inp = mpnn_embeddings.from_pdb("7fbk_C.pdb", "C")
aligner = softaligner.SoftAligner()
out = aligner(inp, chain_type=constants.ChainType.HEAVY,
              deterministic_loop_renumbering=True)
```

The mismatch happens when the alignment's anchor points are incorrect,
causing the sequence-to-IMGT mapping to be off by one or more positions.
