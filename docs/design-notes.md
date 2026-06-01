# Design Notes

The SAbR pipeline is deliberately direct:

```text
structure file or BioPython structure
  -> select chain and extract [N, CA, C, computed CB] coordinates
  -> compute query embeddings
  -> align query embeddings to labelled H/K/L reference embeddings
  -> convert alignment matrix to HMM/ANARCI states
  -> apply numbering
  -> thread numbering back onto a structure
  -> write output or return an in-memory structure
```

Deterministic corrections adjust known antibody regions after neural alignment.
They are applied by default and can be disabled for raw alignment behavior.

Chain type is selected from the embedding label. In `auto` mode SAbR aligns
against all labelled references and passes the best-scoring label to ANARCI. In
explicit `H`, `K`, or `L` mode it aligns only against that labelled reference
and passes the same label to ANARCI.

Custom gap penalties are CDR-only: SAbR sets gap-open penalties to zero for
IMGT CDR positions and leaves gap-extension penalties normal everywhere.

The fourth coordinate channel is the historical computed CB channel. The model
math is not changed by the redesign.
