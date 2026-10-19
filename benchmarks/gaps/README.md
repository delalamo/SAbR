# Issue #226: gap-aware alignment experiment

Date: 2026-10-19
Base: `86c945d06bb67a06da39b55b535966ff3ae8de19`
Branch: `codex/issue-226-gap-aware-alignment`

## Finding

Waiving gap opening at structural breaks helps a small subset of these
examples, but does **not** solve incomplete-structure numbering. Across
147 controlled deletion cases, three improve, 144 are unchanged, and none
worsen. Final IMGT errors decrease from 109 to 104 (4.6% fewer errors).
Completely correct cases increase from 93/147 (63.3%) to 96/147 (65.3%).

This is a small fixture-based pilot, not a replication of the manuscript
benchmark, and the deletion variants are correlated within their parent
structures. All remaining errors are in CDRs.

## Implementation

The existing `structure._detect_gaps` checks each consecutive observed
C–N distance against 2.66 angstroms. It returns the zero-based query row
immediately BEFORE a break. The detector uses coordinates, not residue
number jumps. All 147 artificial breaks were detected in this experiment.

The change threads these indices through candidate scoring AND subsequent
gradient/trace computation. It zeros only the match-to-right opening cost
at the corresponding query row: a right transition skips reference
positions to represent residues missing from the query. Extension costs,
query insertions, and the existing scFv reference-boundary behavior remain
unchanged. Tests distinguish the correct row from an adjacent row and
distinguish deletions from insertions.

The public structural-gap rejection policy remains in place. To exercise
this experimental behavior through the API, pass
`dangerously_allow_structural_gaps=True`; through the CLI, pass
`--dangerously-allow-structural-gaps`. This is appropriate given the
remaining CDR errors.

## Protocol

- SAbR mode, noise 0.0, IMGT numbering, automatic chain-type selection.
- Existing pinned PDB fixtures and independent SAbDab labels from
  `benchmarks/antibody/manifest.json` (including source URLs and checksums).
- Five intact parent chains: 12E8 H/L, 1MEL A, 1LMK A, 1MOE A.
  Their seven variable domains supply the deletion cohort.
- For each domain, remove 1, 3, or 5 observed residues beginning at the first
  labeled position at or above IMGT 10, 29, 45, 58, 81, 91, or 108
  (FR1, CDR1, FR2, CDR2, DE loop, FR3, CDR3).
- Crop individual domains before deletion, recompute coordinate embeddings,
  and compare final residue IDs, including insertion codes, to the ORIGINAL
  labels of surviving residues. Do not generate new ground truth by
  renumbering the shortened sequence.
- Paired baseline disables only the new alignment keyword. Both arms retain
  the existing structural-gap-aware correction skipping. Both arms share
  each newly computed embedding.
- Run extraction, encoding, alignment, numbering, and structure application.
  A separate public API regression test verifies the 1MEL DE-loop rescue.
- Natural-gap controls: full two-domain 6NOU A and 5KVE L, with their existing
  unresolved linker gaps. These do not establish efficacy for naturally
  unresolved CDRs.

## Results

| Cohort | Cases | Labeled residues | Baseline errors | Gap-aware errors | Exact cases, before -> after |
|---|---:|---:|---:|---:|---:|
| Intact controls | 5 | 824 | 0 | 0 | 5 -> 5 |
| Natural linker gaps | 2 | 458 | 0 | 0 | 2 -> 2 |
| Controlled intradomain deletions | 147 | 16,863 | 109 | 104 | 93 -> 96 |

Neither arm failed to produce numbering in any case.

| Deleted region | Cases | Baseline errors | Gap-aware errors |
|---|---:|---:|---:|
| FR1 | 21 | 0 | 0 |
| CDR1 | 21 | 26 | 26 |
| FR2 | 21 | 1 | 0 |
| CDR2 | 21 | 21 | 21 |
| DE loop | 21 | 4 | 0 |
| FR3 | 21 | 0 | 0 |
| CDR3 | 21 | 57 | 57 |

The three improved cases:

| Case | Observed residues flanking the break (source IDs) | Errors before -> after |
|---|---|---:|
| 1MEL A, delete 1 residue starting at IMGT 81 | 72 / 74 | 2 -> 0 |
| 1MEL A, delete 3 residues starting at IMGT 81 | 72 / 76 | 2 -> 0 |
| 1MOE A, second domain, delete 1 residue starting at IMGT 45 | 158 / 160 | 1 -> 0 |

## Why CDR numbering still needs work

Code inspection reveals a downstream obstacle independent of the opening
penalty. The vendored `number_imgt` routine excludes CDR deletion states,
constructs each observed CDR sequence, and calls `get_imgt_cdr` using that
sequence's length. This can erase structurally inferred deletion placement,
even though SAbR's earlier deterministic correction was skipped for the
break. Merely fixing the alignment penalty cannot fix this behavior.

For example, after deleting one residue near IMGT 29 in 12E8 H, both arms
give surviving residues expected at 30 and 35 the final IDs 29 and 30.
The same pattern occurs in CDR2. This supports investigating downstream
length-based reconstruction, but does not prove that every remaining error
has that single cause. Long-CDR insertions, reduced reference positions, and
the structural embeddings may also contribute. The attempted additional
per-residue alignment diagnostic run could not be inspected after the local
execution connection disconnected; no numerical claim about alignment-vs-
numbering attribution is made here.

A useful next experiment is to preserve missing-position information through
IMGT CDR numbering, then repeat this exact benchmark and expand to naturally
unresolved variable-domain loops and lambda chains. Avoid claiming that the
gap-aware change alone makes gapped structures reliable.

## Validation and reproduction

- Full local suite: **1,019 passed, 4 expected failures** (existing
  non-antibody rejection limitations), one warning.
- A subsequently added public API regression test passed; the focused new
  test file reports **12 passed**.
- Tests cover opening-versus-extension costs, exact boundary indexing,
  unchanged insertion costs, multiple breaks, score-only equivalence,
  finite-difference verification of gradients, and a real VHH deletion.
- Source and test artifacts were preserved through the GitHub connector
  after the local execution transport disconnected. Remote copies have
  presentation-only formatting changes; no post-upload rerun was possible.

Run from the repository root:

```sh
PYTHONPATH=src:. python -m benchmarks.gaps.run
PYTHONPATH=src:. python -m pytest tests/test_query_gaps.py -q
```

The benchmark writes `benchmarks/gaps/results.json`, including per-case
errors, selected type, detected break locations, and mismatch details.
The completed first local run produced the aggregate results above.
Its raw JSON remains in the local experiment directory; it was not uploaded
because the local execution transport disconnected before retrieval.

## Manuscript

Diego del Alamo, *Structure-based Antibody Renumbering*, bioRxiv,
posted September 20, 2026:
https://www.biorxiv.org/content/10.64898/2026.09.16.751788v1

DOI: 10.64898/2026.09.16.751788.
The title, abstract, DOI, and posting date were verified. The full-text/PDF
endpoints returned access errors, so the manuscript's exact gap-related
statistics were not independently verified in this session.
