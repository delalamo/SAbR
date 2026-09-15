# Curated structural benchmark

This small, offline benchmark measures agreement with independently sourced
IMGT annotations through SAbR's complete pipeline. It supplements the existing
synthetic and captured-output tests. It is a curated diagnostic set, not an
estimate of accuracy across the PDB or a verified held-out evaluation.

## Run

From an installed source checkout:

```bash
python -m benchmarks.antibody.run --output benchmark-sabr.json
python -m benchmarks.antibody.run --mode softalign --output benchmark-softalign.json
pytest tests/test_benchmark.py
```

No downloads, external annotation tools, mocks, or model stubs run during the
benchmark or tests. Both modes use IMGT, noise level 0.0, and automatic reference
selection; multidomain cases use `scfv=True`. CI runs these tests as part of the
normal suite and uploads both JSON reports as the `antibody-benchmark` artifact.
The benchmark is source-checkout tooling and is excluded from the wheel.

## Cases and provenance

| PDB and chain | Purpose | Input and independent annotation |
| --- | --- | --- |
| [12E8 H/L](https://www.rcsb.org/structure/12E8) | Conventional VH and kappa VL | Full deposited Fab chains, including constant domains; variable domains independently labeled |
| [1MEL A](https://www.rcsb.org/structure/1MEL) | Camelid VHH; unusual loop | 26-residue IMGT CDR3; unresolved N-terminal residue remains absent |
| [6NOU A](https://www.rcsb.org/structure/6NOU) | Real VH–VL scFv | Ixekizumab-derived scFv, including the unresolved linker gap |
| [5KVE L](https://www.rcsb.org/structure/5KVE) | Real VL–VH scFv | ZV-48 scFv, including the unresolved linker gap |
| [1LMK A](https://www.rcsb.org/structure/1LMK) | VH–VL diabody chain | Both domains and the five-residue linker on one deposited chain |
| [1MOE A](https://www.rcsb.org/structure/1MOE) | VL–VH diabody chain | Both domains and the eight-residue linker on one deposited chain |
| 12E8 H, rows 10–119 | N-terminal truncation | Controlled deletion of ten residues from the deposited VH |
| 12E8 H, rows 0–109 | C-terminal truncation | Controlled deletion of ten residues from the deposited VH |
| [1UBQ A](https://www.rcsb.org/structure/1UBQ) | Negative control | Ubiquitin; complete backbone, no antibody domain expected |
| [1LYZ A](https://www.rcsb.org/structure/1LYZ) | Negative control | Lysozyme; complete backbone, no antibody domain expected |

Rows are zero-based and inclusive in this table. The two truncations are
explicitly derived stress tests, not independently deposited truncated proteins.
The diabodies are higher-order assemblies; each benchmark invocation evaluates
one polypeptide, not assembly recognition or a three-plus-domain tandem chain.
The conventional chains and their truncations are related observations. Category
summaries expose this dependence instead of treating the aggregate as a population
accuracy estimate. Lambda chains remain covered by the existing fixture tests;
this initial independent cohort has kappa VLs.

The two scFvs are each run twice: with
`dangerously_allow_structural_gaps=True` for numbering agreement, and under the
default gap rejection policy. The latter runs are policy controls and are
excluded from biological agreement and positive rejection denominators.
No linker coordinates are invented and no domains are concatenated in software.

### Frozen reference labels

[manifest.json](manifest.json) records the PDB, author chain, retrieval date,
source URLs and SHA-256 hashes, SAbDab segment identifiers, and options for each
case. Coordinate fixtures retain only the selected chain's deposited N/CA/C
atoms, including alternate conformers, author numbering and insertion codes.
Side chains, waters, ligands and partner chains are omitted to keep the fixtures
small; the model consumes this backbone representation.

The CSV labels are extracted from the independent IMGT numbering returned by
[SAbDab](https://sabdab.opig.stats.ox.ac.uk/). Each row records the fixture row,
ordered domain, chain type, IMGT position, insertion code, amino acid, original
residue ID, and deposited `label_seq_id`. Mapping uses mmCIF `label_seq_id` and
author residue IDs, **not sequence alignment**. This avoids ambiguity when, for
example, only one of two consecutive terminal serines has coordinates.
Only residues with deposited coordinates are scored; unresolved positions are
not invented. Labels for truncations are inherited from the intact parent.

These are external computational annotations, not experimentally established
numbering truth. They share the IMGT convention with SAbR, but were not generated
by SAbR's encoder, alignment, or numbering pipeline. Training/reference-set
overlap has not been audited. No scientific assets or numbering rules are
changed by this benchmark.

To deliberately rebuild the fixtures, download each unique source URL listed
in the manifest into a separate directory. Name inputs `<pdb>_raw.pdb`,
`<pdb>_raw.cif`, and `<pdb>_annotations.json` using lowercase PDB IDs. Negative
controls need only the PDB file. Check the recorded source hashes; a mismatch
means the upstream source changed and needs review. Then run:

```bash
python -m benchmarks.antibody.curate /path/to/sources --retrieved YYYY-MM-DD
```

[curate.py](curate.py) verifies residue identity against the deposited sequence,
extracts annotations without importing SAbR, and writes the fixtures, labels and
manifest. Review any resulting diff before updating the reports or test limits.

## Metrics

The public `renumber_structure` API supplies the scored output residue IDs.
A second execution of the real encoder and alignment supplies domain boundaries:
output numbering alone cannot distinguish aligned residues from automatically
numbered terminal flanks.

- **Residue agreement:** exact `(domain, IMGT position, insertion code)` agreement
  over independently labeled, resolved variable-domain residues. The documented
  1000-position offset is removed conceptually by comparing each ordered domain.
  Linker and constant-domain residues are outside this denominator. Reports show
  both all-positive agreement (rejections contribute zero matches) and agreement
  conditional on acceptance, with explicit numerators and denominators.
- **Domain boundaries:** expected first/last resolved rows from the labels versus
  the first/last rows assigned by the real corrected alignment, paired by domain
  order and chain type. A domain is exact only when both endpoints match.
  Missing or wrong-type domains fail exact agreement. Signed endpoint errors
  and absolute error over matched endpoints are reported alongside the number
  of matched endpoints and cases with a different domain count. Rejected domains
  remain in the exact-boundary denominator and are absent from conditional MAE.
- **Rejection:** separate counts/rates for positive biological cases,
  non-antibody controls, and default structural-gap policy controls. A positive
  rejection is a false rejection; a negative acceptance is a false acceptance.
  Undefined rates are `null`, not zero.

Detailed JSON includes every numbering mismatch and every predicted/expected
boundary, plus summaries for each category. The aggregate includes the two
controlled truncations, so use category-level results when comparing constructs.

## Initial results and known limitations

The checked-in [SAbR report](results-sabr.json) and
[SoftAlign report](results-softalign.json) have the same discrete results:

| Metric | Result |
| --- | --- |
| Residue agreement, all positives | 1494/1502 (99.47%) |
| Residue agreement, accepted positives | 1494/1502 (99.47%) |
| Exact domain boundaries | 12/13 (92.31%) |
| Mean absolute endpoint error, matched domains | 1/26 residues (0.0385) |
| Positive rejections | 0/9 |
| Non-antibody rejections | **0/2: both controls falsely accepted** |
| Default gap-policy rejections | 2/2 |

The C-terminally truncated VH accounts for all eight numbering disagreements.
In 6NOU, the VH alignment extends one residue into the linker; numbered variable
residues still agree. The full conventional domains, VHH, and both complete-linker
diabody chains agree at every labeled residue and both boundaries.

SAbR is a renumbering pipeline without a calibrated antibody/non-antibody
classifier. Acceptance alone is not evidence that an input is an antibody.
Negative-control expectations are explicit **strict expected failures** in the
test suite, so a future rejection improvement prompts review rather than
requiring continued false acceptance. These failures remain visible in pytest
and in the reports. Positive-case limits allow the documented eight numbering
errors and one endpoint error; all other positive labels and boundaries must
agree exactly. Improvements pass without changing the independent labels.

Invariance tests call the complete public API for every case in both modes,
under a general rigid rotation, translation, arbitrary nonmonotonic/negative
residue numbering with insertion codes, and a combined transformation. They
transform every alternate conformer, require identical residue assignments or
rejection results, and check that inputs are not mutated.

## Attribution

Coordinates: wwPDB/RCSB PDB, with the original deposition and publication linked
above. SAbDab-derived annotations are attributed to the Oxford Protein
Informatics Group and distributed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), as declared by the
[SAbDab API](https://sabdab.opig.stats.ox.ac.uk/api/openapi.json). The CSV files
are derived extracts of those annotations, mapped to resolved coordinates;
truncation labels are subsets. See also
[Dunbar et al., 2014](https://doi.org/10.1093/nar/gkt1043) and
[Schneider et al., 2022](https://doi.org/10.1093/nar/gkab1050).
