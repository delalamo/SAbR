# Fast register-error screen for antibody V domains: SAbR vs ANARCI

## Idea

SAbR numbers each residue from backbone geometry alone; ANARCI numbers it from the deposited
sequence. If a model is correct, both give the same IMGT number. If a segment is threaded off
register, the backbone sits at IMGT position *p* but the residue placed there belongs at
*p ± k*, so the two numberings disagree by *k* across the segment.

**The detector.** A V domain is flagged if it has a run of ≥3 consecutive V-region residues whose
SAbR and ANARCI (IMGT) labels differ. Only single-domain antibody chains (H, K or L) are included;
scFvs and other multi-domain chains are skipped.

**Implementation:**
* Per-chain coordinates come from the PDBe model server.
* Unmodelled residues and the author-to-sequence mapping come from RCSB GraphQL.
* SAbR runs on the V region only, with the chain type from ANARCI.
* This gave identical V-region numbering to full-chain SAbR on 124 of 125 chains tested.
* SAbR time is about 0.3 s per chain on a CPU.
* 9,191 chains were processed in about 20 minutes on 4 cores; that run was limited by network
  requests.

## Validation against the 2024 conkit-validate flags (3–5 Å, snapshot of April 2022)

**Sample.** All 87 entries with a flagged antibody chain, plus 300 random antibody entries with none:
1,498 chains processed.

| | |
|---|---|
| Flagged V-region errors overlapped by a disagreement run | 95/131 (72.5%) |
| Framework errors (≥75% FR residues) | 84% |
| Mixed FR/CDR errors | 76% |
| Errors mostly inside a CDR | 16%. IMGT numbering inside CDRs is length-based, so a register shift there is invisible by design |
| Shift magnitude matches conkit's | 97% (sign convention is opposite) |
| V domains with a flagged error that show ≥1 run | 91% |
| Unflagged V domains with ≥1 run | 4.8% (includes errors conkit missed, e.g. copies) |

**Map check (independent of both methods).** I ranked how well each segment fits the experimental
map among the other residues of the same V domain (RSCC for X-ray, Q-score for EM), and compared
with the same IMGT positions in V domains with no disagreement:
* **X-ray:** detected runs sit at a median 15th–17th percentile, vs 34th for the controls
  (p ≤ 0.001).
* **EM:** the difference is smaller, a median 35th vs 41st–45th percentile (p ≈ 0.05). About a third
  of EM detections fall below the 25th percentile, vs about a sixth of controls.

**Known artifact.** Antibodies with framework insertions relative to germline place those insertions
differently in SAbR and ANARCI. Examples: HIV bnAbs 3BNC117, VRC03, 8ANC195, VRC40.01, mostly in
FR3/DE-loop IMGT 80–86. These recur as false "errors"; runs with no measurable framework shift should
be treated with caution.

## Newer structures (released after 5 Apr 2022, 3.0–5.0 Å, X-ray or EM)

The newer set is 2,961 entries and 9,848 antibody chains, of which 9,191 were processed. The full
April 2022 snapshot was scanned the same way as a baseline (6,692 V domains). All rates below come
from the same detector, so they compare across eras. They are **not** comparable in absolute terms
to the paper's flag rates: this detector catches errors conkit missed, and also has some false
positives.

| % of V domains with ≥1 disagreement run | Cryo-EM | X-ray |
|---|---|---|
| Deposited ≤2017 | 6.8% (n=177) | 1.5% (n=2,185) |
| 2018–19 | 8.6% (n=801) | 0.9% (n=670) |
| 2020–21 | 7.8% (n=2,390) | 1.8% (n=965) |
| 2022–23 | **3.2%** (n=3,268) | 1.2% (n=886) |
| 2024–26 | **4.9%** (n=3,875) | 1.5% (n=612) |

Adjusted odds ratios vs 2018–19 (method × resolution, chain type, V length):
2020–21 0.94 · 2022–23 **0.37** (0.23–0.58) · 2024–26 **0.51** (0.32–0.81).

### How the earlier findings hold up

| Earlier finding (from conkit flags, ≤ Apr 2022) | With SAbR/ANARCI, snapshot | Newer structures |
|---|---|---|
| Antibody error rate rose through 2020–22 in cryo-EM | high 2018–21 (7.8–8.6%) | **fell by about half** after 2022, slight uptick 2024–26 |
| Cryo-EM worse than X-ray at the same resolution | OR 5.7 (3.5–9.2) | OR 3.3 (1.8–6.0): gap narrower but persists |
| SARS-CoV-2 2020 spike (6.4% vs 1.4% other viruses) | 2020: SARS-CoV-2 11.2% vs other viral 11.0%, so **no SARS-CoV-2-specific excess** | SARS-CoV-2 1.6–1.8% vs other viral 4.5–5.2% |
| No difference between auxiliary and antibody-of-interest | interest 8.1% vs auxiliary 3.3% (EM); adjusted OR auxiliary 0.31 | no difference (3.2% vs 3.8%; OR 0.78, n.s.) |
| A prior ≤3 Å structure of the same V region protects (OR 0.32) | **not reproduced** (5.9% vs 4.2%; OR 1.49) | not protective (2.7% vs 3.8%; OR 0.91, n.s.) |
| Errors copied across papers via templates | 10 error families in ≥2 labs; Fab 1B2 continues to 2025 | 13% of newer runs repeat an earlier error from another paper |
| Hot spots: light FR1, heavy FR3 / FR4 | light FR1 52%; heavy FR3 49%, CDR3 flank 21% | light FR1 60%; heavy FR3 32%, FR4 15%, CDR3 flank 18% |

Journal tier, newer structures: CNS 2.5%, CNS sister journals 2.5%, other journals 4.9%,
unpublished 4.7%.

## Caveats

* **Blind spot.** The detector can't see register errors inside CDRs, including CDR-H3.
* **False positives.** Framework-insertion conventions cause some (see above), and the EM map signal
  is modest.
* **Mismatched detectors.** The prior-template result disagrees with the conkit-based result. A
  plausible reason is that conkit-validate compares models to AlphaFold2, which may reproduce an
  erroneous high-resolution template it was trained on. SAbR and ANARCI don't depend on a predictor's
  training data. This remains a hypothesis.
* **Incomplete recent data.** 2025–26 entries are incomplete because many are still held for
  release. scFvs (614 chains) are excluded.

## Files

* `scripts/`:
  * `sabr_scan_light.py`: the scan (fast mode is the default).
  * `anarci_num.py`: ANARCI IMGT numbering, cached.
  * `sabr_eval.py`, `sabr_validate.py`: run detection and validation.
  * `sabr_patterns.py`: the era comparison.
  * `sabr_mapfit.py`: the map check.
  * `gql_new.py`: metadata for the newer entries.
  * `sabr_scan.py`: the original full-mmCIF scan.
* `tables/`:
  * `v_domain_flags.csv.gz`: per V domain, with metadata.
  * `disagreement_runs.csv`, `identical_error_families.csv`, `mapfit_detections.csv`.
  * Logs: `validation.log`, `patterns.log`, `mapfit.log`.
