# Fast register-error screen for antibody V domains: SAbR vs ANARCI

## Idea

SAbR numbers each residue from backbone geometry alone; ANARCI numbers it from the deposited
sequence. If a model is correct, both give the same IMGT number. If a segment is threaded off
register, the backbone sits at IMGT position *p* but the residue placed there belongs at
*p ± k*, so the two numberings disagree by *k* across the segment.

**The detector.** A V domain is flagged if it has a run of ≥3 consecutive V-region residues whose
SAbR and ANARCI (IMGT) labels differ.

**Implementation:**
* Per-chain coordinates come from the PDBe model server.
* Unmodelled residues and the author-to-sequence mapping come from RCSB GraphQL.
* SAbR runs on the V region only (the chain is trimmed to it), with the chain type from ANARCI.
* This gave identical V-region numbering to full-chain SAbR on 124 of 125 chains tested.
* SAbR time is about 0.3 s per chain on a CPU.
* About 10,000 chains were processed in about 25 minutes on 4 cores; that run was limited by network
  requests.

**Multi-domain chains:**
* **scFvs (heavy + light, either order)** use `scfv=True`.
* **Other arrangements** (tandem VHH `HH`, `HLL`, `HKHK`, …) pass the ANARCI domain order as
  `chain_type`.
* **Offsets.** SAbR offsets domain *i* by 1000·*i*; the offset is removed before comparing with
  ANARCI.
* **Single modelled domain.** If only one domain of a multi-domain chain has coordinates (≥20
  residues), that domain is numbered on its own in single-domain mode. Otherwise SAbR can assign a
  tag or linker to the empty reference domain; see the PR notes.
* **Coverage.** 807 of 820 multi-domain chains were processed: 745 in scFv mode, 34 as a single
  modelled domain, 22 with an explicit domain order.

## Validation against the 2024 conkit-validate flags (3–5 Å, snapshot of April 2022)

**Sample.** All 87 entries with a flagged antibody chain, plus 300 random antibody entries with none:
1,539 chains processed, including scFvs.

| | |
|---|---|
| Flagged V-region errors overlapped by a disagreement run | 98/134 (73.1%) |
| Framework errors (≥75% FR residues) | 84% |
| Mixed FR/CDR errors | 77% |
| Errors mostly inside a CDR | 16%. IMGT numbering inside CDRs is length-based, so a register shift there is invisible by design |
| Detected by error length | ≤6 residues 56% · 7–10 residues 74% · >10 residues 92% |
| Shift magnitude matches conkit's | 97% (sign convention is opposite) |
| V domains with a flagged error that show ≥1 run | 91% |
| Unflagged V domains with ≥1 run | 4.7% (includes errors conkit missed, e.g. copies) |

**Map check (independent of both methods).** I ranked how well each segment fits the experimental
map among the other residues of the same V domain (RSCC for X-ray, Q-score for EM), and compared
with the same IMGT positions in V domains with no disagreement:
* **X-ray:** detected runs sit at a median 16th–17th percentile, vs 35th–38th for the controls
  (p ≤ 0.0003).
* **EM:** the difference is smaller, a median 35th–36th vs 41st–45th percentile (p ≈ 0.03–0.09).
  About a third of EM detections fall below the 25th percentile, vs about a sixth of controls.

**Known artifact.** Antibodies with framework insertions relative to germline place those insertions
differently in SAbR and ANARCI. Examples: HIV bnAbs 3BNC117, VRC03, 8ANC195, VRC40.01, mostly in
FR3/DE-loop IMGT 80–86. These recur as false "errors"; runs with no measurable framework shift should
be treated with caution.

## Newer structures (released after 5 Apr 2022, 3.0–5.0 Å, X-ray or EM)

**What was scanned:**
* **Newer set:** 2,961 entries with antibodies, giving 9,745 analysable V domains or chains.
* **Baseline:** the full April 2022 snapshot, scanned the same way (6,891).

**How to read the rates.** All rates below come from the same detector, so they compare across eras.
They are **not** comparable in absolute terms to the paper's flag rates: this detector catches errors
conkit missed, and also has some false positives. For multi-domain chains, a chain counts once if
any of its V domains has a run.

| % with ≥1 disagreement run | Cryo-EM | X-ray |
|---|---|---|
| Deposited ≤2017 | 6.3% (n=189) | 1.9% (n=2,252) |
| 2018–19 | 8.4% (n=818) | 1.6% (n=708) |
| 2020–21 | 7.7% (n=2,482) | 1.7% (n=989) |
| 2022–23 | **3.3%** (n=3,526) | 1.3% (n=908) |
| 2024–26 | **4.9%** (n=4,135) | 1.9% (n=629) |

Adjusted odds ratios vs 2018–19 (method × resolution, chain type, V length):
2020–21 0.89 · 2022–23 **0.37** (0.24–0.58) · 2024–26 **0.51** (0.32–0.80).

**scFvs and other multi-domain chains** (per chain, two or more V domains):

| | Cryo-EM | X-ray |
|---|---|---|
| Snapshot | 0/83 | 12.7% of 110 |
| Newer | 5.9% of 528 | 8.0% of 50 |

The scFvs include the G-protein stabiliser scFv16. Since 2022 it carries two recurring identical
disagreements, one in FR1 (4 entries, 2 labs) and one in CDR3 (2 entries, 2 labs).

### How the earlier findings hold up

| Earlier finding (from conkit flags, ≤ Apr 2022) | With SAbR/ANARCI, snapshot | Newer structures |
|---|---|---|
| Antibody error rate rose through 2020–22 in cryo-EM | high 2018–21 (7.7–8.4%) | **fell by about half** after 2022, slight uptick 2024–26 |
| Cryo-EM worse than X-ray at the same resolution | OR 4.5 (2.9–7.2) | OR 2.9 (1.6–5.0): gap narrower but persists |
| SARS-CoV-2 2020 spike (6.4% vs 1.4% other viruses) | 2020: SARS-CoV-2 11.2% vs other viral 11.0%, so **no SARS-CoV-2-specific excess** | SARS-CoV-2 1.7–1.9% vs other EM antibodies 3.9–5.4% |
| No difference between auxiliary and antibody-of-interest | interest 8.1% vs auxiliary 2.9% (EM); adjusted OR auxiliary 0.25 | no difference (3.3% vs 4.0%; OR 0.63, n.s.) |
| A prior ≤3 Å structure of the same V region protects (OR 0.32) | **not reproduced** (5.6% vs 4.4%; OR 1.44) | not protective (3.0% vs 3.9%; OR 0.82, n.s.) |
| Errors copied across papers via templates | 12 error families in ≥2 labs; Fab 1B2 continues to 2025 | 12% of newer runs repeat an earlier error from another paper; scFv16 recurs |
| Hot spots: light FR1, heavy FR3 / FR4 | light FR1 52%; heavy FR3 49%, CDR3 flank 21% | light FR1 60%; heavy FR3 31%, FR4 15%, CDR3 flank 18% |

Journal tier, newer structures: CNS 3.0%, CNS sister journals 2.9%, other journals 4.8%,
unpublished 4.6%.

## Caveats

* **Blind spot.** The detector can't see register errors inside CDRs, including CDR-H3.
* **False positives.** Framework-insertion conventions cause some (see above), and the EM map signal
  is modest.
* **Mismatched detectors.** The prior-template result disagrees with the conkit-based result. A
  plausible reason is that conkit-validate compares models to AlphaFold2, which may reproduce an
  erroneous high-resolution template it was trained on. SAbR and ANARCI don't depend on a predictor's
  training data. This remains a hypothesis.
* **Incomplete recent data.** 2025–26 entries are incomplete because many are still held for
  release.
* **Uninterpretable chain-type odds ratio.** In the adjusted model, the multi-domain chain-type term
  is collinear with V length (two domains), so its odds ratio is meaningless. Use the raw rates above.

## SAbR issues found during this scan

Fixes were submitted as separate pull requests:
1. **Selections too short to number** (1–4 residues; e.g. a V-region range on a chain whose V domain
   is unmodelled, such as 8TGZ M, 8TXQ E, 8U9Y M/N, 5KOV Q). These crashed with an internal
   `TypeError` or a NumPy reduction `ValueError` instead of a clear error.
2. **Biopython `Structure` objects built in code have no `header` attribute.** `renumber_structure`
   raised `AttributeError`.

## Files

* `scripts/`:
  * `sabr_scan_light.py`: the single-domain scan.
  * `sabr_scan_multi.py`: the scan with scFv and multi-domain support.
  * `anarci_num.py`: ANARCI IMGT numbering, cached.
  * `sabr_eval.py`, `sabr_validate.py`: run detection and validation.
  * `sabr_patterns.py`: the era comparison.
  * `sabr_mapfit.py`: the map check.
  * `gql_new.py`: metadata for the newer entries.
  * `sabr_scan.py`: the original full-mmCIF scan.
* `tables/`:
  * `v_domain_flags.csv.gz`: per V domain or chain, with metadata.
  * `disagreement_runs.csv`, `identical_error_families.csv`, `mapfit_detections.csv`.
  * Logs: `validation.log`, `patterns.log`, `mapfit.log`.
