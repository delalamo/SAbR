# Verification of DE3 (VH CDR3/FR4 slip predicts a VL error in the same Fab): confounding lens

**Verdict: confirmed.** The association survives every adjustment tried. Two qualifications:
* It is not specific to D/E slips.
* The named examples include two entries that the confirmed analysis does not count.

## Reproduction (`v01_reproduce.py`, `tables/hDE_entries.csv`)
* The finding reproduces exactly. Among 2,123 cryo-EM Fab entries (VHH excluded), 32 have a confirmed VH D/E run.
  Of these, 11 have a VL non-D/E first-occurrence error (34.4%), vs 27/2,091 (1.3%) of the others.
* With year bin + resolution and SEs clustered by paper: adjusted OR 36.2, p = 3e-9.
* Counting any VL error (including 8GNH's D−), it is 12/32.
* **Concentration.** 5 of the 11 events come from one paper: Zhou, Cell Res 2021, entries 7CZR/7CZS/7CZU/7CZV/7D00.
  These are three Fab units, and all their κ chains have N-terminal slips. The 11 events come from 8 Fab units in 7
  papers.

## Robustness (`v02_confounding.py`, `v03_more.py`, `v04_strata_lab.py`; `tables/confounding_tests.csv`, `more_tests.csv`, `strata_lab.csv`)

| Check | Exposed | Control | OR | p |
|---|---|---|---|---|
| Entry level, + SARS-CoV-2 + log(n Fab copies) + κ | 11/32 | 27/2,091 | 38 | <1e-6 |
| + Fab map-model fit (lq_z) | | | 45 | <1e-6 |
| + rigid-body-only fit + PDB starting model | | | 43 | <1e-6 |
| + unbuilt internal V residues (H, L) + L N-terminal truncation | | | 36 | <1e-6 |
| + VH has another (non-D/E) error | | | 19 | 8e-4 |
| Fab-in-paper units (identical H/L sequences in a paper collapsed) | 8/24 | 26/1,617 | 29 (CI 11–79) | <1e-6 |
| Fab units, + SARS-CoV-2 + lq_z + VH other error | | | 18 | <1e-4 |
| Mantel–Haenszel over year × resolution × SARS-CoV-2 strata (Fab units) | | | 25 | <1e-6 |
| Fab units, SEs clustered by last author | | | 33 | <1e-6 |
| One random entry per paper (2,000 draws) | | | median 32 | p<0.05 in 100% of draws |
| One random Fab unit per lab (2,000 draws) | | | median 33 | p<0.05 in 100% of draws |
| Adjusted for the VL/VH error rate in the paper's *other* Fab units | | | 24 | 1e-4 |
| Drop Zhou 2021 | 6/27 entries | | 20 | <1e-6 |
| Drop Zhou 2021 + Zhu 2025 | | | 20 | <1e-6 |
| Leave-one-paper-out | | | 22–47 | all p<3e-6 |
| VL outcome = any disagreement run (incl. copied/artifact; tests first-occurrence selection) | 14/32 | 66/2,091 | 19 | <1e-6 |
| VL error not at a chain end/unbuilt loop (origin ≠ 3) | 6/32 | 24/2,091 | 16 | <1e-6 |
| X-ray, same definition | 2/4 | 3/1,056 | large | 1e-4 (tiny n) |

## Qualifications
* **Not D/E-specific.**
  * In one model, VH D/E-only gives OR 46 and VH other-error-only (A/B/C/F) gives OR 13; equality p = 0.17.
  * Any VH error predicts any VL error with OR 18.
  * The co-occurring VL errors are mixed: mode A ×6 (five from the Zhou paper), C ×2, B ×2 (one with E), F ×1.
  * So the evidence points to "this Fab model was built badly" rather than a matched CDR3/FR4 mechanism. D/E–D/E
    co-occurrence is only 2/32 (8GNH, 9PV5).
* **Partly paper-level.**
  * In papers that contain an H-D/E Fab unit, the *other* Fab units have VL errors in 4/38 (10.5%), vs 22/1,579 (1.4%)
    elsewhere (adjusted OR 8.7, p = 0.04). All four events come from one paper (Zhu 2025, 8W5x series).
  * The within-entry association is still much stronger, and survives adjustment for this paper-level rate.
* **Same Fab vs same entry is barely testable.**
  * Nearly all exposed entries contain one Fab sequence.
  * Among entries with ≥2 distinct Fabs, three have errors in both a VH and a VL:
    * 9PV5 and 9PV3: same Fab;
    * 7WP6: different Fabs.
  * A within-paper conditional logit has only 3 informative papers and is uninformative.
* **Examples.** In the claim's example list, 7D3K (VH D+ "not confirmed") and 26PF (VL D− "not confirmed") are *not*
  co-occurrences in the confirmed analysis. Only 8GNH and 9PV5 are.
* **Simpler factors do not explain it.**
  * Map-model fit (lq_z), unbuilt internal residues, N-terminal truncation, year, resolution, SARS-CoV-2 and number
    of copies leave the OR at 36–45.
  * VH unbuilt residues alone do not predict VL errors (OR 1.3, p = 0.68).
