# Verification: pdb_meta "amyloid_artifact" (confounding lens)

**Verdict: confirmed.** The AL amyloid fibril entries are detector artifacts, not register errors. Year, method,
resolution, chain type, SARS-CoV-2 status, paper clustering, unbuilt residues and local map quality cannot explain
the association. The structural evidence is independent of SAbR.

## Reproduction (01_reproduce_counts.py)
* The regex `amyloid(?!-beta)` matches exactly 15 entries. All are cryo-EM λ light-chain fibrils, and all 15 are
  `err_first`. No clean entry and no X-ray entry matches. 13 of the 15 also carry the keyword PROTEIN FIBRIL, and no
  other entry in the set has that keyword.
* These counts reproduce exactly: 15/193 EM error entries, 12/218 error units, 77/488 first-occurrence runs (15.8%),
  12/26 λ error units, and 6/66 mode-A units.
* Minor discrepancies:
  * IGLV1-44 appears in 6 entries, not 7.
  * 9lja and 9ljb have L1 = 9, not 8 or 6.
  * The per-unit λ rates come out as 8.4% → 4.7% (L1 = 8) vs 2.1% → 1.5% (L1 = 9). The claim gives 7.9% → 4.1% vs
    1.5%; the unit definitions probably differ.
* The 15 entries come from 7 papers by 6 labs: Fändrich (Radamaker 2019, and two papers in 2021), Saelices 2024,
  Ricagno 2024, Li D. 2025 and Huang H. 2025.

## Why they are artifacts (independent of the detector)
* **Fold test (02).** CA atoms at sequence-based IMGT framework positions were superposed on 5 reference V domains
  of the same chain type.
  * Amyloid chains: RMSD 18–26 Å (n = 77).
  * Clean chains: median 1.1 Å, maximum 3.5 Å (n = 1,228).
  * Non-amyloid error chains: median 1.6 Å.
  * With SAbR's own numbering, the amyloid chains are still 17–27 Å from the references. No Ig-like mapping exists.
  * No non-amyloid error unit has this non-Ig signature (>10 Å under both numberings). The title-based exclusion is
    complete for non-Ig folds.
* **Run signature.**
  * Amyloid runs cover the whole domain: 63–107 residues, offsets −1 to −48 (median −17).
  * Other runs: median length 8, offsets almost all ±1.
  * SAbR and ANARCI disagree at 97% of V residues in amyloid chains (median 100%). The figure is about 11% for
    other error chains and about 0% for clean chains.
* **Map fit (05).** Amyloid chains fit their maps *better* than clean EM chains (entry median lq_z 0.90 vs 0.38,
  p = 8e-5, n = 15 vs 2,751). Real errors fit worse (0.10, p = 2e-7). So they lack the poor-density signature of
  real errors.
* **Expected count.** A year-bin + resolution + log2(nV) model fitted to non-amyloid EM entries predicts 0.76 flagged
  entries among the 15. All 15 are flagged (P ≈ 2e-20). By paper, 7/7 are flagged.
* **Coverage.** Two more AL fibrils at 4.0 Å, 6HUD and 8CPE, were scanned. They were dropped only by the
  "≥80 modelled V residues" filter (each has 77). SAbR and ANARCI disagree at 90% and 100% of their residues. That
  makes 17/17 scanned AL fibrils "flagged". 6Z1I (3.4 Å) was not scanned.

## Secondary claims (03_secondary_claims.py; BH q across 21 tests)
* **"Antibody alone" excess (confirmed).** 16/193 vs 14/2,574 (p = 1e-11). After exclusion: 1/178 vs 14/2,574
  (p = 1.0). The one remaining entry is 9ZXD.
* **λ L1 = 8 vs L1 = 9 (partly confirmed).**
  * Chain-level adjusted OR (year bin, resolution, method; clustered by paper) falls from 8.4 (2.2–32) to 2.6
    (0.7–9.3), p = 0.15.
  * At unit level the drop is smaller: OR 4.5 → 3.4 (1.06–10.8), p = 0.040, q = 0.057.
  * EM only, unit level: OR 2.7 (0.75–9.9), p = 0.13.
  * "Largely produce" is true at chain level. A weak residual of about 3× remains per unit: 8 non-amyloid L1 = 8
    error units out of 171.
* **Not reported by the investigator.** With the amyloid runs included, the "chain end / next to an unbuilt loop"
  share is inflated: 44.5% → 34.8% of runs, 41.2% → 38.2% of units. Amyloid runs also make up 35/47 "other"-mode
  runs. None of the amyloid entries records a starting model, so starting-model analyses must exclude them too.

## Side observation (04, not part of this finding)
* 6 non-amyloid error units are bovine ultralong-CDR3 scFvs against FMDV: 7D3K/L/M/R, 7FEI/7FEJ and 8GRR, from 3
  papers. Their SAbR numbering fits the reference V domains much worse than ANARCI's: 8.8–17 Å vs 1.0–3.3 Å
  framework RMSD.
* SAbR numbers past IMGT 128 inside the CDR3. This looks like a second class of detector failure, and it should be
  reviewed separately.

## Files
* Scripts: `01_reproduce_counts.py`, `02_fold_test.py`, `03_secondary_claims.py`,
  `04_other_artifact_signatures.py`, `05_mapfit_and_scan_coverage.py`.
* Tables: `fold_test_chains.csv`, `fold_test_summary.csv`, `secondary_claims.csv`, `mapfit_tests.csv`,
  `al_fibril_scan_coverage.csv`, `disagreement_fraction_by_group.csv`, `artifact_signature_chains.csv`.
