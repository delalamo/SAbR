# Verification of DE2_framework_distortion (data-integrity and statistics lens)

**Verdict: confirmed as an association; the scope is narrower than the wording suggests.** The numbers reproduce
exactly and hold up under every robustness check below. The deviation is not specific to the erroneous chain or to
modes D/E: it marks the whole deposited model (entry) and is shared by run-free chains of the same entry and by other
error modes.

## Recomputation
I wrote new code (`v01_recompute_fw.py`) that does not import the investigator's module.
* `fw_rmsd` was reproduced for all 525 overlapping chains (max |diff| 2e-7 Å).
* Headline numbers reproduced (`v02_stats.py`, `verify_tests.csv`):
  * matched: 1.56 vs 1.02 Å, OR 3.18 per SD (SD 0.47 Å), p = 6e-8, 49 errors vs 271 controls;
  * within paper: OR 10.05 per SD, p = 2.2e-4, q = 0.004.
* Reporting issue: the within-paper conditional logit uses only **25 errors in 21 papers** (mean 1.48 vs 1.09 Å).
  The "1.56 vs 1.09" pairs the mean of all 49 errors with the within-paper control mean.
* Integrity gap: the script that wrote `X_struct.pkl` and `err_struct.pkl`, which holds the "confirmed" labels, is not
  in `case_files_DE/scripts/`.
  * This does not matter for this finding. The "all D/E" set gives the same result (OR 3.44).
  * Confirmed and not-confirmed D/E chains have the same RMSD (median 1.45 vs 1.36 Å, p = 0.76).

## Robustness checks (all still significant)
* **Distribution and outliers:**
  * rank test: AUC 0.76, p = 6e-9;
  * log RMSD: OR 3.4;
  * dropping the top 1, 3 or 5 error outliers (7MEJ-B 4.5 Å, 8KHR, 7D3K, 7STF, 7MKL): OR 2.9, 2.6, 2.4, all p < 1e-4;
  * values winsorised at 2.5 Å: OR 3.0.
* **Antibody type:** adding species, V-gene identity, CDR3 length and SARS-CoV-2 leaves OR at 3.2–3.3.
* **Map fit and resolution:**
  * matched, + chain-level map-model fit (lq_z): OR 2.9, p < 1e-4;
  * within paper, + resolution + lq_z: OR 7.3, p = 0.0025.
* **Paper level:** 42 error papers vs 240 control papers, p < 1e-4.
* **Leave-one-paper-out:**
  * matched: min OR 2.87;
  * within paper: OR 5.8–15.9, max p = 6e-4.
  * Within paper × chain type: OR 10.9, p = 0.003.
  * Per paper: errors exceed controls in 18 of 21 papers (median +0.28 Å, Wilcoxon p < 1e-4).
* **Away from the slip:**
  * superposing on FR1+FR2 only: matched OR 2.6, within paper OR 5.1;
  * FR1 and FR2 regional RMSD are elevated on their own;
  * excluding every SAbR≠ANARCI residue changes nothing, so there is no circularity with the detector.
* **Reference choice:** using the mean or minimum over a panel of 4 references per chain type, with a VHH panel for
  nanobodies: OR 3.1, and within paper 9–10.
* **Fresh control sample:** 136 new clean cryo-EM units, matched by year × chain type × nanobody: 1.56 vs 0.97 Å,
  OR 6.8, p < 1e-4.

## What weakens the interpretation
1. **Not specific to D/E.** EM chains with other first-occurrence modes (A, B, C, F, other; n = 104) are just as
   elevated:
   * 1.46 vs 1.02 Å against matched clean, OR 3.5;
   * D/E vs other modes: OR 1.2, p = 0.32 (FR1+FR2 metric p = 0.69).
2. **Mostly entry-level, not chain-level.** Run-free chains in the same D/E entries are also elevated (`v06`):
   * against matched clean: median 1.21 vs 0.97 Å, p = 8e-4;
   * entry-paired, error chain vs correct chains: 1.71 vs 1.54 Å, 14/26 entries higher, p = 0.14, n.s.
   * Within the same chain type and entry: 7/7 higher, p = 0.016 (q = 0.06, small n).
   * Only one identical-antibody copy pair exists (7MLV: 2.28 vs 1.36 Å).
3. **Effect of resolution in controls:** rho = 0.21. This is adjusted for, but the residual effect of local resolution
   was not measured.

## Files
* Scripts: `v01_recompute_fw.py`, `v02_stats.py`, `v03_extra_samples.py`, `v04_adjust_mapfit.py`, `v05_extra_tests.py`,
  `v06_same_entry.py`.
* Tables: `fw_recomputed*.csv`, `sample_groups*.csv`, `analysis_table.csv`, `verify_tests.csv`,
  `verify_mapfit_adjust.csv`, `verify_extra_tests.csv`, `verify_same_entry.csv`, `within_paper_pairs.csv`,
  `lopo_*.csv`, `same_antibody_same_entry_pairs.csv`.
