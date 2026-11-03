# Verification: struct_unusual "antibody_vs_rest_of_model" (data integrity and statistics)

**Verdict: weakened.** The Ramachandran-'allowed' gap (V minus rest of the same model) is real and robust. Its size is
overstated, partly by a bug, and about half of it comes from the error-bearing chains. The clash part is weaker, and the
claim that the rest of the model is "as good as in controls" does not hold for clashes.

## What reproduces
* I re-parsed the 344 validation XMLs and re-labelled the V residues from the raw SAbR records. Per-entry V and rest
  values match `entry_vrpt.csv` exactly (max |diff| 1e-14).
* The investigator's p-value reproduces exactly from their table: effect 3.456, p = 2.74e-12.

## Problems found
1. **Bug: an empty set is scored as 0, not NaN** (`s05b_entry.py`, `100*x/max(n,1)`).
   * 10 case entries have no non-V residues. Their rest value was set to 0, so their "gap" equals their whole
     V-domain value (mean about 9.6 pp).
   * 1 control entry (8i4e: SAbR "multi-domain chain skipped") has no V residues. Its V value was set to 0, giving a
     gap of −6.1.
   * After the fix:
     * Ramachandran gap: 3.04 pp (2.09–4.00), p = 4e-10 (claimed 3.46).
     * Clash gap: 2.40 pp (0.55–4.25), p = 0.011 (claimed 3.60, p = 2.6e-4).
2. **12 of the 159 case entries are AL amyloid fibrils.** These are light-chain V sequences in a non-Ig fold, so they are
   not antibody models.
   * Across the project, all 15 cryo-EM amyloid/fibril entries are flagged `err_first`. That is 7.8% of the 193 cryo-EM
     error entries.
   * This contaminates the error set for every lens, not just this one.
   * Dropping them barely moves this test: 3.06 pp.
3. **The rest of the model is not as clean as claimed.** The zero-filled case entries had pulled down the case mean for
   rest-of-model clashes. After the fix, clashes in non-V residues are higher in cases: +2.6 pp (0.3–4.9), p = 0.029.
   In antigen/other chains alone: +2.6 pp, p = 0.03.
   * About half of the V clash excess (+5.2 pp) is therefore entry-wide.
   * Non-V Ramachandran, rotamer and Q-score remain n.s.
4. **Overlap with the error definition.** A disagreement run covers 14% of V residues in case entries. The table below
   shows the Ramachandran gap as the error sites are removed, adjusted for year bin and resolution (cases/controls =
   147/184 unless stated):

   | V residues kept | Gap | p |
   |---|---|---|
   | All | 3.06 pp | 4e-10 |
   | Excluding run residues | 2.82 pp | |
   | Excluding run ±3 | 2.58 pp | |
   | Excluding run ±10 | 2.54 pp | 2e-7 |
   | Only V chains with no run of any kind (n 107/184) | 1.71 pp (0.78–2.65) | 3e-4 |

   * The last row is about 50% of the claimed effect.
   * The clash gap in error-free chains is not significant (p = 0.06; Mann–Whitney p = 0.16).
   * The clash gap is also not significant within 2020–22 (p = 0.10).

## Robustness of the Ramachandran gap (no-amyloid, fixed table)
* Clustering by paper or by last author: p < 1e-6.
* Leave-one-paper-out and leave-one-lab-out:
  * all V: max p < 1e-5;
  * error-free chains: max p ≈ 1e-3.
* Survives adjusting for each of:
  * SARS-CoV-2, or restricting to non-SARS-CoV-2 entries;
  * restricting to 2020–22: +3.1, p = 2e-4 (error-free chains +1.5, p = 0.04);
  * continuous year (quadratic);
  * auxiliary-reagent role, nanobody content and V share of the model;
  * V and rest Q-scores: effect unchanged; the Spearman correlation between the Ramachandran gap and the Q gap is 0.00.
* The V-minus-rest Q gap is n.s. (−0.015, p = 0.18). However, V-domain Q is itself lower in cases (−0.025; adjusted
  p = 0.07; Mann–Whitney p = 0.02).

## Files
* `v01_residues.py`: independent per-residue labels.
* `v01b_runmask.py`: masks built from the project's run table. Run indices i0/i1 = SAbR record index (verified 97.6%).
* `v02_entry.py`: cross-check against the investigator's table.
* `v03_stats.py`: produces `tests_verify.csv` (93 tests, BH q) and `leave_one_out.csv`.
* `v04_reproduce_and_q.py`: exact reproduction and the density adjustment.
* `v05_confounders.py`: role, nanobody and year checks, plus the amyloid count.
* `entry_metrics_verify.csv`, `entry_metrics_verify2.csv`: per-entry metrics.
* Rows `gap_V_errchains_m10_*` in `tests_verify.csv` have no controls and are descriptive only. Ignore their p and q
  values.
