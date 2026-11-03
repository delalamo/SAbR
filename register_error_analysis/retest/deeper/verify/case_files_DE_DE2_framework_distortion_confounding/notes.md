# Verification of DE2 (framework distortion in D/E error chains): confounding lens

**Verdict: weakened.**
* **The association reproduces exactly** and survives every confounder adjustment I tried.
* **It is not a property of the erroneous chain.** Correctly numbered V chains in the same entries carry about two
  thirds of the excess. Within an entry, the error chain is not significantly more distorted than its correct partner.
* **It is not specific to D/E.** Every other first-occurrence error mode shows the same framework distortion.
* **What it marks:** a poorly built Fab or entry model, not a feature of D/E slips.

## Reproduction (`s00_reproduce.py`)

Same sample and code path as `case_files_DE/s07`: confirmed EM D/E units, one chain per unit.

* **Errors vs matched clean EM units:** 49 vs 271, framework RMSD 1.560 vs 1.015 Å (medians 1.45 vs 0.97).
  OR 3.18 per SD, p = 6e-8.
* **Within paper:** OR 10.05 per SD, p = 2e-4.
  * The conditional logit uses only 25 of the 49 errors, from 21 papers. Their mean is 1.48 Å vs 1.09; the 1.56 quoted
    in the claim mixes in the other 24 errors.
  * D− by mode has q = 0.006, not ≤ 0.005.

The recomputed RMSD (`s01_features.py`) matches the original `fw_rmsd` to 1e-7.

## Confounders and alternative metrics: the association survives (`s02_stats.py`, `tests_all.csv`)

**Alternative definitions of the metric**, all with OR 2.4–3.7 per SD matched (q < 1e-4) and 3.5–13 within paper:
* core framework: excluding IMGT 4–10 and 96–104, and ±3 around any SAbR/ANARCI disagreement, including 1–2 residue
  ones;
* residues more than 15 Å from the 118 slot ("away from the slip");
* residues within 15 Å of it;
* median deviation;
* fraction of residues more than 2 Å off;
* the minimum over a panel of 195 high-resolution (≤ 1.9 Å) X-ray references;
* the best sequence-matched reference (germline proxy);
* FR1, FR2, FR3 and the CDR3 base separately.

**Adjustments** (matched model, OR 2.9–3.6 per SD for each): framework EM Q-score, lq_z, unbuilt V residues,
resolution², continuous deposition date, SARS-CoV-2, mouse, V identity, atypical CDR length, CDR3 length,
starting-model type, fitting protocol, entries per paper, and number of V chains in the entry. With all of them
together, OR 3.2.

**Within paper:**
* adjusted for chain type: OR 9.6;
* adjusted for chain type, resolution and number of V chains: OR 13.2 (95% CI 3.3–53);
* adjusted for Q-score: OR 7.6.

**Clustering:**
* clustering by lab (last author) instead of by paper: unchanged;
* one chain per paper over 300 resamples: median OR 3.4, p < 0.05 in 100% of resamples;
* leave-one-paper-out within paper: OR range 5.8–15.9;
* dropping the 5 most distorted papers: OR 2.4;
* truncating values ≥ 2.5 Å: OR 2.5.

**Strata** (matched): significant for ≤2020, 2021 and ≥2022, for SARS-CoV-2 and other antigens, for resolution
< 3.5 Å and ≥ 3.5 Å, for H, κ/λ, human and mouse, and for good and poor Q-score. Not significant for VHH
(n = 9; OR 1.6, p = 0.13).

**Local map quality.** The error chains do have a lower framework Q-score (0.275 vs 0.376, p < 1e-4), but this
explains only about 10% of the OR.

**The "confirmed" selection does not bias it.** Confirmed and not-confirmed errors have the same RMSD (1.56 vs 1.62,
p = 0.76).

## What weakens it (`s03_extra.py`, `s04_extra_stats.py`, `s05_entry_level.py`)

1. **Shared with correct chains of the same entry.**
   * In a three-group model against matched clean units (adjusted for year bin, resolution and chain type, clustered
     by paper):
     * clean V chains in D/E error entries: +0.35 Å (CI 0.17 to 0.54, n = 66, p = 3e-4);
     * the error chains themselves: +0.53 Å.
   * Entry fixed effects, error chain vs correct chain of the same entry (26 entries), adjusted for chain type:

     | Metric | Error chain minus correct chain | p |
     |---|---|---|
     | Framework RMSD (original) | +0.17 Å (CI −0.09 to 0.43) | 0.20 |
     | Sequence-matched reference | +0.13 Å | 0.31 |
     | Core framework | +0.09 Å | 0.45 |

   * Paired by entry, the error chain is higher in 14 of 26 entries.
   * So the signal is mostly an entry-level (map or Fab-model) property.
2. **Not specific to D/E.**
   * Other first-occurrence EM error modes, vs matched controls (OR per SD):

     | Mode | n | OR |
     |---|---|---|
     | All other modes | 114 | 3.1 |
     | A | 52 | 2.4 |
     | B | 23 | 2.5 |
     | C | 20 | 3.0 |
     | F | 7 | 2.6 |

   * D/E vs the other modes directly: OR 1.18 (p = 0.17).
3. **The extremes come from multiply mis-built chains.**
   * 13 of 49 error chains also carry a non-D/E run (A, B, C or F); they average 2.17 Å.
   * The 26 isolated D/E chains (no other run in the chain, and no error in a partner chain with a different sequence)
     average 1.31 vs 1.02 Å.
     * Matched: OR 2.1, p = 7e-4.
     * Within paper: OR 6.8, p = 0.002, 15 papers.
   * So the effect size roughly halves (+0.30 Å instead of +0.55 Å). It is still significant.
   * The crude `s04` "isolated" row (n = 12, p = 0.06) wrongly counted symmetry copies as partners and is superseded.
4. **The independent "confirmation" is unreliable exactly where frameworks are distorted.** Off-slot FR4 in clean
   controls occurs in 14% of chains with RMSD > 1.4 Å, vs 1.2% of the rest (Fisher OR 13). This does not create DE2,
   given point 3 of the previous section, but "confirmed" carries less weight for distorted chains.
5. **The same antibody modelled elsewhere** (identical V sequence, clean model): only 5 units. The error chain is higher
   in 3 of 5 (p = 0.31). Inconclusive.

## Nulls

* **Within entry:** error chain vs correct partner chain (all three metrics).
* **Specificity:** D/E vs other error modes.
* **VHH stratum.**
* **Selection:** confirmed vs not-confirmed RMSD.
* **Error vs matched:**
  * resolution (3.60 vs 3.54 Å, p = 0.33);
  * unbuilt V residues (p = 0.06).
* **Same-sequence comparison** (n = 5).

## Files

| Script | Purpose |
|---|---|
| `s00_reproduce.py` | Reproduction |
| `s01_features.py` | Recomputed metrics and covariates → `features_recomputed.csv` |
| `s02_stats.py` | Metric, adjustment and strata tests → `tests_all.csv` |
| `s03_extra.py` | Partner, same-sequence and other-mode chains → `features_extra.csv`, `extra_sample.csv` |
| `s04_extra_stats.py` | → `tests_extra.csv`, `paired_error_vs_clean_partner.csv`, `same_antibody_other_models.csv` |
| `s05_entry_level.py` | → `tests_entry_level.csv` |
| `s06_within_paper_ct.py` | → `tests_within_paper_ct.csv` |
| `s07_combine.py` | → `all_tests_BH.csv` (BH across 190 p-values) |
