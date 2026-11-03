# Verification of PM2 ("error papers are cited less"): confounding

**Verdict: weakened.** The association reproduces exactly. It survives every covariate adjustment tried. It fails three tests:
* lab-clustered inference;
* a within-lab comparison;
* restriction to papers that build more than one Fab.

## Set and model

* **Papers:** published cryo-EM papers (EM entries deposited from 2018 onward), error vs clean: n = 1,099 (109 error).
* **Model:** the investigator's base logit, `err ~ log1p(cites/yr) + C(period) + log2(n units) + resolution`.
* **Reproduction:** OR 0.683 (0.54–0.86), p = 0.0013. This matches the claim.

## What weakens it

1. **Lab clustering.**
   * With cluster-robust SEs by last author: OR 0.683 (0.50–0.94), p = 0.018.
   * GEE by lab: 0.69, p = 0.017.
   * Re-running the investigator's 104-test BH with this p gives **q ≈ 0.20 (was 0.034)**.
   * One random paper per lab (200 draws): median OR 0.62, median p = 0.005, and 95% of draws have p < 0.05.
2. **The effect is between labs, not within labs.**
   * Mundlak split: between-lab component OR 0.61 (p = 0.003); within-lab component OR 0.81 (0.53–1.24), p = 0.33.
   * Conditional logit within lab (42 labs with both outcomes, 59 error papers): OR 0.88 (0.58–1.32), p = 0.53.
   * Without size adjustment, error papers rank *higher* in citations than their own lab's clean papers (mean within-lab percentile 0.63 vs 0.56).
   * Lab output and seniority (prior PDB entries as last author) do not absorb the effect (OR 0.69).
   * The low-citation signal therefore marks *labs* whose papers are cited less, not error papers within a lab. The within-lab CI is wide, though, and does not exclude 0.68.
3. **The crude association is null.**
   * Mann–Whitney p = 0.08; crude logit 0.85, p = 0.12.
   * Most of the adjustment comes from period: period alone gives 0.73. This is a Simpson pattern, because 2020–21 had both the highest error rate and the highest citation rates.
   * Adjusting for period is legitimate. Size adds the rest.
4. **The effect is concentrated in small papers.**
   * Papers with ≤ 2 V regions (a single Fab or VHH): OR 0.59, p = 0.003.
   * Papers with > 2 V regions: 0.80, p = 0.27.
   * Stratified Mann–Whitney is significant only at n_units = 2. The interaction is not significant (p = 0.15).
5. **Normalising citations within journal × publication year weakens it:** OR 0.47 for a percentile change from 0 to 1 (0.21–1.05), p = 0.066, n = 836. Journal fixed effects and conditional logit within journal do not weaken it (0.57 and 0.65).

## What does NOT explain it (null confounders; OR stays 0.57–0.73)

* **Time:**
  * publication-year fixed effects 0.69;
  * deposition-year fixed effects 0.66;
  * publication-date spline 0.67;
  * raw log cites + log age 0.67;
  * papers at least 1 or 2 years old 0.65.
* **Size specification:**
  * categories 0.69;
  * quadratic 0.68;
  * cloglog with offset log n 0.70;
  * quasi-binomial per-antibody rate 0.67;
  * plus EM and PDB entry counts 0.70;
  * plus authors 0.72.
* **A size-only null simulation** (binomial or beta-binomial errors that depend only on size, period and resolution) gives null OR 1.00 (0.79–1.23). The observed 0.68 lies below all 500 draws, so the result is not an artefact of mis-modelling size.
* **Antigen and paper type:**
  * SARS-CoV-2 papers 0.58; non-SARS-CoV-2 0.72;
  * viral non-SARS2 0.61; non-viral 0.74 (n.s.);
  * full covariates 0.61; plus nanobody and mouse 0.59.
* **Competition, first-in-PDB and authors:** 0.63–0.68.
* **Local map and modelling:** paper median or minimum lq_z, AlphaFold start, rigid-body fit, atypical L3, minimum germline identity and nanobody fraction: 0.68–0.69. Citations do not correlate with map fit (ρ = −0.006).
* **Error definition:**
  * copied-only papers as controls 0.71;
  * any run as the outcome 0.65;
  * amyloid papers re-included 0.73;
  * CDR3/FR4 modes 0.60;
  * FR1 modes 0.69 (p = 0.06);
  * CDR1/CDR2/FR3 modes 0.64;
  * single-error-antibody papers 0.73.
* **Influence:** leave-one-error-lab-out gives 0.63–0.72 (max p = 0.006); dropping the 11 labs with ≥ 2 error papers gives 0.64; trimming the citation tails gives 0.66.
* **Correction bias:**
  * Highly cited papers are more often coordinate-replaced (OR 1.75 per log unit, p = 5e-5), but replacement is not linked to error status (0.73, n.s.).
  * Among never-replaced papers the citation OR is 0.65 (p = 0.012).
  * Only 4 of 2,344 entries supersede an older entry.
* **Citation source:** Europe PMC only 0.73 (p = 0.04); Crossref only 0.69 (p = 0.02).
* **X-ray replication:** 0.58 (p = 0.08, 16 error papers). EM and X-ray pooled: 0.68 (p = 0.009).

## Files

* `v01_confounding.py` → `v01_tests.csv`: reproduction plus the main battery.
* `v02_size_null_sim.py` → `v02_size_null_sim.csv`.
* `v03_lab_covariates.py` → `v03_lab_covariates.csv`, `v03_lab_table.csv.gz`.
* `v04_correction_bias.py` → `v04_correction_bias.csv`.
* `v05_superseded.py` → `v05_superseded.csv`.
* `v06_heterogeneity.py` → `v06_heterogeneity.csv`, `all_verification_tests_bh.csv`. This BH is across robustness variants, which are not independent tests.
