# Verification: struct_unusual / beyond_site_backbone (data integrity and statistics)

**Verdict: confirmed, with qualifications.** The Ramachandran-'allowed' and clash excess outside the error site is
real. It reproduces exactly and survives every attack tried here. Three parts are overstated:
* the effect size per independent paper;
* the rotamer component;
* the attribution to the error chain itself, as opposed to the entry's antibody model.

## Reproduction (`v02`, `v03`, `v04`)
I recomputed everything independently, from the wwPDB validation XML and the SAbR/ANARCI scan records, without the
investigator's `residues.pkl` or `unit_metrics.csv`.

* **Per-unit values match exactly.** Ramachandran 'allowed' and rotamer outliers: max |diff| 0. Clash: ≤1 residue in
  a few chains, from first vs last altloc.
* **Adjusted model:** +3.75 per 100 (2.32–5.19), p = 3.0e-7, n = 172 / 356.
* **Other metrics:**
  * clash: +6.25, p = 3.4e-4;
  * rotamer: +0.67, p = 0.0076;
  * frameworks: 8.34 vs 4.37 (+3.96, p < 1e-5, clustered by lab).
* **Sample checks:**
  * cases = all 183 cryo-EM first-occurrence units;
  * controls = units with no disagreement run, in entries with no run;
  * no entry appears as both case and control.
* **Exclusions:** 11 cases drop out because their run covers the whole V domain. This is a legitimate exclusion.

## Robustness: all survive (`v_stats.csv`, `v_spatial_tests.csv`, `v_software_region.csv`)

| Check | Effect (per 100) | p |
|---|---|---|
| Cluster by paper | 3.75 | 2e-6 |
| Cluster by last author | 3.75 | 8e-6 |
| Poisson rate ratio | 1.60 (1.32–1.93) | |
| Median regression | +2.17 | |
| Entry-level permutation within year bins | | ≤ 5e-4 |
| Leave-one-paper-out / leave-one-lab-out (min effect, max p) | 3.39 | 3e-6 |
| Drop the top-10 case labs | 3.27 | 1e-4 |
| Lab fixed effects (22 labs with both roles, 51 cases) | 3.25 | 0.047 |
| + recorded software, starting model and fitting | 3.83 | |
| Exclude every disagreement residue ±3 | 3.93 | |
| Exclude ±10 | 3.87 | |
| Spatial exclusion, CA > 10 Å from the site | 4.04 | 5e-5 |
| Spatial exclusion, > 14 Å | 3.63 | 0.001 |
| Exclude residues near gaps or termini | 4.13 | |
| Non-cached controls only (the picker preferred cached entries) | 3.95 | |
| + log(number of V chains in the entry) | 3.52 | |

## Circularity check (`v09_sabr_noise.py`)
Could non-ideal geometry make SAbR emit spurious runs? Test: Gaussian coordinate noise added to 40 clean control
chains.

| Noise σ (Å) | Estimated non-favoured (%) | SAbR numbering changed |
|---|---|---|
| 0 | 5 | — |
| 0.15 | 18 | 0 of 80 runs |
| 0.30 | 42 | 0 of 80 runs |
| 0.50 | 63 | 6% of runs |

Case chains average about 10% non-favoured. Detector-induced circularity is therefore implausible.

## Qualifications
1. **Multi-unit papers inflate the effect.**
   * One case per paper: +2.64 (1.22–4.06), p = 3e-4.
   * Paper-level means: +2.67, p = 5e-5.
   * The headline +3.75 counts 10 units from one paper (Zhu P., Cell Rep 2025) and similar clusters.
2. **The effect is concentrated in multi-error Fvs.**
   * 61 case chains in entries with no error-free V chain: +7.3. This includes 21 Fvs where both chains are cases,
     at about 16% non-favoured.
   * The 111 case chains whose entry also has an error-free V chain: +2.36.
3. **Not shown to be specific to the error chain.**
   * Case chain vs error-free V chains in the same entry (106 entries, entry fixed effects):
     * Ramachandran 'allowed': +0.84 (−0.44 to 2.12), p = 0.20;
     * clash: +1.31, p = 0.07;
     * rotamer: −0.02, p = 0.92.
   * Identical copies of the case antibody in the same entry: 6.6 vs 6.8.
   * Error-free V chains in case entries vs controls: +0.73 (n.s.), clash +3.6 (p ≈ 0.05).
   * This fits an entry-level, antibody-model building practice, as the investigator's interpretation says.
4. **The rotamer component is fragile.**
   * p = 0.016 clustered by lab.
   * p = 0.05–0.08 after spatial exclusion.
   * The investigator's matched Wilcoxon gives p = 0.053, and it is null within entries.
   * The detail line "every sensitivity model gives p ≤ 0.001" is not true for clashes (2020–22 only: p = 0.0044) or
     rotamers.
5. **Reporting inconsistency in notes.md.** The partner row "10.5 vs 5.9" uses all partners, including 43 that are
   themselves case chains (mean 16.7). The adjusted +1.2 was fitted on the 85 error-free partners, whose raw mean is
   7.36.

## Files
| Purpose | Scripts | Tables |
|---|---|---|
| Sample check, independent recompute | `v01`–`v03` | `v_unit_metrics.csv` |
| Statistics and robustness | `v04` | `v_stats.csv` |
| Spatial and gap exclusion | `v05` | `v_spatial_tests.csv` |
| Within-entry contrast and control selection | `v06`, `v07`, `v08` | `v_within_entry_pairs.csv`, `v_entry_level_*.csv`, `v_decompose.csv` |
| SAbR noise test | `v09` | `v_sabr_noise*.csv` |
| Software adjustment and region split | `v10` | `v_software_region.csv` |

BH q over all 98 verification tests: `v_all_tests_bh.csv`.
