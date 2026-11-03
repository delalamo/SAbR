# Verification (confounding lens): "new team" and first-occurrence register errors

**Finding checked** (people_network, `new_team_ab_cryoem`): cryo-EM studies where no audit author had an earlier
antibody cryo-EM PDB entry have about twice the odds of a first-occurrence error.
Reported: 47.2% vs 33.2%; OR 1.99 (1.36–2.91); p = 0.0004; 127 error and 1,175 clean studies.

**Verdict: weakened.**
* **Confounding does not explain it.** The study-level association reproduces exactly. It survives every confounder
  adjustment tried, lab clustering and a within-lab analysis.
* **It is not a general "twice as likely".**
  * It is confined to studies deposited up to 2023 and to studies without a nanobody.
  * It is borderline when each lab is weighted equally.
  * It shrinks when copied runs count as errors.
  * The China-led part depends on how name collisions are handled.
* **Grade: moderate.** It is real, but specific to period and antibody format.

## Reproduction

* 60/450 (13.3%) vs 67/852 (7.9%).
* Base model: OR 1.997 (1.37–2.92), p = 0.00035. The data come from the investigator's cache, `study_experience.csv.gz`.

## Survives (OR 1.78–2.18, all p ≤ 0.015; `tables/v02_adjusted_models.csv`, `v02_mantel_haenszel.csv`, `v03_*`, `v04_*`)

**Clustering and model structure**

| Check | OR (95% CI) | p |
|---|---|---|
| SEs clustered by last-author lab | — | 0.0005 |
| GEE, exchangeable by lab | 1.99 | — |
| Antibody-in-paper unit level, clustered by study (4,079 units, 171 errors) | 1.98 (1.31–2.99) | — |

**Time and resolution**

| Check | OR (95% CI) | p |
|---|---|---|
| Each deposit year | 2.05 | — |
| Half-year fixed effects | 2.18 | — |
| Spline of deposit date | 1.98 | — |
| Resolution spline | 1.98 | — |
| Resolution × year | 2.05 | — |
| Mantel–Haenszel, year × resolution | 1.83 (1.26–2.66) | — |
| Mantel–Haenszel, year × SARS-CoV-2 × resolution | 1.79 | — |

**Single adjustments**

| Added covariate(s) | OR |
|---|---|
| Chain-type composition | 2.06 |
| SARS-CoV-2 | 2.08 |
| Number of entries | 1.96 |
| Mouse, nanobody, atypical CDR1/2, germline identity | 2.06 |
| Antibody never solved before | 1.90 |
| Role | 1.97 |
| Antigen class, first-of-family, novel target | 1.91 |
| Local map fit (lq_z, Q) | 1.99 |
| Unbuilt-residue fraction | 2.02 |
| Starting model and fitting protocol | 1.92 |
| Author count | 1.96 |
| Country | 2.10 |
| Journal tier | 1.98 |
| Last-author antibody experience | 1.81 |

**Combined adjustment and within-lab**

| Check | OR (95% CI) | p |
|---|---|---|
| Kitchen-sink model with all of the above | 1.91 (1.22–2.97) | 0.004 |
| Inverse-propensity weighting (propensity AUC 0.77) | 1.78 (1.12–2.84) | 0.015 |
| Within-lab conditional logit, adjusted for deposit date (31 informative labs, 15 with both new and experienced studies) | 4.2 (1.2–14.5) | 0.024 |

**Leave-one-year-category-out:** OR 1.70–2.45, all p < 0.014.

**Simpler factors do not explain it.** New teams leave *fewer* residues unbuilt (max unobserved fraction 0.19 vs 0.24).
Their local fit is only slightly worse (lq_z 0.08 vs 0.17). Adjusting for either leaves the OR unchanged.

## Weakens

1. **Era** (the split was chosen after the investigator saw the per-year table).
   * Up to 2023: OR 3.06 (17.6% vs 7.4%).
   * From 2024: OR 0.79 (6.4% vs 8.6%).
   * Interaction p = 0.001.
   * Per-year crude ORs:

     | Year | OR |
     |---|---|
     | ≤2019 | 1.6 |
     | 2020 | 1.4 |
     | 2021 | 2.8 |
     | 2022 | 4.8 |
     | 2023 | 3.3 |
     | 2024 | 0.6 |
     | ≥2025 | 0.8 |

   * Full adjustment gives 2.63 up to 2023 and 1.25 from 2024.
2. **Nanobody studies show no effect. The investigator did not report this.**

   | Studies | OR | Error rate, new vs experienced |
   |---|---|---|
   | Any nanobody | 0.82 | 9.0% vs 9.2% |
   | No nanobody | 2.58 | 15.2% vs 7.3% |

   * Interaction p = 0.012.
   * Studies up to 2023 with no nanobody: OR 3.8 crude, 2.70 fully adjusted.
3. **Lab balancing over the whole period.**
   * One study per lab, 500 draws: median OR 1.58 (draw range 1.19–2.09), median p = 0.11; only 22% of draws reach
     p < 0.05.
   * Weights of 1/(the lab's study count): OR 1.57, p = 0.07.
   * Labs with a single study in the set: 11.2% vs 9.1%.
   * Up to 2023 the effect holds under lab balancing: OR 2.54, 96% of draws significant.
4. **Overlap with the error definition.** Experienced teams account for 28 of the 33 copied- or artifact-only
   studies, which are dropped from the main comparison. Counting them as errors gives OR 1.61 (1.11–2.33),
   p = 0.011.
5. **Name rule in China.**
   * The `_na` flag ignores collision-prone surnames. As a result:
     * 41% of new-team studies are China-led, vs 14% of experienced-team studies;
     * 61% of new-team studies have a collision-prone last-author name, vs 16%.
   * Among China-led studies: OR 6.1 with the `_na` flag, but 1.10 with the all-name-key flag.
   * Outside China both flags agree: 1.69 and 2.36.

## Nulls (no difference or no attenuation)

* None of the covariates above changes the OR materially.
* No interaction with:
  * SARS-CoV-2 (p = 0.24);
  * antibody novelty (p = 0.90);
  * mouse (p = 0.40);
  * a collision-prone last-author name (p = 0.67).

## Files

* `v01_build_frame.py`: study frame with covariates. Output to the scratchpad: `cache/verify_people_newteam_conf/`.
* `v02_models.py` → `tables/v02_adjusted_models.csv`, `v02_mantel_haenszel.csv`, `v02_balance_new_vs_experienced.csv`.
* `v03_heterogeneity.py` → `tables/v03_heterogeneity.csv`, `v03_leave_one_year_out.csv`,
  `v03_lab_resampling_withinlab_ipw.csv`.
* `v04_lab_structure_units.py` → `tables/v04_lab_structure_units.csv`.
* `v05_subgroups_bh.py` → `tables/v05_subgroups.csv`, `v05_all_tests_bh.csv` (Benjamini–Hochberg over 108 tests).

## Caveats

* The exposure is the investigator's name-based flag. It is not re-derived from ORCID, apart from the 33 excluded
  studies.
* The era and nanobody splits are post hoc.
* The within-lab analysis rests on 15 discordant labs and 19 new-team studies.
* The lab key is the loose last-author key, so labs with common names merge.
