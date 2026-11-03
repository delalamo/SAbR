# Verification (confounding lens): struct_unusual "antibody_vs_rest_of_model"

**Claim checked:** within an entry, the antibody V domains of error entries have worse geometry than the rest of the
model (V minus non-V Ramachandran 'allowed' +3.5 pp; clash gap +3.6 pp). In controls they do not. The non-antibody
chains are as good as in controls, and the V-minus-rest Q-score gap does not differ.

**Verdict: WEAKENED.** The core association survives every adjustment. The effect size is overstated by about 35%,
and two supporting statements do not replicate.

## What I did
1. **Rebuilt the metric for every cryo-EM entry** (`v01_fetch_vrpt.py`, `v02_parse.py`).
   - Validation XML for all 2,831 EM entries in `entries.csv`; parsed 2,813.
   - Residues are split into: V (any residue in a SAbR record), C (non-V residues on antibody chains, i.e. constant
     domains) and O (other chains).
   - V residues are binned by distance to the nearest SAbR/ANARCI disagreement.
   - The parser reproduces `entry_vrpt.csv` exactly (difference < 1e-12 on 40 entries).
2. **Population test** (`v04_population.py`): 193 first-occurrence error entries vs 2,556 clean entries.
   - Unit = entry, with SEs clustered by paper.
   - Covariates: year bin, resolution, deposition date, SARS-CoV-2, composition (VHH, light-chain, auxiliary-role and
     V fractions, rest size, constant domains present), local map fit (V-rest Q gap, V Q, atom inclusion) and unbuilt
     internal V residues.
3. **Re-analysis of the investigator's own 344 entries** (`v03_sample_reanalysis.py`, `v05_sample_restfix.py`):
   - clustering by paper and by last author;
   - one entry per paper;
   - strata;
   - removing residues near disagreements.
4. **Within-paper paper fixed effects, last-author clustering and a date spline** (`v06_within_paper.py`).
   BH q over all 190 tests is in `tables/all_tests_bh.csv`.

## Results
| Check | Error | Control | Effect (95% CI) | p / q |
|---|---|---|---|---|
| As published (entry clusters) | 3.65 | 0.21 | +3.46 (2.49–4.42) | 3e-12 |
| **Bug fix:** drop 10 case entries with no non-V residues (`rest_*` was set to 0, so their gap = V value; 0 controls affected) | 3.25 | 0.21 | +3.08 (1.95–4.20), paper clusters | <1e-6 |
| **Population**, all EM, paper clusters | 2.91 (n 181) | 0.72 (n 2,552) | **+2.16 (1.13–3.19)** | 4e-5 / q 2e-4 |
| … + all covariates | | | +2.21 (1.26–3.16) | q 1e-4 |
| … unit = paper | | | +2.17 | <1e-5 |
| … last-author clusters / date spline | | | +2.16 / +2.21 | 1e-4 |
| … 2020-22 / 2023+ / ≤2019 | | | +2.10 / +1.95 / +3.85 (n 13) | 0.009 / 0.007 / 0.07 |
| … non-SARS-CoV-2 / SARS-CoV-2 | | | +1.92 / +2.81 | 5e-4 / 0.02 |
| … ≤3.5 Å / >3.5 Å | | | +2.10 / +2.22 | 0.001 / 0.001 |
| **Within paper** (paper fixed effects; 55 papers, 78 error vs 150 clean entries) | 4.02 | 1.72 | **+2.33 (1.17–3.50)** | 1e-4 |
| V residues within 10 of any disagreement removed (definitional overlap), all covariates | | | +1.76 (0.75–2.76) | 6e-4 |
| Only V chains with no disagreeing residue, all covariates | 1.73 (n 125) | 0.72 | +1.13 (0.28–1.98) | 0.009 / q 0.015 |
| Clash gap, population | 4.94 | 1.87 | +2.98 (1.06–4.90) | 0.002 |
| Clash gap, sample after the bug fix | 4.11 | 1.62 | +2.54 | 0.01 |
| **Other (non-antibody) chains, % clash**, population | 20.0 | 16.7 | **+2.77; adjusted +3.21** | 0.011; 0.003 (q 0.006) |
| Same, sample after the bug fix (`rest_clash`) | 19.7 | 16.7 | +2.67 | 0.03 |
| Same, within paper | | | −0.21 | 0.84 |
| Other chains, % Ramachandran 'allowed' | 5.98 | 5.06 | +0.68 | 0.036 (q 0.05) |
| **V minus rest Q gap**, population | −0.073 | −0.033 | −0.037; adjusted −0.044 | 0.002; 1e-5 |
| Same, within paper | | | −0.007 | 0.40 |
| Rama gap after adjusting for the Q gap, V Q and atom inclusion | | | +2.10 | 1e-4 |
| Investigator's controls vs the other 2,371 clean entries (selection check) | 0.24 | 0.76 | −0.55 | 0.018 |
| 34 later error entries of the same units (not in the sample) | 1.30 | 0.76 | +0.43 | 0.71 (n 32) |
| Constant-domain residues minus other chains | | | −0.03 | 0.97 |
| Entries with copied/artifact-only runs vs clean | −0.38 | 0.72 | −1.14 | 3e-4 |
| Clean sibling entries of error papers vs other clean | 1.72 | 0.66 | +1.04 | 0.066 |

## Interpretation
### Survives
None of these confounders explains the association:
* year (bin or spline), resolution, SARS-CoV-2, method (all EM);
* chain composition, including the auxiliary-reagent role;
* paper or lab clustering, and paper fixed effects;
* unbuilt V residues and local map fit.

It is also present in V residues far from the slip, and in error-free V chains.

### Weakened
1. **Size.** The population effect is about 2.2 pp, not 3.5. The excess comes from three sources:
   * a bug: 10 case entries with an empty rest-of-model were scored with rest = 0;
   * control selection: the matched controls have unusually small gaps, −0.55 pp against other clean entries;
   * case selection: only the first entry per unit was used, and later entries of the same unit show a smaller gap.
2. **Definitional overlap.** About 20% of the gap comes from residues at or near the disagreement, i.e. the slip
   itself (2.16 → 1.73 pp). In error-free V chains the gap is about half (1.05–1.13 pp).
3. **"Rest of the model is as good as in controls" is not true for clashes.** Non-antibody chains carry +2.8–3.2 pp
   more clash residues in the population, and +2.7 pp in the investigator's own sample once the bug is fixed. This
   excess vanishes within paper (−0.2), so it is a paper/lab-level property, not an entry-level one.
4. **"V-minus-rest Q gap does not differ" is not true in the population** (−0.04, q < 0.005). Adjusting for it leaves
   the Ramachandran gap almost unchanged (2.16 → 2.10), so the geometry excess is still not explained by density.

## Caveats
* 18 EM entries lack a validation report or parse (6 returned 403).
* Entries with no non-V residue are excluded from gap tests.
* Q-score is a crude proxy for local resolution; no local-resolution maps were used.
* SAbR numbering is structure-based, so poor V geometry could in principle feed the error detection itself. The
  error-free-chain result limits but does not exclude this.
* Constant domains (C) are present in only about 700 entries.
