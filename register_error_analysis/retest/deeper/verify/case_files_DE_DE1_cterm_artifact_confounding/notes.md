# Verification of DE1_cterm_artifact (case_files_DE), confounding lens

**Claim under test.** About a quarter of D/E runs are not reproduced by a framework-superposition test. In D+ this is
almost entirely a SAbR chain-end artifact in models that stop at W/F118: D+ models ending at IMGT <=121 are not
confirmed in 10/12, vs 1/28 longer models (Fisher p = 8e-7, OR 135).

**Verdict: confirmed.** The association survives every adjustment I tried. A direct experiment then shows the
mechanism is real and deterministic. Two corrections:

* The real cut-off is the model ending at **IMGT <=119** (<=120 for VHH), not <=121.
* One of the 10 cases (30US, which ends at 121) is not explained by the artifact.

## Scripts

* `v01_population_chainend.py`: across all 15,865 scanned chains, the D+ call rate by where the model ends.
* `v02_truncation_experiment.py`: a causal test. SAbR is rerun exactly as in the scan on correct models after
  computationally deleting every residue past IMGT T.
* `v03_confounding_stats.py`: reproduces the Fisher test, then stratifies and adjusts it, runs an equal-information
  check and tests simpler factors.

## Results

### 1. The association reproduces and survives confounding control (`tables/v03_confounding_tests.csv`)

Values are "not confirmed" counts, models ending <=121 vs ending later, unless marked otherwise.

| Analysis | Ends <=121 | Ends later | Effect | p | q (BH) |
|---|---|---|---|---|---|
| Reproduced, all D+ | 10/12 | 1/28 | OR 135 | 8.0e-7 | 1e-5 |
| EM only | 10/12 | 1/19 | | 1.5e-5 | |
| Non-SARS-CoV-2 only | 9/11 | 0/14 | | 2.7e-5 | |
| Heavy/VHH only | 8/10 | 0/20 | | 7.7e-6 | |
| EM, resolution <=3.4 Å | 6/7 | 0/9 | | 8.7e-4 | |
| EM, resolution >3.4 Å | 4/5 | 1/10 | | 0.017 | |
| One unit per paper | 9/11 | 1/26 | | 4e-6 | |
| One unit per last author, EM | 9/11 | 1/15 | | 1.6e-4 | |
| Exact stratified: year bin, method × SARS2, year × method, chain type × method | | | | 1e-4 to 2e-4 | |
| Firth logit adjusted for year + resolution + SARS2 | | | OR 67 (4.3–1045) | 0.0026 | |

* Year, resolution, SARS-CoV-2 status and method have no effect of their own in the Firth models (all p > 0.4).
* There is a covariate imbalance: 1/12 short models are SARS-CoV-2 vs 14/28 long ones, and all short models are EM.
  Neither explains the association.

### 2. Equal-information check: the result is not from fewer testable residues

The investigator's confirmation rule was re-applied to every unit using only IMGT 116–118. My re-implementation
reproduces 95% of their labels.

* Long D+ models are still confirmed in 24/28.
* Short models are not confirmed in 9/12.
* Fisher p = 3.9e-4 (one unit per paper: p = 1.4e-3).
* The superposition test therefore keeps its sensitivity at a chain end. It uses FR1–FR3 only, so truncation cannot
  change it.

### 3. Causal test: deleting FR4 from correct models is enough to create a D+ call (`tables/v02_truncation_summary.csv`)

**Sample.** 80 clean cryo-EM chains (30 H, 20 K, 15 L, 15 VHH), one per paper, from the investigator's matched-control
pool.

**Pipeline check.** The full-length rerun reproduces the stored scan numbering exactly (100% of residues) and gives
0/80 runs.

C-terminal D+ run after keeping residues up to IMGT T:

| T | 117 | 118 | 119 | 120 | 121 | 122 | 124 | full |
|---|---|---|---|---|---|---|---|---|
| C-terminal D+ run | 75/80 | **80/80** | **80/80** | 10/80 (8/15 VHH, 2/30 H, 0 K/L) | 0/80 | 0/80 | 0/80 | 0/80 |

* **Deltas:**
  * T = 118: mostly +1 (58/80), but up to +9.
  * T = 120 in VHH: +2 or +3.
  * This covers the odd large-delta cases: 7D3K +11, 8GRR +15, 9FTE (VHH, ends at 120) +6.
* **Not caused by unobserved-residue metadata.** SAbR's code ignores terminal unobserved residues, and deleted
  residues were not declared as unobserved. The chain end alone triggers it.
* **Whole scan** (`tables/v01_rates.csv`): every chain whose model ends at 118 (15/15, 4 units) or 119 (8/8) has a
  C-terminal D+ run. All 15 ending at 118 are counted as first-occurrence D+ errors. Only 1 of the 271 clean controls
  ends at <=121: a selection effect of the same artifact.

### 4. Simpler factors

* **Unbuilt FR4 residues and local map.**
  * Within EM D+, "ends <=121" is the same variable as "≥5 unbuilt residues in the 10 after 118", so they cannot be
    separated statistically.
  * The truncation experiment shows the chain end, not missing density, is the cause.
  * Poor local map does **not** explain non-confirmation. The opposite holds: non-confirmed EM D+ runs have *better*
    Q-score at 116–118 (median 0.43 vs 0.12, p = 0.002; relative to the chain median p = 0.02). This independently
    suggests those residues sit correctly in density.
* **D− and E.**
  * Non-confirmation is associated with unbuilt FR4 residues after 118: 7/14 vs 2/19, p = 0.019.
  * It is not associated with resolution or Q-score.
  * Year is mildly associated (median 2024 vs 2021, p = 0.04). n is small, and this is only partly the same
    artifact.

### Case-level reconciliation of the 10 unconfirmed short D+

* **Ends at 118/119, artifact rate 100% in clean truncations (8 units, 7 papers):** 7K22-L1, 7T3D-H, 8TQI-C, 8TQI-I,
  8PQ2-L, 9DN2-H, 7D3K-H and 8GRR-H.
* **9FTE-B:** a VHH ending at 120, where the artifact rate is 8/15.
* **30US-D:** a VHH ending at 121, where the artifact rate is 0/80. Shifts at 116–118 are (+1, 0, +1), and 120–121
  are −1. **Not explained by the artifact**, and possibly a real slip.
* **The two confirmed short models,** 8KHR (H, ends at 120) and 8W5Q (ends at 121), lie outside the artifact zone.

## Nulls

* No confounding by deposition year, method, resolution, SARS-CoV-2 status, chain type, paper or last-author
  clustering.
* Resolution does not differ between confirmed and unconfirmed D/E runs (p = 0.6–0.8).
* The superposition test is not less sensitive at a chain end.
* Terminal unobserved-residue metadata is not the trigger.

## Corrections to the investigator's text

* **"SAbR numbers the last Trp 119"** describes only 6 of the 10 cases. 7D3K, 8GRR and 9FTE have shifts of +6 to +15,
  which the truncation experiment also produces. 30US is not artifact-explained.
* **The cut-off** should be "ends at <=119 (<=120 for VHH)".
* **The "1.8% clean false-positive rate"** excludes IMGT 116. Including 116, 10% of clean chains have a +1/−1 slot
  at 116, so "any FR4 off-slot" is 12% (33/271). Under the majority rule actually used, the clean false-positive rate
  is 0%.
* **Mode A 48/57 (84%)** reproduces. The ~1% null holds only under the majority rule: about 0.4–1.7% depending on
  the run window, vs 30% for "any off-slot residue at 1–12".
