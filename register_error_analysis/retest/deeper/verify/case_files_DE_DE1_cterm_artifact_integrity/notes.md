# Verification of DE1_cterm_artifact (case_files_DE), from a data-integrity and statistics angle

**Verdict: confirmed.** The direct experiment below makes the finding stronger than claimed. Some details need correcting.

## What I checked

1. **Investigator's code.**
   * The script that turns slot shifts into verdicts (`err_struct.pkl`, `independent_check_DE_units.csv`) is not in
     `scripts/`.
   * I inferred the rule: a run is confirmed if at least half of its positions in IMGT 116–124 shift in the direction
     of the SAbR–ANARCI delta.
   * Applied to their slot shifts, this rule reproduces **80/80** verdicts (`v04`).
   * No unit had different verdicts on different chains (174 chains checked).
2. **Fisher test.** It reproduces exactly: D+ models ending at ≤121 have 10/12 not confirmed vs 1/28 for longer models,
   OR 135, p = 8.0e-7.
   * Leave-one-paper-out (37 papers): the largest p is 7.8e-6.
   * One unit per paper: 9/11 vs 1/26, p = 4.1e-6.
3. **Independent geometry** (`v02`), written from scratch:
   * coordinates from the PDBe ModelServer;
   * three different references per chain type (1K4C, 1YY8, 4M48, and λ 5NH3/5NGV/4TOY), none of them 5DT1 or 1N8Z;
   * a global and a local superposition core;
   * an "indeterminate" class when residues lie more than 2.5 Å from any reference slot.

   Results:
   * Of the investigator's 20 "not confirmed" runs, 14 stay not confirmed, 5 become indeterminate and 1 is confirmed.
   * Of their 56 confirmed runs, 48 stay confirmed, 5 become indeterminate and 3 are not confirmed.
   * For the seven core cases (7T3D, 8GRR, 7K22, 8TQI-C, 8TQI-I, 8PQ2, 9DN2-H), every reference gives shift 0 at
     116–118. The Cα of the sequence-W/F118 lies 0.5–1.06 Å from reference slot 118.
   * D+, determinate verdicts only: models ending at ≤119 are 7/7 not confirmed vs 3/30 for longer models
     (p = 1.2e-5, q = 3e-5).
4. **Direct test of the mechanism** (`v01`). The rerun copies the scan's SAbR pipeline and reproduces the scan calls for
   all 12 short D+ chains.
   * **Clean chains cut after ANARCI 118 or 119** (75 chains: 25 H, 25 κ, 25 λ): SAbR reports a D+-like run in
     **75/75** (95% CI 0.95–1.00), with or without the unobserved-residue metadata. Untruncated chains: 0/75.
   * **The same chains cut at other positions:** after 117, 67/75; after 120, 3/75; after 121, 1/75; after 122 or
     later, 0/75.
   * **Confirmed long D+ models cut after 118:** 27/27 still flagged. At a chain end SAbR therefore cannot tell a real
     slip from a correct model.
   * **ImmuneBuilder models of the same 10 short-ended antibodies** (correct register by construction): the full model
     gives 0/10 runs. When cut to the deposited model's end, or given the deposited model's gaps, they reproduce the
     run in **8/10**: 7K22, 7T3D, 8PQ2, 8TQI-C, 8TQI-I and 9DN2 with the same +1, 7D3K +1 and 8GRR +3. The two without
     a run are 8W5Q and 9FTE, which end at 120–121.
5. **Population view** (`v03`, all scanned chains):
   * EM units whose model ends exactly at IMGT 118: 4/4 carry a first-occurrence D+ call. Ending at 119–121: 8/26.
     Ending at 122–124: 3/150. Ending at ≥125: 17/4,090.
   * No X-ray chain ends at 118–121.
   * All 25 D+ chains that end at ≤121 have the run ending at the last modelled residue.
   * The 8 units ending at 118–119 (21 chains) are 20% of all D+ first-occurrence units (25% of EM D+ units) and 3.7%
     of all 218 first-occurrence units.

## Corrections to the claim

* **The cut-off is ≤119, not ≤121.**
  * The artifact is complete at ≤119 and nearly absent at 120–121 (4% and 1%).
  * 9FTE and 30US (ending at 120–121) are not explained by it. In the predicted-model test, 9FTE cut at 120 gives no
    run; 30US was not tested.
  * Their geometry is indeterminate: residues lie 4–7 Å from any slot.
  * 7D3K (ending at 119) also has indeterminate geometry: its W118 lies 14.7 Å from slot 118. So 7 of the 10, not 10,
    are "in the canonical slot".
* **"SAbR numbers the last Trp 119" is true for 6 of the 8 short-ended chains.** For 8GRR SAbR numbers it 133, and for
  7D3K 129. Numbers past 128 are impossible and could be flagged automatically.
* **The quoted 1.8% clean-control off-slot rate covers IMGT 118–124 only.**
  * Over 116–124, the window actually used for D+ verdicts, it is 12.2% (7.3% in my recheck).
  * This does not affect the "not confirmed" counts, but some short-model "confirmed" calls may be noise.
* **Mode A side-claim reproduced:** 48/57 confirmed, and the 0.9% null when the same window and sign rule is applied
  to clean chains. Note that 30% of clean chains have some off-slot residue at IMGT 1–12, mostly at position 8.

## No difference / not supported

* SAbR runs on untruncated clean chains: 0/75.
* Clean chains truncated at ≥122: 0/75 at each cut-off.
* Truncation at 121: 1/75 (p = 1 vs untruncated).
* Truncation at 120: 3/75 (p = 0.24).
* Predicted models of 8W5Q and 9FTE cut at their deposited ends: no run.

## Files

* Scripts: `v01_sabr_truncation.py`, `v02_geometry_recheck.py`, `v03_population_chainend.py`, `v04_stats.py`.
* Tables in `tables/`: `v01_sabr_runs.csv`, `v02_geometry_units.csv`, `v02_geometry_controls.csv`,
  `v03_chainend_rates.csv`, `v03_short_chain_dplus.csv`, `v04_tests.csv`.
