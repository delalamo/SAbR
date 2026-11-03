# Verification of DE3_VH_VL_cooccurrence (case_files_DE): data integrity and statistics

Scripts: `v01_reproduce.py` (independent rebuild from chains.csv.gz + runs_first_occurrence.csv + err_struct.pkl),
`v02_robust.py` (LOPO, unit/paper collapse, within-paper MH, GEE, permutation, outcome definitions, X-ray),
`v03_strata.py` (SARS-CoV-2, 2020-21, single-Fab strata). Tables: `entry_table.csv`, `robustness.csv` (with BH q),
`lopo.csv`, `strata.csv`.

## Reproduction
* Exact reproduction: 2,123 EM Fab entries, 32 with a confirmed VH D/E slip; VL with a non-D/E first-occurrence error
  11/32 (34%) vs 1.29%, Fisher OR 40, p = 1e-12; the investigator's logit with paper-clustered SE gives OR 36, p = 3e-9.
  No join bug: run keys = err_first chains (459 = 459), all 155 EM D/E chains are in err_struct.pkl.
* Wording: `L_oth` excludes VL D/E runs. "VL has some first-occurrence error" is really 12/32 (37.5%) vs 2.0% (OR 30).

## What holds
* Paper level: 9/23 papers vs 2.7% (OR 23, p < 1e-6). Antibody-in-paper (VH unit) level: 9/24 vs 2.5% (OR 24).
* Leave one paper out: the worst case is dropping Cell Res 2021 (10.1038/s41422-021-00487-9), giving OR 22, p = 2e-6.
* Paper-level permutation within year bin: p < 5e-5. Within-paper Mantel-Haenszel (11 informative papers): OR 14, p = 0.02.
* The effect holds in SARS-CoV-2 entries (OR 32) and in other entries (OR 26), in 2020-21 (OR 19) and outside it (OR 34),
  and in single-Fab entries (OR 15). X-ray shows the same direction (2/4).
* GEE exchangeable by paper (outcome: VL any error), adjusted for year bin, resolution, chain-mean lq_z and number of VL
  chains: OR 14.9 (p < 1e-4).
* 9YC6 VL is a real error, not a SAbR failure: an 82-residue +5 run, and Cys93 sits 17 A from Cys23 (no disulfide).

## What weakens it
1. **Effect size is inflated.** 5 of the 11 events are one paper (Cell Res 2021: 7CZR/7CZS/7CZU/7CZV/7D00; two κ
   sequences, the same N-terminal VL run copied across symmetry copies and entries). Correct units give OR 15-24, not 36.
2. **Not specific to CDR3/FR4 slips.** A non-D/E VH error predicts a VL error almost as well (19.7% vs 1.9%, OR 12.7).
   In a joint GEE: H_DE OR 8.7, H_other OR 7.9. Among entries with any VH error, D/E vs other gives OR 2.9, p = 0.04
   (q = 0.04). This is general clustering of errors within an entry.
3. **Not the same mechanism.** The VL errors are mostly other modes (A ×6, C ×2, B ×2, F ×1). Both chains with D/E:
   2/32 (adjusted p = 0.06).
4. **"Same Fab" was not tested.** The analysis is at entry level. Only 3 multi-antibody entries have errors in both VH
   and VL: in 9PV5 and 9PV3 they are in the same Fab, in 7WP6 they are in different Fabs (83H7 VH, 36H6 VL). So this
   cannot separate Fab level from model/entry level.
5. **The outcome is held to a lower standard than the exposure.** The exposure uses confirmed D/E runs only, but the VL
   outcome uses all runs. 3/11 events (the 7CZR/7CZS/7D00 κ c4203 mode-A run) were "not confirmed" by the
   investigator's own N-terminal check. 6/12 VL events are origin 3 (chain end / next to an unbuilt loop). Removing
   origin 3 leaves 6/32 vs 1.2% (OR 20), which is still significant.
6. **Examples are inconsistent with the confirmed-only set.** 7D3K's VH D+ (+11) and 26PF's VL D− are "not confirmed"
   in err_struct.pkl. 8GNH's VL error is a D− run, so it is not counted in the 11/32.

## Nulls / no attenuation
* Adjusting for map-model fit (lq_z) and the number of VL chains did not attenuate the effect (GEE OR 14.9 → 14.9).
* SARS-CoV-2 status and the 2020-21 period: no modification (similar ORs in each stratum).
