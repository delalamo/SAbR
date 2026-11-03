# Verification (confounding lens): FR3START_INDEL_B — "an ANARCI indel at IMGT 66–75 marks mode-B errors"

**Verdict: refuted as a marker of register errors.** The association reproduces exactly and survives every classical
confounder. But it is mainly a detector artifact created by the error definition itself:
* ANARCI places a CDR2/FR3 indel at 66–75; SAbR places it structurally, usually inside CDR2.
* The residues between the two placements become a ≥3-residue "run" even when the model is built correctly.
* In 5 of the 7 error units, a check that uses no numbering shows the deposited model is not register-shifted.
* One possibly genuine at-site case remains (8jag), which is not significant.

## 1. Reproduction (v01, v06)
* EM unit-level test, controls = units with no run of any kind: **7/25 B units vs 6/4,036 clean units**.
  Adjusted OR 222 (54–906), identical to `s07` T7.
* Every EM unit carrying the feature is either a B error (7) or clean (6). None has a copied- or artifact-only run.
* Of the 6 clean controls, 3 are the same antibody (VRC-PG04, 67A insertion) in 3 papers.

## 2. Classical confounders do not explain it (v06; all p < 1e-9)
| Adjustment | OR |
|---|---|
| + SARS-CoV-2 | 215 |
| Cluster by last author | 222 (56–879) |
| + atypical CDR2 length | 181 |
| Heavy/VHH only | 201 |
| Non-SARS-CoV-2 only | 223 |
| Deposited 2022+ only | 656 |
| Control = every non-B unit | 234 |

* X-ray 3–5 Å shows the same association: 2/7 vs 1/1,914.
* All 7 B units with the feature were deposited 2022–2026, so the 2019–21 error peak is not driving it.
* The 7 come from 7 different labs.
* **Unbuilt residues do not explain it.** None of the 7 runs is labelled "next to an unbuilt loop".
* **Local map fit does not explain it.** The run Q-score percentile within the V domain is 48 with the feature and 44
  without (p = 0.79).

## 3. Definition overlap: the feature itself produces SAbR/ANARCI disagreement (v02, v05, v07)
**Scan chains with the indel modelled contiguously:**

| | Any disagreement in IMGT 55–80 | ≥3 residues |
|---|---|---|
| EM, with the indel | 73% | 32% |
| EM, without | 0.7% | 0.5% |
| X-ray 3–5 Å, with the indel | 23% | 13% |
| X-ray 3–5 Å, without | 0.3% | 0.2% |

* The X-ray rows with the indel are dominated by large light-chain deletions, which never disagree.

**High-resolution X-ray structures with the same ANARCI pattern**, run through SAbR (`v05`):

| Structure | Pattern (same as) | SAbR/ANARCI disagreement |
|---|---|---|
| 9pyn_B, 2.34 Å, **same antibody as 9pyh** | κ deletion at 66 | 4 residues, 67–70 |
| 6oca_C, 2.11 Å | 67A–C insertion (8r4d) | 10 residues |
| 9lad_B, 2.02 Å | 70A insertion (8bw0) | 3 residues |
| 3se9_H, 2.0 Å | 67A insertion (the clean VRC-PG04 controls) | 2 residues |

* For 3se9, the 2-residue disagreement matches the EM copies exactly. Those controls are "clean" only because 2 is
  below the 3-residue run threshold.

**"Only that location"** (C 0/22, A 1/53) is exactly what a gap-placement artifact predicts. It is not evidence of a
modelling mechanism.

## 4. Map-free and numbering-free checks of the 7 units (v03, v09, v10)
**Map-free.** SAbR was run on ImmuneBuilder and IgFold models of the same sequences. The run window disagrees by ≥3
residues in **6/7 antibodies with the feature vs 2/17 B antibodies without it** (p = 0.0013).

**Numbering-free method.** Superpose the framework, then ask whether each run residue sits on its own position in a
same-sequence reference or on a neighbour's.
* Positive control 7wvl (documented template threading): distance to own residue 3.7 Å, to a neighbour 0.6 Å; 100% of
  residues sit closer to a neighbour.
* Non-feature EM B errors such as 7sp7, 8snd, 7e39 and 6p65: 88–100% closer to a neighbour.

**The 7 feature units:**

| Unit | Evidence | Classification |
|---|---|---|
| 9pyh | vs the 2.34 Å X-ray of the same antibody: own 0.48 Å, neighbour 3.58 Å, 0% shifted | artifact |
| 8bw0 | own 0.7 Å vs neighbour 3.5 Å; SAbR numbering on both predictions identical to the deposited one | artifact |
| 8snc | 0–17% shifted; numbering on predictions identical to the deposited one (unlike same-lab 7sp7/8snd, which are genuine shifts) | artifact |
| 8r4d | no 3-residue shift; numbering on predictions identical to the deposited one; 6oca reproduces it | artifact |
| 9om5 | chains c/g/k (the run) vs a/e/i (same sequence, same map): 20–30% shifted, own 2.0 Å vs neighbour 3.1 Å; a local conformational variant that flips SAbR's gap placement, not a register slip | not a register shift |
| 8jag (8jam, 8jap) | 88–100% closer to a neighbour vs IgFold; the ImmuneBuilder loop differs by more than 7 Å | **possibly genuine**, unresolved |
| 22ge | run at 53–60, off the 69A insertion; poor framework superposition | unresolved, off-site |

## 5. After reclassification (v11, Fisher)
| Scenario | Error units | Control units | OR | p |
|---|---|---|---|---|
| 5 non-shift units moved to controls; 8jag and 22ge kept | 2/20 | 11/4,041 | 41 | 0.002 |
| Only the at-site case (8jag) | 1/19 | 11/4,041 | 20 (CI 5–165) | **0.055** |

The anecdote in the finding's detail (7wvl) and 7mkl do **not** carry the feature.

## Caveats
* The predicted models (ImmuneBuilder, IgFold) may also be wrong in loops.
  * The 9pyh conclusion rests on a same-antibody X-ray; 9om5 rests on its own NCS copies.
  * 8bw0, 8snc and 8r4d rest mainly on predicted models, plus same-pattern X-ray structures of different antibodies.
* The high-resolution same-pattern check covers 3 patterns, with one structure each.
* The original `validation.jsonl` is not a <3 Å set: most of its entries are 3–5 Å. My earlier "hires" grouping from it
  was dropped.
* A side observation, not tested formally: about 5 of 17 non-feature EM B antibodies (7nk2, 7ty0, 7oh0, 9heo, 12iu) are
  also in register with the predictions. More B runs may be artifacts.

## Files
* Scripts: `v01_reproduce.py`, `v02_determinism.py`, `v03_predicted_B.py`, `v04_pdbwide_census.py`, `v05_hires_sabr.py`,
  `v06_stats.py`, `v07_anydis_and_fit.py`, `v08_summary_tests.py`, `v09_register_vs_prediction.py`,
  `v10_direct_pairs.py`, `v11_reclassified.py`.
* Tables:
  * `v06_stats.csv`, `v08_all_tests_bh.csv` (BH q), `v08_feature_units_evidence.csv`, `v11_reclassified.csv`;
  * `v03_predicted_B.csv`, `v05_hires_sabr.csv`, `v09_register_vs_prediction.csv`, `v10_direct_pairs.csv`;
  * `v07_anydis.csv`, `v07_fit_B_runs.csv`, `em_units_with_feature.csv`;
  * `v04_pdbwide_patterns.csv`, `v04_hires_candidates.json`.
