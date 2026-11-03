# Verification of SN3 "rare IMGT column in the run window" (lens: confounding)

**Verdict: weakened, close to refuted as a modelling finding.**
* **The count reproduces exactly:** 14 of 199 cryo-EM unit-modes, against 0.94 expected (0.47%), O/E 14.9.
* **The statistical association also survives the usual confounders:**
  * deposit year, resolution, chain type, SARS-CoV-2, nanobody, species, atypical CDR length and unbuilt gaps (paper-clustered OR 4.9);
  * alternative control sets (O/E 7.7 to 14.9).
* **But most of it comes from the error definition itself:**
  * An insertion at a rare IMGT column is numbered at a different column by ANARCI (sequence) and SAbR (structure).
  * The disagreement run is the stretch between the two placements, so the same run appears in correct, high-resolution structures.
* **After removing these cases**, what remains is mostly one bovine lab, which is already a known finding.

## What was done
| Step | Script | Result |
|---|---|---|
| Reproduce; alternative control sets | `v01_reproduce.py` | **Reproduced:** 14/199 vs 0.94 expected. **Controls:** all non-first-error EM units O/E 12.3; all EM units 10.4; clean X-ray 7.7. Selection of "clean" controls is not the cause. |
| Per-residue numbering of the 14 positives | `v02_case_numbering.py` (`v02_case_numbering.log`) | **In 14/14 the run contains the rare column. In the 10 non-bovine positives, it is the offset between ANARCI's and SAbR's placement of the same insertion.** For example, 7cwu: ANARCI 1-4,4A vs SAbR 0-4. 22ge: ANARCI 9,9A-C,10 vs SAbR 8A-D,9. 12iu: ANARCI puts the insertion in CDR2, SAbR at FR3 71A-C. |
| Does a rare column predict a run regardless of method or period? | `v03_population.py`, `v06_adjusted.py` | **Any run in window W given a rare column, FR3 excluded:** EM OR 13.5 (6.8–26.7); X-ray (3–5 Å) OR 25.3 (7.1–91). Rare column × EM interaction 0.76 (p 0.70). **First-occurrence run:** X-ray OR 13.3 (p 3e-4). Rare column × EM interaction 0.92 (p 0.90). Rare column × 2020–21 peak interaction 0.71 (p 0.66). **This is not a cryo-EM or period-specific risk factor.** |
| Same insertions in correct structures | `v04_select_hires.py`, `v05_hires_scan.py` | **High-resolution X-ray (1.1–2.6 Å) chains with rare columns:** 15/44 have a run touching the rare column, vs a 4.7% detector background, p 8e-10. **Same column families as the positives: 12/18.** H 4A 5/5 (1.75–2.5 Å; same run 3-4A / SAbR 2..4 as 7cwu and 8fsj). H 85A-G 4/4. H 67A-C (6oca) 1/1. H 73 (9wmg) 1/1. H 70A (9lad) 1/1. |
| Classify positives; recount | `v07_recount.py` (`v07_positive_classification.csv`) | **Tool artifact, reproduced at high resolution (6):** 8c8t, 7cws/7cwu, 8fsj, 7nk2, 8bw0, 8r4b. **Confirmed model deviations (5):** 4 bovine from one FMDV lab (Lou Z./Kun L.; 2 papers; 2 of these 4 are D0, with no register shift); 8ulu κ (the same V domain is numbered identically at 1.75 Å in 8uki). **Probable model deviation (1):** 7t6x λ. **Untestable insertion-placement form (2):** 22ge rabbit, 12iu κ. |
| Adjusted models | `v06_adjusted.py` | **Unit × window, EM, paper-clustered:** M1 9.4; +SARS2/VHH/species 8.5; +atypical CDR length 7.2; +unbuilt gap 4.9 (2.1–11.6); without the CDR3/FR4 windows 4.4 (p 0.008). The association survives, but it does so equally in X-ray (see above). |
| BH | `v08_bh.py` → `all_tests_bh.csv` | 32 tests |

## Recount (matched expectation, as the investigator computed it)
| Positives counted | Observed | Expected | O/E | p (q) |
|---|---|---|---|---|
| As published | 14/199 | 0.94 | 14.9 | 2e-12 |
| Excluding the 6 high-resolution-reproduced artifacts | 8/193 | 0.93 | 8.6 | 6e-6 |
| Confirmed model deviations only | 5/193 | 0.93 | 5.4 | 0.003 (0.004) |
| Confirmed, register shift only (no D0) | 4/193 | 0.93 | 4.3 | 0.015 (0.018) |
| Confirmed + probable, excluding the bovine FMDV lab | 2/186 | 0.88 | 2.3 | 0.22 (0.25) |
| Confirmed only, excluding the bovine FMDV lab | 1/186 | 0.88 | 1.1 | 0.59 (0.65) |

* **The investigator's fixed-window results are made entirely of artifacts and untestable cases:**
  * A 6.8%: 7cws and 8fsj are artifacts; 7t6x and 22ge.
  * B 16%: 8bw0 and 8r4b are artifacts; 22ge and 12iu.
* **Rare columns in the CDR3 (D) window in cryo-EM:**
  * human long CDR3: 18/19 clean;
  * rabbit: 8/8 clean;
  * cow: 3/3 erroneous, all from the Lou/Kun lab.
  * X-ray bovine ultralong structures (Crowe, Wilson) are clean.

  So the D-window signal is a single lab, the same lab already behind "cow 6/19, 5 from one lab".

## No difference
* Rare column × method (EM vs X-ray) interaction, for any run (p 0.70) and for first-occurrence runs (p 0.90).
* Rare column × 2020–21 peak interaction (p 0.66).
* Choice of control set:
  * clean, all-non-error or X-ray controls change O/E only from 14.9 to 7.7;
  * matching on species and atypical CDR length does not lower the expectation (0.80).
* Two high-resolution λ with an IMGT-10 residue (4r26, 4fqc), and 9nj9 λ (2.6 Å EM), give no run.
* CDR2 60A/61A long CDR2s give no run in 15 high-resolution structures (alpaca, rabbit, human, cow). CDR3 111A/112A give no run in 7 (rabbit/human). So not every rare column triggers the detector; the problem is specific to the families above.

## Caveats
* **High-resolution check:**
  * It uses a different structure carrying the same column insertion, not the same antibody, except for 8uki/8ulu and bovine.
  * H 4A and H 85A-G are tested well (5 and 4 structures); H 67A-C, 73 and 70A have one structure each.
  * 22ge (unique rabbit FR1 insertion) and 12iu κ have no high-resolution relative.
* **A run at an insertion is not proof the model is right.** The disagreement simply cannot tell a register error from a numbering convention. Both are expected at any resolution.
* **The 4.7% background** is the detector's rate on the validation set's unflagged chains, not on a resolution-matched set.
* The bovine model deviations are real (15/15 high-resolution bovine VH agree; see `../antigen_competition_capsid_target_confounding`). But they are one lab, and 2 of the 4 have no register shift.
