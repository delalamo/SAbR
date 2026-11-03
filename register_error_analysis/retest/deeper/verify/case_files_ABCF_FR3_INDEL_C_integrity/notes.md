# Verification of FR3_INDEL_C (case_files_ABCF): data-integrity and statistics lens

**Verdict: weakened, close to refuted.**
* The numbers reproduce exactly.
* The interpretation does not hold. An atypical FR3 length marks places where the detector disagrees with ANARCI, not
  places where cryo-EM modellers made errors.
* 8 of the 10 exposed error units are not first-occurrence register errors at an indel.
* What is left is 2 units (7te9, 9dm7). That cannot support "strong".

## Reproduction (v01, v04)
* The investigator's T1 reproduces exactly:
  * cases 10/24;
  * clean controls 59/4,036;
  * adjusted OR 52.1 (19.3–141), p 7e-15;
  * Fisher OR 48, p 1.7e-12.
* T1-excl also reproduces: 8/22, OR 40.5 (13.7–120).
* Recomputing the feature with a fresh ANARCI run (human+mouse HMMs, every vseq) gives the same result:
  * 99.9% of chains agree;
  * 10/24 vs 63/4,018, OR 48.
* No join or code bug was found in s07.

## Problems found
1. **Most exposed cases are not genuine errors.**
   * *Method.* v05b superimposes each case on another structure of the identical V region and checks residue
     correspondence by CA. v07 re-runs SAbR on those reference structures.

   | Case | Finding | Evidence |
   |---|---|---|
   | 9bop, 9e89 | Confirmed artifacts | Investigator |
   | 8c8t | HIV FR3-insertion class (3BNC117), the same class as 4xnz, which the investigator excluded as a known artifact | 8/8 EM units with the 85A–D insertion carry a C run, and 7 of them were already classed as copied/artifact upstream |
   | 7m3l | The same Fab at 2.4 Å (7m3n) shows the identical 90–95 disagreement | Coordinates agree within 0.33 Å, with no offset |
   | 9yfu | The same nanobody at 2.94 Å (8tb7) shows the identical 81–84 disagreement | Coordinates agree within 0.70 Å, with no offset |
   | **7y0o** | Not a model error | See below |
   | 8xlk | A coordinate copy of 7te9 (same Fab2 anti-GluN2B, another lab, 2 years later), not a first occurrence | Run residues within 0.36 Å |
   | 9ml5 | Rabbit VH; FR3 = 37 is the rabbit germline norm, not an indel | 25 of the 59 "atypical-FR3" clean controls are also rabbit |
   | 7te9 | Supported as a local deviation | The same Fab at 2.46 Å X-ray (7te4) agrees with ANARCI. 7te9's CA at 93–95 sits 4.2–4.7 Å off. This is not a clean ±1 shift; it is a different placement of a real somatic deletion |
   | 9dm7 | No reference structure | SEQRES lacks 2 non-adjacent residues ("NSLRAEDT" becomes "NSREDT") of an otherwise invariant 4D5-type library framework. This is plausibly a sequence-entry error |

   * **7y0o in detail.** Its clean twin 7y0n (same antibody R15, same lab, same deposition day) has *the same register*:
     * over FR3 and CDR3 (84–114), each CA is 0.3–2.4 Å from the same residue and never closer to a neighbour;
     * yet SAbR flags 19 residues in 7y0o and 1 in 7y0n;
     * so the run is a SAbR gap-placement flip at the gap-94 deletion, not a model error.
2. **The association is not cryo-EM specific** (v04 d0/d1). The pattern is a detector property.

   | | Cryo-EM | X-ray 3–5 Å |
   |---|---|---|
   | C units that are FR3-atypical | | **6/7** (vs 2.9% of clean units; OR 202), 5 of them confirmed or class artifacts |
   | First-occurrence C rate, FR3-atypical units | 14.5% | 10.5% |
   | First-occurrence C rate, typical units | 0.35% | 0.05% |

   * The interaction OR (EM vs X-ray) is 0.15 (0.01–1.6), p = 0.11: if anything, the effect is weaker in cryo-EM.
3. **The control set is selected.** 28 FR3-atypical EM units whose C runs were classed upstream as copied or artifact are
   dropped from both arms. Counting all non-C units as controls lowers the OR from 52 to 31.
4. **The detail "clean one-residue deletions have the gap at 81 (44/44)" is not reproduced.** Clean units also carry
   heavy FR3 gaps at:
   * 83 (9cv7, 9cu6);
   * 85 (9jrt);
   * 92 (4xhj, X-ray);
   * 94 (7y0n, the identical twin of error case 7y0o).

## Residual association (v06)

| Case set | Cases | Clean controls | Fisher OR | p | Adjusted OR | q (BH, investigator's family) |
|---|---|---|---|---|---|---|
| All non-genuine positives removed | 3/17 | 59/4,036 | 14 | 0.002 | | 0.004 |
| Rabbit removed from both arms | 2/16 | 34/3,981 | 17 | 0.009 | 27 | 0.017 |
| Only the reference-supported case (7te9) | 1/16 | | 7.7 | 0.13 | | 0.19 |

These are post-hoc numbers resting on 1–2 antibodies.

## Files
* v00_anarci_fresh.py
* v01_reproduce.py
* v02_circularity.py
* v03_pattern_members.py
* v04_stats.py
* v05a_find_same_v.py
* v05b_coord_register.py
* v06_residual.py
* v07_sabr_refs.py
* tables/v01_reproduce.csv
* tables/v01_C_error_units.csv
* tables/v02_*.csv
* tables/v03_pattern_members.csv
* tables/v04_tests.csv
* tables/v04_feature_discordant_chains.csv
* tables/v05a_same_v.csv
* tables/v05b_coord_register.csv
* tables/v05b_coord_summary.csv
* tables/v06_residual.csv
* tables/v07_sabr_refs.csv

## Caveats
* The CA-correspondence test needs a structure of the same V region. 9dm7 and 9ml5 have none.
* 7te9's deviation is real relative to 7te4, but a conformational difference at the deletion cannot be excluded.
* Rabbit identification uses q_species_all from seq_novelty. chains.csv labels these chains "human".
* There is a minor data inconsistency: the identical nanobody is flagged vhh=1 in 9yfu and vhh=0 in 9jrt.
