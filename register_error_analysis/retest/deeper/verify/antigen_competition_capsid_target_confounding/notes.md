# Verification of "capsid_target": confounding

**Verdict: weakened.** The entry-level association is real. It still holds after adjusting for year, resolution,
SARS-CoV-2 status, chain type, chain count, unbuilt residues, local map fit, antibody organism and journal tier.
But "antibodies against capsids have more errors" does not hold up as a property of the target:

* About a quarter of it is numbering artifacts.
* It rests on 2–3 groups of co-authors (lab networks), not 12 labs.
* It is absent within labs.
* It is absent for human antibodies.

## What reproduces

* **Headline:** EM entries, capsid vs other, OR **4.72** (2.53–8.82), clustered by paper. This is the same
  32/154 vs 146/2598 as the investigator's.
* **Clustered by co-author-merged lab network:** OR 4.72 (2.22–10.0), p = 6e-5. With a wider Rao Z. network:
  p = 6e-4.
* **Adjustments that leave it at OR 3.9–4.8:**
  * log2 number of V chains;
  * antibody organism (from PDB entity annotation) and nanobody;
  * lowest germline identity and atypical L1/L3;
  * any V-region gap and N-terminal truncation;
  * worst-chain lq_z;
  * starting model and fitting protocol;
  * journal tier.
* **Strata:** it holds in all periods: ≤2019 4.9, 2020–21 9.4, 2022+ 3.4. It also holds at both resolution bins,
  in non-SARS-CoV-2 entries, and against other viral non-SARS-CoV-2 targets (4.0).

## What weakens it

1. **Numbering artifacts and CDR-internal disagreements**
   * **7 of 32 capsid error entries are confirmed numbering artifacts**, vs 1 of 146 other EM error entries.
     These are the rare-indel patterns that SAbR also reproduces on X-ray structures at ≤2.5 Å (case_files_ABCF
     lens):
     * FR1 gap at 26: 7eaj, 7eak, 8gsc, 8gsd, 8gse, 8gsf (Wang X./Zhu L. IBP network);
     * FR3 gap at 84+85: 9e89 (Hafenstein).
   * 4 more capsid entries are CDR3-internal disagreements with no framework shift (D0, delta undefined): bovine
     7d3l, 7d3m, 7fei and 7fej.
   * Together, 11/32 vs 4/146, Fisher p = 1e-6.
   * **Six of the nine capsid F_CDR1 entries are the FR1-gap-26 artifact.** So the investigator's mode signature
     ("9 of 14 F_CDR1, 7 of 10 D0") is mostly detector behaviour.
   * **With artifacts removed:** OR 3.64 (1.60–8.27), and 3.03 with all covariates. With the CDR-internal
     disagreements also removed: 3.15. Keeping only errors not next to an unbuilt residue: 2.90.
2. **Twelve labs are seven co-author networks**, from author lists:

   | Network | Papers | Notes |
   |---|---|---|
   | Hafenstein (Hafenstein S., Hafenstein S.L., Charnesky A.J.) | 6 | |
   | Xiamen (Cheng T., Li S.W.; Xia N.) | 3 | |
   | IBP (Wang X., Feng R., Fu W.; Zhu L., Zheng B., Wang K.) | 3 | 2 of these papers are artifacts |
   | FMDV/bovine (Lou Z., Kun L. = Li K.) | 3 | |
   | Zhu P. | 1 | |
   | McKenna | 1 | |
   | Hogle | 1 | |

   * **Paper level, clustered by network, artifacts removed:** OR 2.90 (1.28–6.58).
   * **Dropping the Hafenstein network alone:** 1.97 (p = 0.09).
   * **Dropping the top two networks:** 1.67 (p = 0.32).
   * **Dropping the top three:** 0.95.
   * The investigator's leave-one-lab-out treated "Hafenstein, S." and "Hafenstein, S.L." as separate labs and did
     not merge co-author networks.
3. **No within-lab effect.** In a conditional logit stratified by lab network, the capsid OR is:
   * 1.18 (0.39–3.55) with all errors;
   * 0.47 (0.12–1.86) with artifacts removed.

   Only 9 labs are informative. The networks behind most capsid errors publish only capsid antibodies by EM:
   Hafenstein 7/12 entries, Lou/Li K. 7/7, Zhu P. 6/14, McKenna and Hogle. So "capsid" cannot be separated from
   "these labs". In mixed labs, the capsid error rates are similar to the other entries: Xiamen 8% vs 5%.
4. **There is no effect in human antibodies.** At the unit level:

   | Antibodies | Capsid | Other | OR |
   |---|---|---|---|
   | Human | 2/80 | 71/2278 | 0.98 |
   | Mouse | 18/98 | 15/634 | 9.2 |
   | Other non-human (rat, cow, rabbit, dog) | 10/41 | 11/146 | 3.9 |

   * Interaction OR 0.20, p = 0.018.
   * OR 4 would predict 9.1 human errors; 2 were observed (p = 0.004).
   * The signal is mouse, rat, bovine and rabbit antibodies from capsid-specialist labs.
5. **The within-capsid lq_z result is the generic pattern.** Worst-chain lq_z, error minus clean:
   * capsid: −0.36;
   * non-capsid: −0.37.

   It is also partly circular, because a misregistered model fits its map worse.

## Nulls (no difference or no change)

* Number of V chains per entry: OR unchanged at 4.80. Capsid entries have slightly fewer V chains.
* Journal tier: 4.69.
* Unbuilt residues, N-terminal truncation and lq_z: 4.48–4.75.
* Capsid vs other, all-human-antibody entries: 2/40 vs 67/1269 (p = 1). Paper level: 2/23 vs 48/569.
* **Bovine long CDR-H3 is not a SAbR artifact.** SAbR and ANARCI agree completely on 15/15 high-resolution X-ray
  bovine VH (CDR3 22–65), and on 3/3 VL (`bovine_hires_check.csv`). So the Lou Z. bovine CDR3 disagreements are
  deviations of the models, not tool failures. They stay in as errors, except that the D0 entries have no
  register shift.

## Caveats

* Lab networks were merged by hand from the author lists of the capsid error papers only. Non-capsid labs were not
  merged in the same way.
* The conditional logit has only 9 informative labs.
* The R9 McNemar test, mixed labs erring on other targets but not on capsids, is biased by the different numbers of
  entries per class. Treat it as descriptive.
* The artifact list is from the case_files_ABCF lens: one high-resolution structure confirms the FR1-26 pattern
  (5AAM), and 3/3 confirm the FR3-84+85 pattern.
* The antibody organism comes from the PDB entity source annotation. The `species` column in chains.csv calls
  bovine and rat antibodies "mouse".

## Files

* **Scripts:** `v01_extract.py`, `v02_gaps.py`, `v03_reanalysis.py`, `v04_bovine_hires.py`, `v05_case_table.py`.
* **Tables:** `tests_capsid_confounding.csv` (with BH q), `capsid_error_entries_flags.csv`,
  `em_entries_verify.csv.gz`, `bovine_hires_check.csv`.
* **Logs:** `v03_reanalysis.log`, `v05_case_table.log`.
