# Case files for modes A, B, C and F: what links the error structures?

**Short answer.** Reading the cases one by one, the thing that recurs is a mismatch at the slipped strand. On one side
is the antibody's actual sequence length; on the other is whatever the modeller started from or counted from. That
reference can be:
* a template antibody;
* an assumed standard numbering;
* an absent N-terminus.

The size of the shift often equals the size of the length difference.

A second link is not a modelling error at all. Some recurring "errors" are a **numbering artifact of rare
germline/somatic indels**. Two numbering schemes place the gap differently. SAbR places it structurally; ANARCI places
it by sequence HMM. The same disagreement appears on 1.4–2.5 Å crystal structures.

Neither link involves the lab, the antigen race or a tool.

## Sample and data

**Cases.** There are 29 cases: 8 in mode A and 7 in each of B, C and F. In each mode, 6 are cryo-EM and 1 is X-ray
(A has 7 cryo-EM).
* They are chosen from the ranked candidates in `tables/candidates_<mode>.csv`: long runs, poor site fit, one per lab
  and paper, spread over 2010–2026.
* The 15 AL amyloid fibrils are excluded.
* Two choices were swapped out:
  * 4xnz, an HIV bnAb with an FR3 insertion (a known artifact class);
  * 9pur (an undefined shift). 9yc6 was taken instead.

**Controls.** There are 45, with no disagreement run anywhere in the entry and the same method and chain type. Each
case gets up to three:
* the same antigen, from another lab, within ±18 months;
* otherwise, the same period and target class;
* a **near miss**: the same local hazard, for example an unbuilt N-terminus or unbuilt residues in the window, but
  modelled correctly.

Same-study controls are excluded (same paper, or same lab within 90 days).

**Data for each entry** (`s02`–`s05`):
* RCSB REST entry, mmCIF and the wwPDB validation XML;
* EMDB entry and validation analysis;
* Europe PMC open-access methods (full text for 14 of 29 cases and 33 of 45 controls);
* the starting models named in the text or the PDB record, with their V regions compared to the case at the error
  location;
* the same V region elsewhere in the PDB, and same-antigen entries from other labs within ±6 months;
* the local site: sequence, unbuilt residues, Q-score/RSCC percentile, local resolution, validation outliers, B-factor,
  and contacts with antigen or partner chains.

Each case file is in `case_files.md`; the summary row with the "clue" is in `tables/case_summary.csv`.

## Per-mode commonalities (from the case files)

### A, FR1 N-terminus (8 cases)

**What the cases share.** In 6 of 8 cases, the strand has no N-terminal anchor:
* SEQRES starts after IMGT 1: 7tz5 at 2, 9mnb at 3 (a rabbit VH), 8e8l at 4;
* or the first residues are not built: 7jhg, 8rvn and 6bpe start at IMGT 3–4.

**What the starting points were.** Other-antibody templates with the *same* N-terminal length (7czu, 8rvn, 6bpe), an
ABodyBuilder model (8e8l) and an AlphaFold2 model (9mnb). An AlphaFold start did not prevent the slip.

**The long A runs** (6ad0: 1–32; 7tz5: 2–31) end in CDR-L1:
* 6ad0 is a mouse IGKV1-117 with L1 = 11, built from a template with L1 = 6.

**One case is really a copy.** 7jhg, 7jhh and 7m74 (Melcher) inherit the identical −2 slip from 6CMO (Xu H.E., same
institute). 6CMO is a recorded starting model whose light chain is 98% identical. The identical-sequence copy rule
missed it.

**The hazard is not sufficient.**
* 5 of 15 A controls share the truncation (first residue at IMGT 3–4) and were built correctly.
* The period control 7whz has a window sequence *identical* to 7tz5 (IGLV1-40, starting at IMGT 2) and was built
  correctly.

### B, CDR2 to FR3 (7 cases)

**Where the slip sits.**
* 4 of 7 cases sit at the antigen interface.
* The controls touch antigen at least as often, so this is not a distinguishing feature.

**The clearest case is template threading.** In 7wvl (X-ray), "the sequence of the STEC90-C11 Fab was slowly changed
to that of P4A2". The template CDR-H2 has 7 residues and P4A2 has 8, so FR3 slid by +1.

**A slip can also happen without any length mismatch.** In 7mkl the templates were chosen by BLAST (1KIQ). The CDR2
lengths are equal, yet the slip sits at the antigen interface. A later "Atomic model" revision did not fix it.

**Scan-wide** (`s07` T7), an indel at IMGT 66–75 (the CDR2/FR3 boundary) is the location-specific marker:

| | Error units | Clean units | OR (95% CI) | q |
|---|---|---|---|---|
| Indel at IMGT 66–75 → B | **28% (7/25)** | 0.15% | 222 (54–906) | < 1e-6 |
| Same feature → C (location control) | 0/22 | | | n.s. |
| Same feature → A (location control) | 1/53 | | | n.s. |

**One case may be an artifact.** 12iu's κ CDR2 gap pattern disagrees with SAbR in all 3 chains that carry it.

### C, FR3 (7 cases)

**The defining feature is an FR3 indel.** The genuine cases are:
* **9dm7: a template-length mismatch.** The Fab was built "based on 5UCB". The case FR3 is two residues shorter, and
  FR3 slipped by +1 over 72–92.
* **7e23 and 7stf: a κ repeat slip.** κ FR3 FSGSGSGTDF slipped by exactly **−2**, which is the SG-repeat period. These
  are 2 of the 2 κ mode-C units.
* **Unbuilt neighbours:** 7kra (9 unbuilt; hand-built de novo, even though the germline V had 2 earlier high-res
  structures), 7mhz (4) and 8w5e (6; mouse IGHV5).

**Two cases are numbering artifacts:**
* 2x7l (FR3 gaps at 80+84);
* 9yfu (gap at 85; probable).

**Scan-wide** (T1, T2), atypical FR3 length marks mode C:

| | Error units | Clean units | OR (95% CI) | q |
|---|---|---|---|---|
| Atypical FR3 length → C | **41.7% (10/24)** | 1.46% (59/4,036) | 52 (19–141) | < 1e-6 |
| Same, after removing confirmed artifacts | 36.4% (8/22) | 1.46% | 41 (14–120) | < 1e-6 |
| Same feature → B | 36% | | | |

**The gap position decides which.** Clean chains with a one-residue FR3 deletion have ANARCI's gap at 81 (44/44
clean). The error chains have it at 84, 96, 82, 89 or 94.

### F, CDR1 (7 cases)

**In most cases, the CDR1 length differs from the starting model and the shift is absorbed downstream:**

| Case | Starting model and length difference | Shift |
|---|---|---|
| 3jbf | VHH template 1I3U, CDR1 8 vs 5 | **−3**, exactly the difference |
| 9ml5 | Rabbit CDR-H1 SSSYWICW (9) docked from human Fab templates (8 or 10) | +1 |
| 7e5y | X-ray molecular replacement with a κ Fab (L1 = 6) for a λ chain (L1 = 8) | +1 over FR2–FR3 |
| 6lht | Mouse IGKV8-30, L1 = 12 | **+5** over CDR1–CDR2 |
| 9yc6 | Mouse IGKV1-117, L1 = 11 | **+5** over the whole chain, from CDR1 to the end |

**One case is an artifact, and it links two labs.**
* 8gse (Fu W., anti-echovirus-3 Fab 6D10) and 7eaj/7eak (Feng R., anti-echovirus-3 Fab 5G3) are different labs and
  different antibodies.
* They share a rare mouse VH **FR1 deletion** (SCQA-FYTF; ANARCI gap at 26).
* They also share the same 27–35 disagreement with an undefined shift.
* SAbR reproduces it on the 2.49 Å structure 5AAM, so it is a numbering artifact (T3/T4).

## What links the structures across modes

1. **Some recurring "errors" are numbering artifacts of rare indels** (`s08`–`s10`, `tables/hires_gap_check.csv`).

   **The test.** SAbR was run on high-resolution X-ray structures carrying the same ANARCI gap patterns.

   | ANARCI gap pattern | High-res structures that disagree with ANARCI | Where |
   |---|---|---|
   | Heavy FR3 gap at 84 | **4/4** (1.38–2.0 Å) | IMGT 81–83, +1 |
   | FR3 gap at 84+85 | **3/3** (1.5–2.5 Å) | +2 |
   | FR3 gap at 80+84 | **1/1** (2.3 Å) | |
   | FR1 gap at 26 | **1/1** (2.49 Å) | 27–35 |
   | FR3 gap at 81 (negative control) | **0/4** | |
   | FR3 gap at 96 | 0/1 (7te4) | |

   * The gap-at-96 result means the 93–95 runs in 7te9, 7teb and 8xlk are real deviations of the EM models.

   **How much this explains:**
   * **6 of 31 mode-C units** (12 runs; 4 of 7 X-ray C units: 2x7l, 30ad, 30bg and 9pf9; plus 9bop and 9e89);
   * **2 of 13 mode-F units** (7 runs);
   * no A or B units.

   These are the "same error in different labs": antibodies from IGHV3-53/3-66, IGHV1-69, IGHV3-7, and a mouse VH1
   gene (the FR1 deletion). The indels make the two numbering methods disagree whoever builds the model.

2. **A length mismatch between the starting point and the antibody, located at or just upstream of the slip.**

   **The textbook cases:** 7wvl, 9dm7, 3jbf, 9ml5, 7e5y, and the long-L1 mouse κ chains 6ad0, 6lht and 9yc6. The shift
   size equals the indel size in 3jbf (−3), 7e23/7stf (−2, the repeat period) and 9yc6/6lht (+5).

   **At case-file level, mismatch is not discriminating:**

   | | Cases | Controls | p |
   |---|---|---|---|
   | Named different-antibody template differs in length at the site | 24% | 18% | 0.56 |

   The controls handled their mismatches. A mismatch is a hazard, as are a missing N-terminal anchor and unbuilt
   neighbours. The outcome depends on whether the modeller recounted the residues.

3. **Rabbit antibodies** (cases 9mnb and 9ml5; near-miss control 9ml7) are over-represented scan-wide (`s11`):

   | | Error units | Clean units | Adjusted OR (95% CI) | q |
   |---|---|---|---|---|
   | Any first-occurrence error | **5.3% (7/133)** | 1.57% | 3.3 (1.4–7.7) | 0.03 |
   | B | | | 8.1 | 0.03 |
   | A | | | 4.6 | 0.11 |
   | C | | | 6.5 | 0.08 |

   Rabbit VH have N-terminal deletions, an extra CDR-H1 Cys and atypical CDR lengths, so this fits point 2. Weak: 7
   error units.

4. **Near-identical template copies hide among first-occurrence errors** (`tables/near_copy_matches.csv`).
   * Seven runs, in 6 units, have an earlier 90–99.9% identical V carrying the same run at the same location with the
     same shift:
     * 7jhg, 7jhh and 7m74 ← 6cmo (cross-lab, starting model recorded);
     * 8snc and 8snd ← 7sp7 (same lab);
     * 7e3c ← 7czr (93%: a shared germline κ chain, possibly coincidental);
     * 9pv5 ← 7wp6 (mode E).
   * That is about 2–3% of first-occurrence units.

## What does NOT link them (case-file level; `tables/casefile_tests.csv`, Fisher/Mann-Whitney; nothing has q < 0.3 except the two below)

**The two that differ** are site fit (0.31 vs 0.51 percentile) and site B-factor z (0.66 vs 0.00, q = 0.005). Both are
partly consequences of the error.

**No difference:**

| Feature | Cases | Controls |
|---|---|---|
| Unbuilt residues near the site | 31% | 24% |
| SEQRES N-terminal residues unbuilt (A) | 38% | 33% |
| Site contacts antigen | 41% | 41% |
| Site contacts partner chain | 52% | 34% |
| Local resolution, site minus domain | 0.00 Å | 0.01 Å |
| Clash excess at site | n.s. | |
| Antibody template named | 71% | 67% |
| Different-antibody template | 48% | 40% |
| Methods: homology tool | 21% | 15% |
| Methods: AlphaFold | 7% | 24% |
| Methods: rigid body | 7% | 27% |
| Methods: ISOLDE, Rosetta, local refinement, Fv-only, "poor density" | n.s. | |
| Methods: icosahedral/capsid | 29% | 9% (p = 0.17) |
| Prior high-res structure of the same V region | 14% | 20% |
| Contested antigen (≥5 other-lab entries within ±6 months) | 21% | 31% |
| First antibody structure of the antigen | 48% | 30% (p = 0.14) |
| Germline identity | 0.90 | 0.90 |
| Identity to nearest earlier V | 0.86 | 0.87 |

**Competition.** For the SARS-CoV-2 cases, the same-antigen structures from other labs within ±6 months were
overwhelmingly clean, at about the field rate:

| Case | Other-lab same-antigen entries with errors |
|---|---|
| 7czu | 9/133 |
| 7e39 | 22/197 |
| 7mkl | 11/177 |
| 7wvl | 9/258 |
| 7e5y | 23/198 |
| 9ml5 | 5/52 |

The other cases had no competitors at all.

**Deposition metadata.** Deposition and processing site, revision history and software showed no shared pattern.
Two cases (7mkl and 7mhz) had an "Atomic model" revision that left the run in place.

## Caveats

**Sample size and selection.**
* The case files are small (29 vs 45) and curated toward clear cases, not random.
* Case-file tests are descriptive.
* Open-access methods exist for only 48% of cases.

**The artifact evidence is thin per pattern.**
* It rests on 1–4 high-res structures per pattern.
* Other indel patterns (FR3 gaps at 85, 89, 94, 66–68, 74–75; κ CDR2 59–62) are unconfirmed and could also be
  artifacts.
* `sabr/out/validation.jsonl` is *not* purely <3 Å (it contains 7te9), so the "hires" columns in
  `tables/gap_patterns.csv` are unreliable. Use `hires_gap_check.csv`.

**The indel tests partly overlap the detector's definition.** SAbR and ANARCI can only disagree by convention at
indels.

**The scan models are fragile.** Few events, very wide CIs; zero-cell tests fall back to unadjusted Fisher.

**Near-copy detection** uses a 90% identity cut-off, which can pair different antibodies that share a germline light
chain.

## Files

* **Scripts:** `s01_select.py` … `s12_write_casefiles.py`.
* **Case files:** `case_files.md`, `methods_snippets.txt`.
* **Tables** (in `tables/`):
  * case selection and controls: `cases.csv`, `controls.csv`, `candidates_*.csv`;
  * per-entry detail: `local_sites.csv`, `context.csv`, `methods_flags.csv`, `templates.csv`;
  * case-file results: `case_table.csv`, `case_summary.csv`, `casefile_tests.csv`;
  * scan-wide tests: `scan_tests.csv`, `rabbit_tests.csv`, `all_tests_bh.csv`;
  * artifact analysis: `gap_patterns.csv`, `gap_artifact_runs.csv`, `hires_gap_check.csv`, `artifact_runs.csv`,
    `artifact_share_by_mode.csv`;
  * copies: `near_copies.csv`, `near_copy_matches.csv`.
* **Caches:** `scratchpad/deeper/cache/case_files_ABCF/`.
