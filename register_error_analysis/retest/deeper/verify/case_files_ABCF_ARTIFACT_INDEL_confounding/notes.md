# Verification of ARTIFACT_INDEL (case_files_ABCF), confounding lens

**The claim.** Some recurring mode C and mode F runs are numbering artifacts of indels, not register errors. The
evidence offered is that SAbR disagrees with ANARCI in the same window on high-resolution X-ray structures. The
claimed patterns are FR3 gaps at 84, 84+85 and 80+84, and an FR1 gap at 26. The control pattern is an FR3 gap at 81.

**Verdict: the core claim survives every adjustment and gets stronger.** A pure detector artifact cannot be produced by
year, method, lab or map quality, and the data agree:
* the disagreement appears at 1.4–2.75 Å from 12 labs (1995–2026);
* it appears on map-free predicted models;
* it never appears in the gap-81 controls.

Three details need correcting:
* the germline labels;
* "rare indel" as the mechanism for half the mode C units;
* "deterministic".

The effect size is a lower bound.

## What was done (scripts v01–v06b; tables/)

| Script | What it does |
|---|---|
| v01_census.py | PDB-wide census of each ANARCI gap pattern, from the 52k-sequence ANARCI cache. Each pattern exists in only 6–42 entries PDB-wide, so s09 had tested nearly everything available. |
| v02_scan_determinism.py | Uses existing SAbR output for every scan chain that carries a pattern, and counts disagreement in the window. |
| v03_hires_sabr.py | Re-runs SAbR on **every** X-ray entry ≤3.3 Å that carries a pattern (44 entries; s09 tested 15). It uses the scan's own protocol: ANARCI on SEQRES, domain trimming, unobserved residues. It was run twice: with the scan-default ANARCI species (human+mouse) and with all species (v03b). |
| v04_predicted_models.py | Runs SAbR against ANARCI on the ImmuneBuilder and IgFold models of every C_FR3/F_CDR1 first-occurrence run (42 run windows, 31 antibodies). |
| v05_stats.py | Fisher exact tests, Mantel–Haenszel tests across method and period strata, an exact McNemar test, and BH correction. |
| v06 / v06b | Local map fit (Q-score/RSCC percentile) and unbuilt residues at the run windows, compared with clean chains. |

## Results (tables/v05_tests.csv; all q are BH across these 14 tests)

| Test | Claimed patterns | FR3 gap-81 control | p (q) |
|---|---|---|---|
| T1: X-ray ≤2.75 Å, entries | **15/16** | 0/17 | 1.5e-8 (2e-7) |
| T1b: X-ray ≤2.75 Å, one per antibody | **12/13** | 0/14 | 7.5e-7 (5e-6) |
| T2: 3–5 Å scan, one per antibody | **8/9** | 0/17 | 5.8e-6 (2e-5) |
| T2 stratified by method (EM 4/5 vs 0/12; X-ray 4/4 vs 0/5) | | | Mantel–Haenszel 2.2e-4 |
| T2 stratified by period (≤2019, 2020–21, ≥2022) | | | Mantel–Haenszel 4.2e-4 |

**T3: predicted models.** The question is whether a predicted model reproduces the run window. The claimed set is 8
artifact windows; the other set is 34 C/F windows. Fisher p values are 5e-5 (IgFold) and 0.002 (ImmuneBuilder).

| Predictor | Claimed windows reproduced | Other C/F windows reproduced |
|---|---|---|
| IgFold | **8/8** | 7/34 |
| ImmuneBuilder | **6/8** | 5/34 |

* ImmuneBuilder misses only the two windows with a 2-residue gap at 84+85 (9e89, 30bg). However, 30bh, the
  sister antibody of 30bg, disagrees at 2.75 Å.
* Genuine cases are not reproduced: 9dm7, 7kra, 7stf, 7e23, 8w5e, 9ml5 and the 7te9 family.

**T4: ANARCI species setting.** It matters. The same 20 gap-81 rabbit controls disagree **20/20** with ANARCI's
all-species HMMs and 0/20 with the default (exact McNemar p = 1.9e-6).

**Labs and periods.** The 16 high-resolution claimed-pattern entries come from 12 labs, deposited 1995–2026. The 17
controls come from 12 labs, deposited 2013–2026. The artifact units come from 8 papers and 8 labs, and include 4 X-ray
and 4 EM units. None involves SARS-CoV-2.

**Simpler factors explain nothing.**
* No artifact run has an unbuilt residue in or within 3 residues of its window (0/8 units, vs 4/35 other C/F units).
* Local fit at IMGT 81–83 matches the same loop in clean chains:

| Method | Artifact chains | Clean chains | p |
|---|---|---|---|
| EM | 0.36 (n = 3) | 0.36 (n = 968) | 0.87 |
| X-ray | 0.30 (n = 9) | 0.28 (n = 504) | 0.91 |

* Within C/F runs, the window fit percentile does not separate artifacts from the others (0.36 vs 0.46, p = 0.10).

## Corrections to the finding

1. **The germline labels are wrong for 3 of the 6 mode C units.**
   * 2x7l, 9bop and 30ad (and 9ml5) are **rabbit** VH: IGHV1S69/IGHV1S45 at 78–90% identity under ANARCI's
     all-species HMMs.
   * chains.csv labels them human IGHV3-66 or IGHV3-53 at 55–61% identity, because the scan used only the human and
     mouse HMMs.
   * Their FR3 "deletion" is the ordinary rabbit germline FR3 length, not a rare indel. Under the human HMM, ANARCI
     puts the gap at 81 or 84 depending on residues 80–84 (KTSS gives 81; KASS, LASS or TASS gives 84). SAbR always
     puts it at 81.
   * Within rabbit VH, ANARCI's gap position alone decides the outcome:
     * gap at 84: 7/7 high-resolution entries disagree (4 antibodies);
     * gap at 81: 0/20 disagree.

   Only 30bg (IGHV1-69, 97%), 9pf9 (IGHV2-5, 97%), 9e89 (IGHV3-7-like, 76%) and the mouse IGHV1 FR1 deletion
   (8gs*, 7eaj/7eak) are real rare indels. "IGHV3-53/3-66" invites a false SARS-CoV-2 public-clonotype reading.
2. **The rule is near-deterministic, not deterministic.**
   * 7kbk (alpaca VHH, 2.09 Å, gap at 84) agrees with ANARCI, while 6dbd (VHH, 1.76 Å, the same NANTVYL motif)
     disagrees.
   * 9ml5 (rabbit EM, FR3 identical to 30ad) disagrees at 85–95 instead of 81–83.
   * So 12 of 13 high-resolution antibodies follow the rule.
3. **Some confirmations are thin.**
   * The gap-at-80+84 pattern has only one antibody, the 2x7l family (5dhv, 2.3 Å).
   * The FR1 gap at 26 has 2 antibodies (5aam and 1ghf at 2.49 and 2.7 Å, plus 5aaw).
   * 9pf9 (human IGHV2-5) has no high-resolution sequence analog. It is supported by both predicted models.
4. **The effect is a lower bound.** Both predictors also reproduce other non-claimed windows:

   | Mode | Further probable artifacts | Revised count |
   |---|---|---|
   | C | 4xnz and 8c8t (FR3 insertion class), 9yfu (FR3 gap at 85), 7m3l (90–95) | up to ~10 of 31 units |
   | F | 5dum; 8idm (IgFold only) | 3–4 of 13 units |

5. **Knock-on effect.** The investigator's rabbit association is partly this artifact. 9bop is one of the 2 rabbit EM
   C_FR3 error units, and one of the 7 rabbit error units.

## Caveats
* The predicted models were trained on PDB structures, but they cannot have memorised the 2024–26 artifact entries
  (9bop, 30ad, 30bg, 9e89, 9pf9). They also did not reproduce older genuine errors from the 2020–21 training era.
* The n values are small by construction: these patterns are rare PDB-wide. The tests are exact, but they are
  effectively a deterministic check rather than an epidemiological association.
* SAbR's structural gap placement is itself a convention. "Artifact" means that disagreement here does not indicate a
  wrong model. It does not mean ANARCI is wrong.
