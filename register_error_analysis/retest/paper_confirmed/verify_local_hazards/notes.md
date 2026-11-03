# Adversarial check of the paper-confirmed "local hazards" re-test

This checks `retest/paper_confirmed/local_hazards`. All code here is my own. I reused only these inputs: the cached
SAbR records, `runs_all.csv`, the ledger, `chains.csv.gz`, Table S1, `lh_frame` window-fit values and the
`analysis_units` frame, the last for per-mode features I did not re-extract.

## 1. Case set re-derived (`scripts/v01_rederive.py`, `tables/v01.log`, `tables/v01_cases_rederived.csv`)

**Result: identical.**
* Table S1 has 162 errors on scanned chains, of which 131 have at least 50% of their residues in the V region.
* Status counts are 69 / 16 / 4 / 5 / 1 / 36, the same as the investigator's. Every error matches on status and run.
* The strict set is 67 chains, 38 entries, 34 antibody × paper units, 25 papers (49 EM, 18 X-ray). With copies it is
  87 chains.
* The latest release among case errors is 2022-02-02.
* No Table S1 error overlaps more than one run.

Minor points:
* **No duplicate records.** No (pdb, chain) appears in more than one SAbR output file. The opposite file orders used
  in `build_cases.py` for REC and RUNREC are therefore harmless.
* **5uk4 case mismatch.** Twelve Table S1 rows on 5uk4 use upper-case chain IDs and match the scan only if case is
  ignored. They are the antigen chains: the antibodies are the lower-case chains, and the ranges are residues
  154–179. No case is lost.
* **`paper_control` is not exactly as documented.** It excludes only entries with a Table S1 V-region error of at
  least 50% (`S.pdb`), not "entries anywhere in Table S1". As a result, 363 of the 6,239 control chains come from
  entries that Table S1 flags on another chain or region.
  * In addition, 190 in-window chains were dropped from the controls because they carry some other run.
  * A stricter control set fixes both: entry nowhere in Table S1, and only runs overlapping the mode window ±5
    excluded. It gives the same answer (EM gap OR 5.50 vs 5.36).
* **Snapshot end.** The latest release among Table S1 entries in the scan is 2022-03-30. All 39 conkit-missed
  in-window units were released on or before that date, so "missed by conkit" is not an artefact of the snapshot ending.

## 2. Key numbers recomputed (`v02`–`v06`)

| Number | Investigator | Mine |
|---|---|---|
| EM gap (unbuilt within 5 or N-terminus unbuilt) | 13/26, 50% vs 16.9%, MH 5.36 (2.32–12.4) | identical. One per paper 6.9 (2.4–19.5). Paper-cluster bootstrap CI 1.6–37.5. Leave-one-paper-out min OR 4.17, max p 0.001 |
| Same, **alignment-based definition** (SEQRES residues not modelled; no SAbR labels) | — | 53.8% vs 25.6%, OR 4.02 (1.68–9.65), p=7e-4. One per paper 5.9. Het vs rest p=0.40 |
| EM bottom-quartile fit | 1.06 (0.44–2.54) | identical. Bootstrap CI 0.31–2.54. Conkit-missed in-window EM: 2.40 (1.11–5.19) |
| X-ray fit | 10.2 (2.3–44), n=10 | identical. 9 papers, leave-one-out min 8.9 |
| Mode A, first built residue | median 3 vs 1, MW p=0.013. >3: 22% vs 2.8% | **SAbR label: reproduced. ANARCI position of first built SEQRES residue: median 1 vs 1, p=0.86. >3: 2/9 vs 9.1%, p=0.20. >1: 33% vs 33.5%** |
| D+ CDR3 length, all methods | 17 vs 11, p=0.046 | **Heavy only: 18 vs 15, p=0.44. Within-type percentile sign test 5/8, p=0.73.** Cases are 88% heavy vs 54% of controls |
| Mode C, not Asp81 (EM heavy) | 2/4 vs 4.0%, p=0.010 | 2/4 vs 4.1% (own ANARCI lookup), p=0.011 |
| C-flank, positive shift | 11/13 events | 11/13 events, 12 papers, 10/12 papers mostly positive (p=0.039). **Exact +1 in only 6/13**: 3 are +2, one +3, one +9 (7c80, a mode F run spanning IMGT 30–128) |
| CDR3/FR4 share | 12/36 | 12/36, 95% CI 0.19–0.51. One-sided p vs 0.20 = 0.042 |

## 3. Verdict review

**Pooled gap and "either": holds (agree).**
* Robust to an independent definition of "unbuilt", to the control set, to leave-one-paper-out and to paper
  clustering.
* SAbR-label circularity does not drive the pooled gap: the two definitions agree on 68/69 strict case runs.
* Two chains each give two units: 7e23 B (C, D+) and 7e3c B (A, C). This is handled by the one-per-paper analysis and
  the bootstrap.

**Bottom-quartile fit, EM.**
* "Attenuated" is acceptable, but the headline "does not replicate in cryo-EM" overstates it.
  * The CI includes the full-set OR (1.76–1.85).
  * The minimum detectable OR is 3.0.
  * Heterogeneity p=0.25. Against the conkit-missed in-window errors, z p≈0.17.
* Copies restore it (1.81). The X-ray result is robust.

**Mode A, "N-terminus not built": holds → same direction, underpowered.**
* The SAbR first-label measure is partly the error itself. For an N-terminal run with shift −k, SAbR's label for the
  first built residue sits k above ANARCI's; this is true of 5/9 strict units.
* On the ANARCI/SEQRES measure the strict set shows only 22% vs 9% (p=0.20). Copies included: 41% vs 9% (p=6e-4).
  In the full set (rest) it holds on both measures.
* The note "model starting after IMGT 1 holds, 5/9 vs 24%" is SAbR-label based and has the same problem.

**D+ long CDR3: holds → null still null.**
* The result is a chain-type composition effect.
* In the full set (rest), within chain type there is no difference either (heavy 15 vs 16, p=0.62 in EM).
* The pooled "holds" also needed X-ray pooled with EM.

**D+ other features: holds → only "unbuilt nearby" holds.**
* Unbuilt nearby: all methods p=0.022; EM 3/4, p=0.07. It is part of the pooled gap.
* IGHJ6 was diluted by light-chain controls. Within heavy chains it is 2/7 vs 15%, p=0.29.
* SARS-CoV-2 against 2020–21 controls is 5/8 vs 40%, p=0.28: an era effect, not a local hazard.

**"Absent" verdicts that are really untestable.** The table gives the probability of seeing this few cases if the case
rate equalled the full-set case rate.

| Trend | Observed | P(this few) | Comment |
|---|---|---|---|
| Mouse IGHV5 (mode C) | 0/5 | 0.29 | Only 1 mouse heavy C case |
| Mouse IGKV8 (mode E) | 0/3 | 0.57 | All three cases human |
| Mouse (mode E) | 0/3 | 0.18 | |
| Atypical CDR2 (mode B) | 0/4 | 0.23 | |
| D+ poor local map | n=4 | — | Directions are mixed: site Q-score lower, domain z not |

These are "cannot test", not evidence of absence.

**Full-set nulls mislabelled.**
* **Fv-only.** The full-set trend was itself n.s. (OR 1.45, 0.88–2.37), so the verdict should be **null still null**.
* **SEQRES FR4 truncation.** Null after adjustment in the full set, so also **null still null**.
* **SEQRES N-terminal truncation.** "Absent" is fair: 0/9 against a crude 32% rate gives P=0.03.

**Template CDR3 mismatch: underpowered → cannot test.**
* The only paper-set evidence is the single case 7V2A←6Z3P, which is already in the junction README.
* The quoted p=0.047 treats its 3 chains as independent.

**Starting model, PDB vs none.** p=0.025 holds only at chain level. In EM it is p=0.18 at unit level and OR 1.27,
p=0.79 at paper level. "Cannot test" stands, but the note should not say this part "points the same way" with that p.

**C-flank +1.**
* "Holds" is supported if "+1" means a positive shift. Exactly +1 accounts for only 6/13 events.
* The 7c80 F-mode whole-domain run (+9) should not be counted as a C-flank junction event. Without it the count is
  10/12, still significant.

**CDR3/FR4 share ≥ a fifth.**
* Holds, narrowly: 12/36, one-sided p=0.042 against 20%.
* The 33% is inflated by SAbR/ANARCI confirming CDR3/FR4 Table S1 errors 30/33 times, against 16/43 in the middle of
  the domain. Table S1's own share on these chains is 25%, which still meets "a fifth".

**No change.** Everything else (adjacent CDR, A residues, F, κ/λ lengths, H2-9, canonical pull, protocol, D−)
stands as stated.

## Files

* `scripts/v01_rederive.py`: case set.
* `scripts/v02_gap_independent.py`: alignment-based unbuilt and first built residue per chain.
* `scripts/v03_pooled_stats.py`: MH (RGB CI), one per paper, paper bootstrap, leave-one-paper-out, heterogeneity,
  control variants.
* `scripts/v04_permode_checks.py`, `scripts/v05_modeA_circularity.py`, `scripts/v06_misc.py`.
* `tables/v0*.log`, `tables/v03_pooled.csv`, `tables/v02_gap_frame.pkl.gz`.
