# Paper-confirmed re-test: local hazards at the slipped strand, and how the model was built

**Question.** Do the "local hazard" trends hold when the errors are restricted to the cases confirmed by both detectors?
The trends come from deeper README §4.2, deepdive/README and junction/README. The restricted set is the Table S1
errors (conkit-validate, PDB snapshot of April 2022) that a SAbR/ANARCI run also flags.

**Case sets.** Units are antibody × paper × mode.

| Set | Definition | Units | EM | X-ray | Papers |
|---|---|---|---|---|---|
| `paper_strict` (primary) | first-occurrence Table S1 errors confirmed by SAbR/ANARCI | 36 | 26 | 11 | 25 |
| `paper_incl` | strict + copies, near copies and the loose-ledger case | 45 | 35 | 11 | 32 |

* **Controls.** Paper-control chains: released in the paper window, entry not in Table S1, no run of any origin.
  Each case is compared with controls at the same IMGT window and of the same chain class (stratum = mode × heavy/light).
* **Full-set reference.** I recomputed it with the same code: deeper first-occurrence runs, strict-ledger artifacts
  removed, controls = every chain with no run, all years. The published numbers are quoted as well.
* **`rest`.** The full first-occurrence set minus paper units, all years: 149 units. Its subset `rest_win` (35 units)
  is the in-window errors that conkit missed.

**Statistics.**
* Pooled tests: Mantel–Haenszel OR across strata.
* Paper-clustered version: GEE logistic regression with stratum fixed effects and paper-clustered robust SEs.
* One-case-per-paper version (`opp`).
* Heterogeneity: a z-test of log ORs, paper_strict vs rest; also a case-only GEE (paper_strict vs rest, stratum-adjusted).
* Power: `min_det_OR80` is the smallest OR detectable with 80% power for the given n and control prevalence.

## Scripts (`scripts/`) and tables (`tables/`)

| Script | What it does | Outputs |
|---|---|---|
| `lh01_features.py` | Runs `chain_feats()` from `deepdive/scripts/deepdive.py` verbatim, on every case run at its mode window and on every clean chain at all 7 windows | `tables/lh_frame.pkl.gz` (reusable, chain × window) |
| `lh02_pooled.py` | Pooled hazards (deepdive_overall definitions) | `pooled_hazards.csv/.log`, `analysis_units(_full).pkl.gz` (reusable unit frame) |
| `lh03_per_mode.py` | Per-mode deepdive features | `per_mode_features.csv/.log` |
| `lh04_modes_flanks.py` | Mode distribution, Table S1 region × confirmation, CDR3 flank asymmetry | `mode_distribution.csv`, `s1_region_by_status.csv`, `flank_asymmetry_paper.csv`, `modes_flanks.log` |
| `lh05_template_protocol.py` | Template CDR3 mismatch (reuses `junction/tables/template_cdr3.csv`), starting model and fitting protocol | `template_cdr3_paper.csv`, `protocol_rates.csv`, `template_protocol.log` |
| `lh06_lengths_termini.py` | Canonical pull (literature_dean s06 / synthesis s02 logic), H2-9 / κ L1-17 / λ L1-13 (s03 model), SEQRES truncation, Fv-only | `canonical_pull_*.csv`, `length_tests_paper.csv`, `termini_paper.csv`, `fv_only_paper.csv`, `lengths_termini.log` |
| `lh07_checks.py` | Model end inside FR4, era composition, units per paper | `checks.log` |

## Results

### Pooled hazards (cryo-EM unless stated)

| Trend (full set) | paper_strict | Clustered / one per paper | X-ray (11 units) | Heterogeneity vs rest | Verdict |
|---|---|---|---|---|---|
| Unbuilt residues within 5, or N-terminus unbuilt (48% vs 16%; recomputed 47% vs 19%, OR 4.35) | 13/26 = 50% vs 16.9%. OR 5.36 (2.32–12.4), p=1e-5 | GEE 7.5 (1.4–40), p=0.02. 17 units: 7.2 (2.5–20) | 36% vs 8.6%, OR 6.8 (1.8–25), p=0.005 | z p=0.60; case-only OR 1.18, p=0.79 | **holds** |
| Bottom-quartile site fit (32% vs 25%; recomputed 37% vs 25%, OR 1.76) | 27% vs 25.8%. OR 1.06 (0.44–2.54), p=0.93 | GEE 1.06 (0.42–2.69). One per paper: 1.16 (0.41–3.3) | 80% vs 24.5%, OR 10.2 (2.3–44), p=2e-4 | z p=0.25 | **attenuated** in EM (min detectable OR 3.0); holds in X-ray; all methods 1.9 (0.97–3.7), p=0.08 |
| Either of the above (66% vs 37%) | 65% vs 38%. OR 3.17 (1.39–7.23), p=0.007 | GEE 3.19 (1.42–7.17). One per paper: 3.9 (1.3–11) | 73% vs 29%, OR 6.1 (1.6–23) | z p=0.85 | **holds** |
| Adjacent CDR of atypical length (44% vs 38%) | 42% vs 38%. MH OR 2.7 (0.48–15), p=0.55 | GEE 2.6 (0.5–13) | 55% vs 50% | z p=0.49 | **same direction, underpowered** |

**Caveat on the atypical-CDR feature.** The heavy-CDR3 "atypical" flag is true in 99.6% of heavy controls. Heavy C,
D and E strata therefore carry no information, and the 44/38 headline is mostly a matter of chain-type mix.

**Copies included.** With copies the pattern is the same: gap 4.3 (2.1–8.6); bottom-quartile fit 1.8 (0.9–3.6);
either 4.1 (2.0–8.8).

### Per-mode features (deepdive README table)

Paper strict units per mode: A 11, F 4, B 4, C 5, D+ 8, D− 1, E 3. Values below are EM, with all methods where EM n < 5.

**Mode A**
* **First built residue.** IMGT 3 vs 1 (Mann–Whitney p=0.013).
* **Model starts after IMGT 3.** 22% vs 2.8%, OR 9.8 (2.6–49), p=0.028; one per paper p=0.022.
* **Unbuilt nearby or N-terminus unbuilt.** 56% vs 24%, p=0.044. **Holds.**
* **Heavy Q1.** 60% vs 57% (full set 25% vs 56%). Absent, with n=5; the rest-of-set comparison gives p=0.085.
* **Heavy E5.** 20% vs 2.3%, p=0.12. Same direction.
* **κ I2.** 75% vs 87%. Same direction.
* **With copies,** E5 is 43% vs 2.3% (p=0.001) and κ I2 is 60% vs 87% (p=0.039).

**Mode F** (n=3 EM / 4 all methods)
* Mouse: 67% vs 23%, p=0.13.
* Atypical CDR1: 67% vs 17%, p=0.08.
* Verdict: same direction, underpowered.

**Mode B** (n=4, 3 papers, all methods)
* **Atypical CDR2:** 0/4 vs 2.7%, while the full set has 31% vs 2.8%. Absent (minimum detectable OR 23).
* **Nanobody:** 0/4.
* **Unbuilt nearby:** 25% vs 1.7%, p=0.068.

**Mode C** (n=5)
* **Mouse IGHV5:** 0/5 vs 2.7%, against 22% vs 1.7% in the full set (rest 31%, heterogeneity p=0.28). Absent.
  * The in-window mouse IGHV5-12 C errors (7TE9/7TEB) are among those conkit missed.
* **Not Asp at IMGT 81:** 2/4 heavy chains vs 4%, OR 24 (3.8–141), p=0.010. Holds.
  * Caveat: SAbR places FR3 deletions at 81.
* **Unbuilt nearby:** 80% vs 1.7%, p<1e-6. Holds.

**Mode D+** (n=4 EM, 8 all methods)
* **Unbuilt nearby:** all methods 50% vs 15%, OR 5.6 (1.5–21), p=0.022 (EM 75% vs 28%, p=0.07). Holds.
* **Long CDR3:** median 17 vs 11 across all methods, p=0.046 (EM p=0.33). Holds when all methods are pooled.
* **Poor local map:** domain z 0.23 vs 0.19, and site Q-score 0.34 vs 0.38 (p=0.54). The full set has z −0.82 vs
  +0.40 and Q 0.27 vs 0.45. **Absent** (heterogeneity vs rest p=0.35–0.56).
* **IGHJ6:** 25% vs 8%. Same direction.
* **SARS-CoV-2:** 62.5% vs 16%, p=0.004 (all methods).
* **Fv-only:** 25% vs 32% in EM. Absent.

**Mode D−.** One strict unit (7K66, X-ray). **Cannot test.**

**Mode E** (n=3; all three human, from 7CZR/7D00, 7E39 and 7EY5)
* Mouse IGKV8, mouse and atypical CDR1 are each 0/3, against 12% vs 0.9% for IGKV8 in the full set.
* Verdict: absent, underpowered (minimum detectable OR about 50).

### Mode distribution and the C-flank

**CDR3/FR4 share** (D+, D−, D0 or E).

| Set | Units with a CDR3/FR4 mode | Share |
|---|---|---|
| paper_strict | 12/36 | 33% (44% of papers) |
| rest | 67/162 | 41% (Fisher p=0.45; OR 0.71, 0.34–1.53) |
| full first-occurrence | — | 41% |
| deepdive units | — | 36% |
| Table S1 itself | — | 21% |

* **Verdict: holds.** CDR3/FR4 errors are still at least a fifth.
* **Full mode distribution,** paper_strict vs rest: permutation p=0.50.

**C-flank +1 dominance.**
* Runs on the CDR3 C-flank (IMGT 118–128, excluding origins 1a/4/3) are +1 in 26/28 runs (93%) and 11/13 events
  (85%; binomial p=0.022, 95% CI 55–98%).
* The junction README value is 87%; the rest of the set gives 77% (p=0.72).
* N-flank vs C-flank events: 1 vs 13 (p=0.002).
* **Verdict: holds.**

### Starting model and fitting

**Template CDR3 length mismatch.** EM chains in the window whose PDB starting model contains an antibody domain of
the same type:

| Template relationship | CDR3/FR4 errors (paper_strict) |
|---|---|
| CDR3 length differs by ≥2 | 3/125 |
| Same length or ±1 | 0/220 (p=0.047) |
| Identical antibody | 2/180 |

* The 3 chains are a single independent case, **7V2A ← 6Z3P**, which is already one of the junction README's 3.
* Of the 67 strict case chains, 44 record no starting model at all.
* **Verdict: same direction, no independent replication.**

**Fitting protocol** (EM, window, cases plus paper controls).
* **CDR3/FR4 errors:**
  * rigid-body only 5/437 (1.14%);
  * ab initio 0/156;
  * flexible 0/359.
  * Fisher p=0.33; only 2 papers. Same direction, underpowered.
* **Any mode:** rigid 1.37% vs ab initio 1.28%, OR 1.07 (0.21–4.1); adjusted and paper-clustered 1.08 (0.09–13). Null.

**Starting model recorded.**
* None recorded vs PDB template: 2.16% vs 0.87% (OR 2.5, p=0.025, any mode). This is the same direction as junction
  README's OR 0.56 for a PDB template.
* For CDR3/FR4 errors only: 0.77% vs 0.62% (n.s.).

**AlphaFold starting model.** No in-window chain records one; the first appears after the window, and 290 EM chains
record one afterwards. **Cannot test.**

### Lengths

**Canonical pull** (built length more common in the reference than the true length).
* paper_strict: 6/13 items vs 4.0 expected (p=0.12); one per paper 6/12 (p=0.13); EM 3/7 vs 2.0.
* Excess per item: 0.15, vs 0.14 in the rest of the set (heterogeneity p=0.89).
* Full set: 39/84 vs 27.0 (p=1e-4).
* **Verdict: same direction, underpowered.**

**H2-9.**
* EM: 4/103 exposed units are cases, vs 9/399 of the reference. Fisher OR 1.75 (0.59–5.8); adjusted and
  paper-clustered 1.75 (0.56–5.5), p=0.33.
* Full set: 2.45 (1.58–3.81). Heterogeneity p=0.50; minimum detectable OR 4.7.
* All methods: OR 1.5, p=0.32.
* **Verdict: same direction, underpowered.**

**κ L1-17.**
* EM: 2/17 vs 6/176; adjusted OR 3.6 (0.97–13.6), p=0.056.
* Full set: 3.58. Heterogeneity p=0.99.
* **Verdict: same direction, underpowered.**

**λ L1-13.**
* No EM λ cases with L1 of 8 or 9. All methods: 1/67 vs 0/108.
* The full set without amyloid is n.s. (OR 2.6, p=0.13).
* **Verdict: null still null.** It is essentially untestable.

### Fv-only models and truncated SEQRES

**Fv-only, CDR3/FR4 modes.**
* EM: 1/7 (14%) vs 34%, OR 0.32 (0.04–2.7). All methods: 25% vs 27%.
* Full: deepdive D+ 55% vs 40% (p=0.10) and E 61% vs 40%; recomputed OR 1.45 (0.88–2.37).
* Heterogeneity p=0.13.
* **Verdict: absent or reversed,** underpowered (minimum detectable OR 8.5).

**SEQRES N-terminal truncation, mode A.**
* EM: 0/9 vs 5.5%. All methods: 1/11 vs 4.0% (p=0.36). Given the model start: MH 0.29 (0.03–2.5).
* Full set: 32% vs 7.8% crude, 1.2–3.8 adjusted.
* **Verdict: absent.**
* The underlying factor, the model starting after IMGT 1, still holds: 5/9 vs 24%, p=0.044.
* The paper mode-A cases have full SEQRES with an unbuilt N-terminus.

**SEQRES FR4 truncation, D/E.**
* 0/7 (EM) and 0/12 (all methods) vs 0.3–0.6% of controls.
* The underlying factor, the model ending inside FR4: 1/7 (14%) vs 2.8% of controls (p=0.19), vs 30% in the rest of
  the set (p=0.66).
* **Verdict: absent.** In the full set it was already null after adjustment.

## Selection by conkit (and by SAbR confirmation)

**1. AlphaFold2 training set.**
* Only 2/36 paper_strict units were released before 2018-05, against 8/35 of the in-window errors that conkit missed.
* Old X-ray errors therefore barely enter the paper set. Its 11 X-ray units are mostly 2018–21.
* All the EM-vs-X-ray contrasts here are among recent structures.

**2. Region.**
* In-window errors that conkit missed are 59% CDR3/FR4, vs 33% of paper_strict (p=0.037). conkit under-detects
  CDR3/FR4 slips, so the "≥ a fifth" result is if anything conservative.
* Confirmation by SAbR/ANARCI selects the other way:
  * Table S1 errors touching IMGT ≥105 are confirmed 30/33 times;
  * Table S1 errors at CDR1–FR3 (IMGT 27–104) are confirmed 16/43 times (p=0.006).
* So the mode mix of the paper-confirmed set is shaped by both detectors.

**3. Masking of specific features.**
* The mouse IGHV5 (mode C) in-window cases were conkit misses. This alone can explain its absence in the paper set.
* The model-end and Fv-only signals in the full set are partly the chain-end artifact (removed here) and post-2022
  entries.

**4. Gap and either-hazard associations.**
* These are not produced by conkit's selection:
  * conkit-missed in-window errors show them too (gap OR 2.6, 1.2–5.9; either 3.9, 1.6–9.5);
  * heterogeneity tests are null.

**5. Few papers.**
* 25 papers. Three 2021 SARS-CoV-2 papers contribute 12 of the 36 units (7E39/7E3C 5, the 7CZ* series 4, 7E23 3).
* The paper-clustered and one-per-paper estimates are given for the pooled tests. For the per-mode tests, n is in
  single digits, so "absent" verdicts there are weak: their minimum detectable ORs are 10–50.

**6. Era.**
* The window predates AlphaFold starting models, and most entries record no fitting protocol.
* The protocol tests rest on 2–3 papers.
