# Paper-confirmed re-test: are the antibodies themselves ordinary? (deeper README §7)

**Question.** The deeper dive concluded that "the antibodies are ordinary; the models are not". Does that still hold
if we keep only the register errors that conkit-validate flagged in Sanchez Rodriguez et al. (IUCrJ 2024, Table S1,
PDB snapshot April 2022) and that our SAbR/ANARCI scan confirms as first occurrences?

The full-set conclusions come from:

- `deeper/README.md` §7;
- the lenses `seq_novelty`, `struct_unusual` and `literature_dean`;
- the synthesis length rechecks.

## Short answer

**Yes. By sequence, germline and predicted-structure yardsticks, the paper-confirmed antibodies are at least as
ordinary as the full set.**

**Still true (or still null):**

- No excess of first-time germlines, rare CDR lengths, humanisation or rare residues at canonical, Vernier or
  interface positions.
- No excess distance to earlier antibodies once the same antibody is set aside.
- The orientation and H3 base predicted from sequence are normal.
- The paper-confirmed antibodies are rarely represented in X-ray ≤3 Å statistics.
- The deposited models carry distorted VH/VL cores.

**What changed:**

1. **Deposited VH–VL orientation outliers are attenuated.** In EM, 27% of case Fvs are outliers vs 15% of controls
   (OR 2.0, 0.8–5.1). Seven of the 10 case outliers are one Fv model repeated across the dominant Cell Res 2021
   paper (7CZR–7D00). Without that paper the rate is 15% vs 15%. The Mahalanobis excess is a fifth of the rest of
   the deeper set (+0.29 vs +1.44, heterogeneity p=0.017). Outside that paper, every outlier has a distorted core,
   as in the full set.
2. **Rare CDR lengths are *under*-represented.** 15% of case units vs 26% of controls (OR 0.35, 0.12–1.00; EM OR
   0.17, 0.04–0.76). This differs from the rest of the deeper set (heterogeneity p=0.02). The full-set conclusion
   ("rare lengths not more common") still holds; this set is, if anything, more ordinary.
3. **The ImmuneBuilder ensemble-uncertainty signal at the error site holds.** EM +0.16 Å (0.03–0.29), p=0.02,
   matching the full set (+0.16). In the full set this signal was not significant within 2020–22.
4. **Species over-representation is weaker and comes entirely from nanobodies.** All 7 non-human/mouse units are
   VHH: alpaca germline call 21% vs 6%, OR 3.0 (1.0–8.9). Rabbit and bovine are absent (0/34). That absence is
   compatible with chance at this n.

**Conkit's selection shapes the sequence-level picture.** Within the same window, the 34 paper-confirmed units were
compared with the 37 first-occurrence error units that conkit missed. The confirmed units are:

- closer to earlier PDB antibodies: nearest different antibody, OR 2.1 per SD, p=0.012;
- more germline-like: germline V identity, OR 1.7 per SD, p=0.026;
- less mutated: OR 0.59 per SD, p=0.03;
- less often of rare CDR length: 15% vs 49%, OR 0.13, p=0.014;
- never bovine or rabbit: 0 vs 6 units.

So the "ordinary antibody" result is partly *produced* by the paper set's selection. Two explanations fit:

- AlphaFold2 contact maps are more reliable for familiar sequences;
- the conkit-missed set contains more residual detector artifacts at rare lengths and in non-human frameworks.

## Data and design

| Set | Definition | Units (antibody × paper) | EM / X-ray | Papers |
|---|---|---|---|---|
| `paper_strict` (primary) | unit has a chain with a "confirmed first occurrence" Table S1 error (`paperconf/chain_flags.csv.gz`) | 34 | 24 / 10 | 25 |
| `paper_incl` | + copies, near copies, 1 loose-ledger case | 42 | 32 / 10 | 32 |
| `control` | all chains of the unit are `paper_control` (window ≤2022-04-05, entry not in Table S1, no run) | 2,378 | 938 / 1,440 | 968 |
| `rest` | deeper first-occurrence units, not paper-confirmed; synthesis-ledger strict artifacts and amyloid removed; all years | 145 | 124 / 21 | 117 |
| `rest_win` | `rest` released in the paper window (errors conkit missed) | 37 | 29 / 8 | 34 |
| `clean_all` | units with no run of any kind, all years (the full-set control) | 6,151 | 4,036 / 2,115 | — |

**Sequence-level tests** (`s01_units.py`, `s04_seq_tests.py`):

- Unit = antibody × paper (the `unit` key of `chains.csv`). Features are taken verbatim from
  `seq_novelty/tables/units.csv.gz` (earliest chain of the unit) and `literature_dean/tables/units_literature.csv.gz`.
- **M1:** logistic GEE clustered by paper: case ~ feature + year bin + resolution + chain type (+ method when
  pooled).
- **M2:** M1 + nanobody + non-human/mouse + SARS-CoV-2.
- **PW:** M1 with one vote per paper per group.
- **M3:** seq_novelty's restriction to the first structure of the antibody, human/mouse non-nanobody.
- Continuous features are expressed per SD of the window controls.
- Zero cells fall back to Fisher with a Haldane CI.

**Structural tests** (`s02`, `s03`, `s05`):

- Cases: 66 case Fv instances (47 strict: 37 EM, 10 X-ray, 19 papers) and 15 case VHH chains.
- Controls: a random window sample of 150 EM + 150 X-ray control Fvs and 40 + 40 control VHHs, one per entry,
  paired by core centroid as in struct_unusual.
- ABangle: struct_unusual's implementation (Dunbar 2013 frame) on SAbR numbering, with CA coordinates from the RCSB
  model server.
- ImmuneBuilder: run with struct_unusual's runner (model 0, unrefined, plus ensemble error) on all 35 unique case
  sequences and on 140 control Fvs + 80 control VHHs. 3 control sequences were too truncated to model.
- Site uncertainty: each case site (the matched run's IMGT window ±2) is compared with the mean of all same-class
  controls at the same positions.
- Inference: cluster-robust by paper, plus one vote per paper.

**Heterogeneity:**

- *Sequence:* a z-test of the M1 log ORs, strict vs `rest`/`clean_all` (`het_z_p`), and a direct same-era
  case-only contrast, `paper_strict` vs `rest_win` (`het_case`).
- *Structure:* the strict EM estimate vs struct_unusual's non-paper cases vs its controls (EM, 2018–26), re-fitted
  with the same model.

**Power:** `mde_OR80` is the smallest OR detectable at 80% power for the case n and the control prevalence.

**Dominant paper:** 10.1038/s41422-021-00487-9 (7CZR–7D00) contributes 18 of the 37 strict EM Fv instances.
`s07_dominant_paper.py` re-runs the key tests without it.

## Results

Unless stated otherwise:

- "strict" is `paper_strict` vs window controls, M1;
- "full" is the published deeper value, followed by my recomputation on `rest` vs `clean_all` (EM);
- "het" is the strict vs rest z-test.

### VH–VL orientation (struct_unusual §a; README §7)

| Trend (full) | Paper strict | One vote per paper / without dominant paper | Incl. copies | Het | Verdict |
|---|---|---|---|---|---|
| **Mean angles unchanged** (all six n.s.; HC2 −0.93°) | EM: HL +0.35°, HC1 −0.07°, LC1 −0.51°, LC2 +0.90°, dc +0.08 Å (all p≥0.06). **HC2 −1.70° (−2.88 to −0.52), p=0.005.** X-ray (10 Fvs, 7 papers): LC1 −1.6°, p=7e-4; others n.s. | HC2 PW p=0.13. Without the dominant paper: HC2 −1.5°, p=0.13; LC1 X-ray persists (7 papers) | HC2 −1.5°, p=0.003 | HC2 vs rest (−1.03) p=0.45 | **null still null** for 5 of 6 parameters. HC2 is nominally shifted, in the same direction as the full set and rest; not robust to one vote per paper |
| **More outliers** (36% vs 16% outside the 99% ellipsoid, OR 2.7; Mahalanobis 3.36 vs 2.57; rest recomputed +1.44, OR 3.1) | **EM 10/37 (27%) vs 23/150 (15%), OR 2.04 (0.82–5.06), p=0.12.** MDE 3.2. Mahalanobis +0.29 (−0.05 to +0.63), p=0.09. X-ray 0/10 vs 5/150 | PW OR 2.06 (0.60–7.0). Without the dominant paper: 3/20 (15%) vs 15%, OR 1.1 | 22/56, OR 4.1 (1.5–10.9) (copies 7M7E–7S6D add outliers) | Mahalanobis p=0.017 (smaller); outlier OR p=0.46 | **attenuated.** 7 of 10 case outliers are one Fv of one paper |
| **Outliers explained by distorted cores** (undistorted only: +0.10, p=0.67) | Core RMSD to consensus **+0.19 Å (0.10–0.28), p=2e-5** (EM); X-ray +0.18, p=0.006. Undistorted cores only: EM 7/32 vs 15/135, OR 1.8 (0.6–5.2); Mahalanobis +0.22, p=0.08 | Without the dominant paper: **0/15** outliers in undistorted Fvs; the 3 remaining outliers (6LHT, 7C80, 6CMO) all have distorted cores | — | core RMSD p=0.008 (half the rest's +0.38) | **holds** outside the dominant paper. Its Fv is an outlier with a near-threshold core (0.74 vs 0.84 Å), so the "explained" part is not testable there. Distorted cores themselves hold, at half the size |
| **Orientation predicted from sequence normal** (Mahalanobis 1.60 vs 1.41, p=0.24) | EM +0.02 (−0.33 to +0.36), p=0.93. All methods +0.08, p=0.64. X-ray +0.25, p=0.37. Predicted core RMSD +0.003 | PW +0.18, p=0.31; without the dominant paper +0.24, p=0.23 | +0.10 | p=0.22 | **null still null** (CI width ±0.35) |

### CDR lengths (literature_dean; synthesis s03)

| Trend (full) | Paper strict | PW / other | Het | Verdict |
|---|---|---|---|---|
| **Rare CDR lengths not more common** (any CDR length <5% in the X-ray ≤3 Å reference: error rate 4.2% vs 4.3%, OR 0.85, q=0.48; recomputed rest vs clean EM 33.9% vs 28.0%, OR 1.11) | 5/34 (14.7%) vs 26.1%, **OR 0.35 (0.12–1.00), p=0.049**. EM 2/24 vs 29.3%, **OR 0.17 (0.04–0.76), p=0.02**. X-ray 3/10 vs 24%, OR 0.73 | PW 0.42 (0.14–1.24). Incl. copies 0.29 (0.11–0.80). Without the dominant paper EM 0.20 (0.05–0.89) | **p=0.021** vs rest. Same era: paper vs conkit-missed 15% vs 49%, OR 0.13, p=0.014 | **null still null** (no excess). The paper set is in fact *depleted* of rare lengths, unlike the rest, which is most likely conkit selection |
| **Specific canonical lengths** (reused from `../local_hazards/tables/length_tests_paper.csv`; s03 model; error rate by length) | H2-9 vs H2-10 (full OR 2.45): EM 4/103 vs 9/399, adj OR 1.75 (0.56–5.46), MDE 4.7; all methods 1.52 (0.62–3.71). κ L1-17 vs L1-11 (full 3.1–3.6): EM 2/17 vs 6/176, adj OR 3.62 (0.97–13.6), p=0.056; all methods 2.22 (0.62–7.87). λ L1-13 (full 2.6, n.s. without amyloid): no EM λ L1-8/9 case; all methods 1/67 vs 0/108 | One case per paper: H2-9 OR 1.97; κ L1-17 OR 2.8 | H2-9 p=0.50; κ L1-17 p=0.99 | H2-9 and κ L1-17: **same direction, underpowered**. λ L1-13: **cannot test** |

### CDR-H3 base and ImmuneBuilder uncertainty (struct_unusual §b, §e)

| Trend (full) | Paper strict | PW / other | Het | Verdict |
|---|---|---|---|---|
| **Predicted H3 base kinked for VH as often as controls** (97% vs 98%) | **31/31 (100%) vs 133/138 (96%)**. EM 24/24 vs 65/68. X-ray 7/7 vs 68/70 | Deposited bases (context): kinked in 23/31 VH vs 93% of controls, OR 0.13. The slip produces the extended base, as in the full set | — | **null still null** |
| **VHH predicted extended more often** (50% vs 70% kinked, OR 0.31, weak) | 9/14 (64%) vs 56/79 (71%), OR 0.69 (0.07–7.2); EM 4/7 vs 28/39, OR 0.53 | PW 0.78 | p=0.50 vs rest (OR 0.21) | **same direction, underpowered** (5 VHH papers) |
| **Ensemble uncertainty higher at the error site** (0.46 vs 0.32 Å, +0.16, q=0.006; n.s. within 2020–22) | **EM 0.48 vs 0.32 Å, +0.16 (0.03–0.29), p=0.019**, 49 sites in 17 papers. All methods +0.18 (0.04–0.33), p=0.012. X-ray +0.25 (−0.05 to +0.55), p=0.11 | PW EM +0.17 (0.05–0.29), p=0.005. Without the dominant paper +0.22, p=0.008. Incl. copies +0.22, p=0.001 | p=0.50 (rest +0.22) | **holds** (EM); X-ray same direction, underpowered |
| Whole-chain ensemble uncertainty (+0.06) and CDR3 (+0.11, p=0.07) | Chain EM +0.086 (0.018–0.154), p=0.013; X-ray +0.10, p=0.008. CDR3 EM +0.07, p=0.24; X-ray +0.33, p=0.02 | PW chain +0.105, p=2e-5 | chain p=0.63; CDR3 p=0.47 | Chain: **holds**. CDR3: **same direction, underpowered** in EM |

### Germline and sequence novelty (seq_novelty; full set = M2/M3 nulls)

| Trend (full) | Paper strict (all methods; EM) | PW / M3 | Rest recomputed (EM) | Het (z; same-era case-only) | Verdict |
|---|---|---|---|---|---|
| **First-time V gene in the PDB** (1.1% vs 0.5%, q=0.75) | 2/34 vs 3.5%: crude OR 2.1 (0.6–7.8), p=0.34. Year-adjusted M1 5.9 (1.02–34), p=0.048. EM 1/24 vs 0.5%, M1 7.3 (0.8–64) | PW 4.0 (0.6–28), p=0.16. MDE 4.9 | 1.28 (0.18–9.1) | 0.13; 2 vs 0 units | **null still null** (fragile). Two units in two papers (human IGKV1D-17, 7E39; mouse IGKV4-50, 7K66). Only the year-adjusted model reaches p<0.05 |
| **First-time J gene** (0 vs 0.1%) | 0/34 vs 0.6% | MDE 12.9 | 0/124 | — | **null still null** (no power) |
| **First-time V–J pair** (3.8% vs 3.5%) | 4/34 (11.8%) vs 11.4%, OR 1.79 (0.62–5.2); EM 1.24 (0.27–5.6) | PW 2.2 (0.7–6.6). MDE 3.2 | 0.89 | 0.54 | **null still null** |
| **Nearest earlier PDB antibody: no difference after same-antibody and species adjustment** (raw OR 0.69/SD; M3 0.90) | Any earlier V: 0.862 vs 0.894, M1 0.69/SD (0.53–0.90), p=0.006, **M3 0.88 (0.45–1.71)**. Nearest *different* antibody: 0.857 vs 0.854, OR 0.95 (0.67–1.36); EM 1.12 | PW (any) 0.60. PW (different) 0.82 (0.61–1.11) | Any 0.62 (M3 0.91); different 0.70 (M3 0.92) | 0.35 / 0.10; same era: paper cases *closer* than conkit-missed, OR 2.09/SD, p=0.012 | **null still null.** The raw "further from earlier antibodies" reproduces only through the same-antibody term and vanishes in M3, as in the full set |
| Earlier structure of the same antibody (13% vs 33%, OR 0.32, by construction) | 5/34 (14.7%) vs 30.6%, OR 0.40 (0.16–1.01), p=0.053; EM 0.52 | **PW 0.25 (0.09–0.70)**. Without the dominant paper 0.27 (0.09–0.85) | 0.30 | 0.69 | **same direction, underpowered** under M1; holds with one vote per paper. Largely by construction |
| **Germline identity / somatic mutation: no difference after species adjustment** (identity <85%: OR 1.8 raw, M2 0.94) | Identity 0.893 vs 0.879, OR 1.18/SD (0.72–1.95); <85% 29.4% vs 29.3%, OR 0.96 (0.43–2.2). SHM 10.2 vs 11.2, OR 0.85 (0.53–1.37) | M2 identity 1.64; SHM M2 EM 0.51 (p=0.04, *fewer* mutations) | Identity 0.73/SD (M2 1.00); <85% OR 2.0 (M2 1.16); SHM 1.32 (M2 0.93) | Identity p=0.049; SHM p=0.07. Same era: paper cases more germline-like, OR 1.72/SD (p=0.026), fewer SHM 0.59/SD (p=0.03) | **null still null.** Even the raw low-identity association is absent; the paper set is more germline-like than the conkit-missed errors |
| **Humanisation** (0.5% vs 1.3%) | 0/34 vs 2.9% | MDE 5.3 | 0.63 | 0.98 | **null still null** (no power for OR <5) |
| **Rare residue at canonical positions** (26% vs 20%, q=0.59) | 20.6% vs 19.3%, OR 1.05 (0.38–2.9); EM 0.55; X-ray 2.9 (0.7–12.5) | PW 1.31. MDE 2.8 | 1.39 (p=0.10) | 0.65 | **null still null** |
| **Rare residue at Vernier positions** (31% vs 26%) | 29.4% vs 24.0%, OR 1.26 (0.56–2.8); EM 1.89 (0.81–4.4) | PW 1.49; EM PW 2.36 (0.92–6.1). MDE 2.7 | 1.29 | 0.87 | **null still null** (EM point estimate above 1) |
| **Rare residue at VH–VL interface positions** (16% vs 13%) | 17.6% vs 13.8%, OR 1.59 (0.68–3.7); EM 1.91 (0.72–5.1) | PW 1.54; M3 EM 3.6 (1.00–13.2) | 1.15 | 0.75 | **null still null** |

### Species (seq_novelty; README §4.4 and §7)

| Trend (full) | Paper strict | PW / other | Rest recomputed (EM) | Het | Verdict |
|---|---|---|---|---|---|
| **Non-human/mouse germlines over-represented** (27.9% vs 12.8%, OR 2.97; README "OR 3.0") | 7/34 (20.6%) vs 9.2%, OR 2.08 (0.75–5.75); crude 2.7 (1.2–6.1), p=0.034. M2 2.47 (0.94–6.5). EM 5/24 vs 11.3%, 2.21 (0.72–6.8). X-ray 2/10 vs 7.8%, 1.71 | PW 1.52 (0.57–4.0). MDE 3.4. Without the dominant paper 2.25 (0.80–6.4) | 3.05 (1.92–4.85) | 0.56 | **same direction, underpowered** |
| … nanobody (VHH) (full recomputed OR 1.9; seq_novelty hallmarks 2.0, M2 0.65) | 7/34 vs 8.5%, OR 2.02 (0.69–6.0). Alpaca germline call (all 7 non-human units) **OR 3.00 (1.01–8.9), p=0.047**; EM 3.17 (0.91–11.1) | PW nanobody 1.04; alpaca PW 2.06 (0.72–5.9) | nanobody 1.91; alpaca 2.38 | 0.98 / 0.65 | **same direction, underpowered.** Alpaca borderline holds; not robust to one vote per paper (7 units in 6 papers) |
| … rabbit (full OR 3.3–4.2; 7 units, one an artifact) | 0/34 vs 1.3% | MDE 8.0. About 1.7 cases expected at OR 4 (P(0)≈0.18) | 4.07 | 0.36; conkit-missed in window 2/37 | **absent or reversed** (0 cases; compatible with chance) |
| … bovine (full OR 8–11; mostly one lab) | 0/34 vs 0.4% | MDE 16. About 1.4 cases expected (P(0)≈0.25) | 11.0 | 0.57; conkit-missed 4/37 | **absent or reversed** (0 cases; compatible with chance; a single-lab cluster in the full set) |

### Representation in curated statistics (literature_dean T14)

| Trend (full) | Paper strict | PW / other | Rest recomputed (EM) | Het | Verdict |
|---|---|---|---|---|---|
| **Any X-ray ≤3 Å structure of the same V domain: 7% vs 21%** (error rate 1.5% vs 5.0%, OR 0.29) | **3/34 (8.8%) vs 46.2%, OR 0.16 (0.05–0.51), p=0.002.** EM 3/24 (12.5%) vs 30.1%, 0.33 (0.11–0.96), p=0.04. X-ray 0/10 vs 56.7%, Fisher p=2.5e-4 | PW 0.13 (0.03–0.50). Incl. copies 0.30. Without the dominant paper 0.12 | 0.26 (0.12–0.55) | 0.45; same era vs conkit-missed 0.55 | **holds.** This is partly circular (an earlier same-antibody X-ray structure would make many errors "copies") |

## Conkit selection: how it can produce or mask these trends

1. **Conkit missed about half of the in-window first-occurrence errors.** Compared directly (same era, `het_case`),
   the 34 confirmed units differ from the 37 missed units:

   | Measure | Confirmed | Missed | Estimate | p |
   |---|---|---|---|---|
   | Identity to the nearest different antibody | 0.857 | 0.798 | OR 2.09 per SD | 0.012 |
   | Germline identity | 0.893 | 0.805 | OR 1.72 per SD | 0.026 |
   | Somatic mutations | 10 | 18 | OR 0.59 per SD | 0.03 |
   | Rare CDR length | 15% | 49% | OR 0.13 | 0.014 |
   | First-time CDR length combination | 1/34 | 6/37 | — | — |
   | Bovine or rabbit | 0 | 6 | — | — |
   | SARS-CoV-2 | 47% | 16% | — | — |

   - **Interpretation.** Conkit-validate scores deposited contacts against AlphaFold2 contact predictions, which
     are most confident for sequences close to the training data and to common canonical lengths.
   - **Consequence.** The paper set is therefore enriched for ordinary antibodies, and the sequence-level
     "ordinary" verdicts are partly *produced* by the selection.
   - **What the selection masks.** A real novelty or rarity effect (rare lengths, divergent or highly mutated V
     domains, bovine or rabbit frameworks) would be hidden in this set. The rare-length depletion (OR 0.35) is
     the clearest sign.
   - **Alternative.** The conkit-missed set also holds more of our own residual artifacts, which concentrate at
     rare indels and in sparse germline sets (rabbit, bovine; `deeper` §3). Conkit is right not to flag those.
     The two explanations cannot be separated here.
2. **AlphaFold2 training set.** Conkit is near-blind to errors in structures released before 2018-05.
   - Only 2 of the 34 confirmed units are that old (1KEN 2001, 3JBF 2015), vs 8 of 37 conkit-missed in-window
     units and 38% of the controls.
   - Adjustment for year bin handles the composition.
   - "First time in the PDB" questions are therefore tested on 2018–2021 antibodies, which is exactly the
     interesting window (pre-AlphaFold starting models). Nothing in this family needed AlphaFold starting models.
3. **ImmuneBuilder training overlap.** ImmuneBuilder was trained on SAbDab structures to 31 Jul 2021 (≤3.5 Å, no
   internal gaps). Some paper cases may be in its training set, so predictions could partly reproduce the deposited
   (erroneous) model.
   - This would bias the predicted orientation and H3 base towards the deposited model. Yet every case VH is
     predicted kinked although 8 of 31 deposited bases are extended.
   - It would also lower the ensemble uncertainty at the site, so the observed +0.16 Å is, if anything,
     conservative.
4. **Structural tests are less exposed to conkit selection than the sequence ones.** Conkit selects on deposited
   contacts, which the core-distortion and outlier measures also describe. That should *inflate*, not mask,
   deposited-model abnormality. The attenuated orientation-outlier excess is therefore not a selection artifact in
   the obvious direction. It is driven by which papers happen to be in the set (one dominant paper).

## Caveats

- **Small n.**
  - 34 units in 25 papers (24 EM, 10 X-ray).
  - Structural tests use 47 strict Fv instances in 19 papers. 18 of the 37 EM Fv instances come from one paper.
  - The X-ray strata (10 units, 7–9 papers) can only show large effects.
- **Window controls differ from the deeper controls.** They are 2001–2022, 61% X-ray and 46% with an X-ray ≤3 Å
  structure. Year bin, resolution and method are adjusted for.
  - EM window controls have a higher ABangle Mahalanobis (3.05) than the deeper's 2018–26 controls (2.57), so the
    absolute numbers are not comparable across sets. Only the case−control contrasts are.
- **Many tests.** Roughly 25 features × 3 strata × 5 comparisons, with no multiplicity correction. The isolated
  nominal results (first-time V gene p=0.048, HC2 p=0.005, LC1 X-ray, alpaca p=0.047) should not be over-read.
- **Not covered here.** The κ cis-Pro95 and canonical CDR conformation rows of README §7 are model-level
  measures; they are re-tested in `../model_fingerprint`. The specific canonical lengths are reused from
  `../local_hazards`.
- **Germlines.** Human/mouse germline genes are assigned to nanobodies (ANARCI sets); the species call uses all
  species.

## Files

**Scripts (`scripts/`)**

| Script | What it does |
|---|---|
| `aoutil.py` | Loaders, group labels (reuses `model_fingerprint/tables/chain_frame.csv.gz` and the synthesis ledger), statistics |
| `s01_units.py` | Builds `tables/units_frame.csv.gz` (unit frame, reusable) |
| `s02_fv_sample_abangle.py` | Samples case and control Fvs, fetches CA coordinates (cache `scratchpad/paperord/ca`), computes ABangle and deposited H3 base; writes `tables/fv_abangle_window.csv` and `struct_sample.csv` |
| `s03_ib_jobs.py` | ImmuneBuilder jobs (`tables/ib_jobs.csv`); models in `scratchpad/paperord/ib`, run with `deeper/struct_unusual/scripts/s07_ib_run.py` |
| `s04_seq_tests.py` | Sequence tests: `tables/seq_tests.csv`, `seq_tests_compact.{csv,txt}` |
| `s05_struct_tests.py` | Structural tests: `tables/struct_tests.csv`, `struct_tests_compact.txt`, `struct_fv_frame.csv`, `struct_chain_frame.csv`, `struct_site_err.csv` |
| `s06_summary.py` | Key rows per trend: `tables/summary_key_rows.csv` |
| `s07_dominant_paper.py` | Sensitivity without 7CZR–7D00: `tables/dominant_paper.csv` |
