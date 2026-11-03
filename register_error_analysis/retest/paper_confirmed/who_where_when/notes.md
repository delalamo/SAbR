# Paper-confirmed re-test: people, places, publication, antigen and competition

**Question.** The deeper dive (README §4.3, §4.4 and §6) found trends about who built the erroneous models, where,
when, for which antigen and under what competition. Do those trends hold if the errors are restricted to the Table S1
errors (conkit-validate, Sanchez Rodriguez et al. 2024, PDB snapshot of April 2022) that SAbR/ANARCI also confirms?

**Short answer.**
* **The competition nulls stay null.** Other labs on the same antigen within ±1, ±6 or ±12 months give OR 0.99–1.16.
  * The early SARS-CoV-2 race looks exactly as it does in the full set: the first 25 entries are clean, and 9 of
    entries 26–50 are paper cases from two papers.
* **The people nulls mostly stay null.**
  * No lab, institution, facility or co-author group recurs beyond chance in the cryo-EM scan.
  * One institutional co-author cluster links two capsid papers.
* **The publication and timing nulls stay null.**
* **Several trends hold or strengthen,** but the set is small (16–18 EM papers or studies):
  * lower citations;
  * less lab and team antibody experience;
  * capsid targets;
  * the first antibody structure of an antigen family;
  * "no funding record".
* **Two results do not hold:**
  * The Krios/300 kV deficit is absent: 89% vs 88% on a Krios. The in-window errors that conkit missed carry the
    whole deficit (56% vs 88%).
  * The country null is broken: 9 of 16 paper-case EM papers are China-led, vs 22% of controls.
    * This excess is specific to the paper-confirmed subset. Only 4 of the 21 in-window errors that conkit missed are
      China-led (Fisher p = 0.036).
    * It is a property of which errors conkit flagged, not of the error population.
* **Teams new to antibody cryo-EM.** The full-set effect is confined to 2022–23, and the window ends before then.
  * The lens's name-collision flag gives OR 2.6 (0.9–7.6).
  * It falls to 1.8 once China-led and SARS-CoV-2 studies are adjusted for, and to 0.7 with the all-names flag.
  * This reproduces the full set's ≤2021 picture: a flag-dependent null.

## Sets and method

| Set | Definition | EM | X-ray |
|---|---|---|---|
| `case` (primary) | unit contains a paper_case_strict chain | 27 entries / 17 papers / 18 studies | 11 entries / 8 papers |
| `case_i` (sensitivity) | + copies, near copies and the loose-ledger case | 41–42 entries / 23 papers | 11 entries |
| window controls | every V chain is paper_control: released ≤2022-04-05, not in Table S1, no run of any kind | 616–628 entries / 284–298 papers or studies | 706–834 entries |
| `rest_w` | in-window ledger-clean first-occurrence errors that are not paper cases (conkit missed them, or S1 did not confirm) | 30–32 entries / 21–24 papers | 8 |
| `win_all` | `case` ∪ `rest_w`: every in-window first-occurrence error (removes conkit's selection) | 57–59 entries / 37–42 papers | 19 |
| `rest` | every ledger-clean first-occurrence error that is not a paper case, all years, vs all clean units | 120 entries / 92 studies | 17–19 |

**Units.** The unit follows each lens:

| Lens | Unit | Source |
|---|---|---|
| Competition and target class | Entry | `verify/antigen_competition_no_competitive_pressure_confounding/entries_frame.csv.gz` |
| Experience and people | Study | people_network cache plus the two new-team verifier frames |
| Publication and timing | Paper | `paper_meta/paper_analysis_table.csv`, entries deposited from 2018 |
| Microscope, processing, size and software | EM entry | `pdb_meta/tables/em_entry_analysis_frame_with_amyloid.csv.gz`; per-chain software fields in `chains.csv.gz` |

Every entry-level test is repeated at paper level, one row per paper (`level = paper`).

**Statistics.**
* **Models:** logit with the lens's covariates.
  * Year category: ≤2019 / 2020 / 2021+ in the window; five categories for all-years groups.
  * Resolution and size.
  * Cluster-robust SEs by paper (entry level) or by last author (paper and study level).
* **Sparse cells:** exact conditional OR, with a CMH or Fisher p.
* **Heterogeneity:** z-test of log ORs, `case` vs `rest` and `case` vs `rest_w`.
* **Power:** minimum detectable OR at 80% power for binary exposures (`summary_all_tests.csv`).
  * These ignore clustering, so they are optimistic.

## Results by trend

`full` is the full-set estimate and its source. Estimates are EM unless marked.

### Competition and antigen (README §6, §4.3)

| Trend | Full set | Paper-strict | Incl. copies | One per paper / clustered | Heterogeneity | Verdict |
|---|---|---|---|---|---|---|
| Other labs on the same accession, ±1/6/12 months (per doubling) | Pre-treatment adjusted 1.16 / 1.09 / 1.07; ±12 months unadjusted 0.94 (verify comp-conf) | ±1: 1.16 (0.78–1.71); ±6: 1.09 (0.82–1.46); ±12: 1.07 (0.82–1.41); adjusted 1.08 (0.82–1.42); n = 27 / 616 | 0.96–0.98 | Paper level, ±12: 0.94 (0.77–1.15), 17 / 298 | vs rest p = 0.40, vs rest_w p = 0.16 | null still null |
| Same family, ±12 months | 0.895 → 1.03 adjusted | 0.99 (0.74–1.34) | 0.89 | Paper level 0.87 (0.70–1.09) | p = 0.54 | null still null |
| Race: another lab within ±1 month | 0.90; adjusted 1.50 (0.98–2.30) | 52% vs 45%, OR 1.31 (0.34–4.99); adjusted 1.40 | 0.70 | Paper level 0.72 (0.24–2.15) | p = 0.71 | null still null (minimum detectable OR 3.1) |
| Competition, X-ray | 1.03–1.21, n.s. | ±12: 0.97 (0.70–1.34); race 0.89 (0.19–4.1); 11 / 706 | same | — | p = 0.5 | null still null |
| SARS-CoV-2 vs other | 6.7% vs 6.4%, OR 0.91; pre-treatment adjusted 1.72 (0.99–2.97) | 52% vs 32%, OR 2.78 (0.59–13.0); adjusted 3.23 (0.54–19) | 1.28 | Paper level 1.60 (0.51–4.99) | vs rest p = 0.11, vs rest_w p = 0.07 | null still null (the entry-level excess is one 7-entry series) |
| Early SARS-CoV-2 race | First 25 EM entries clean; entries 26–50 hold 10 errors, 7 from one paper | First 25 clean. Ranks 26–50 hold 9 paper-case entries from 2 papers (7CWT/7CWU and the 7CZR–7D00 series). Paper cases at ranks 34–49, 99–102, 151 and 190; none after 2021-08. | same | — | — | holds (descriptive): the same "race" signal is one or two papers |
| Fewer competitors (paper_meta, any other-lab DOI ±180 days) | 0.47 → class-adjusted 0.60; family level 0.75 n.s. | Accession 62% vs 74%, OR 0.55 (0.16–1.89); family 0.32 (0.07–1.41); class-adjusted 0.48 / 0.21; 16 / 281 | 0.40 / 0.25 | Lab-clustered | vs rest p = 0.94 / 0.27 | same direction, underpowered |
| Niche antigen in 2020–21 (no other lab ±12 months; non-SARS-CoV-2; paper level) | OR 8.3 (2.2–31) | 4 of 60 niche vs 1 of 71 contested; adjusted 5.1 (0.49–53) | 5.1 | — | p = 0.79 | same direction, underpowered (5 case papers) |
| First antibody structure of the family | 2.2 → 1.3–1.4 n.s. adjusted | 30% vs 11%, OR 4.0 (1.2–13.6); with capsid/enzyme and lab experience 2.9 (0.84–9.8); X-ray 5.6 (1.6–19) | 2.9 | Paper level 4.1 (1.3–12.6) | vs rest p = 0.29 (paper level 0.07) | same direction, underpowered once adjusted (crude holds) |
| Capsid targets | Entry OR 4.7 → about 1.0 after artifacts and two groups removed | 15% vs 6%, OR 3.1 (1.04–9.1); paper level 4/17 vs 18/298, OR 5.0 (1.7–14.6) | 2.2 | Shared co-author cluster counted once: 3/16, OR 3.6, p = 0.08; human antibodies only 2.8, p = 0.34 | p = 0.79 | same direction, underpowered at the conservative unit (artifacts are already excluded; 3 independent groups) |
| Enzyme targets | 3.8 → EC definition 1.4–1.9 n.s. | 7% vs 2%, OR 4.4 (0.67–29); paper level 2/17 vs 6/298, OR 6.5, p = 0.06 | 20.6 | — | p = 0.96 | same direction, underpowered |
| Fiducial-only entries (protective) | 1.0% errors, OR 0.17 | 4% vs 12%, OR 0.26 (0.03–2.5) | 0.35 | — | p = 0.56 | same direction, underpowered |

### Who: experience, people, places (README §4.3, §6)

| Trend | Full set | Paper-strict (studies, EM; 18 / 298) | Incl. copies | One per lab / sensitivity | Heterogeneity | Verdict |
|---|---|---|---|---|---|---|
| New team, lens flag (collision-prone surnames ignored) | 1.99 (1.36–2.91), confined to 2022–23 (4.7–8.4); ≤2021 0.96 (all names) to 2.23 (lens flag) | 56% vs 33%, OR 2.64 (0.92–7.6), p = 0.07 | 2.08 | One study per lab: median OR 1.81, p = 0.55. + China + SARS-CoV-2: 1.81 (0.68–4.85). No-nanobody studies 3.7 (1.05–13); nanobody studies 0.86. | vs rest p = 0.51 | null still null for the window. The window (≤ early 2022) is outside the 2022–23 period that carries the full-set effect. As in the full set ≤2021, the excess depends on the name rule. |
| New team, all-names flag / co-author identity rule | ≤2021: 0.96 / 1.15–1.26 | 13% vs 19%, OR 0.68 (0.15–3.0) / 1.47 (0.46–4.6); 15 / 275 | 0.57 / 1.18 | Non-China-led studies 1.86 (0.31–11) | p = 0.11 | null still null |
| No prior cryo-EM entry on the team | 1.77 (1.16–2.68) | 39% vs 16%, OR 3.46 (1.25–9.6) | 2.88 | + China: 1.81 (0.63–5.2); non-China 0.94 | p = 0.11 | attenuated: the excess is China-led studies under the collision rule |
| Most experienced author, prior antibody cryo-EM entries (per doubling) | 0.89 (0.81–0.97) | 0.61 (0.45–0.81), p = 7e-4; all names 0.78 (0.62–0.98) | 0.67 | + China: 0.66; non-China 0.74 (0.54–1.01) | vs rest p = 0.006, vs rest_w p = 0.01 | holds (stronger than the full set) |
| Last author's lab, prior antibody entries (per doubling) | 0.90 (0.83–0.98); 0.88 in antigen_competition | 0.76 (0.63–0.94), p = 0.009 | 0.76 | Non-China 0.71 (0.52–0.97); + China 0.80 (0.63–1.02) | vs rest_w p = 0.009 | holds. The in-window errors that conkit missed show none (1.06). |
| Multi-lab hub person (core-staff proxy) | 0.67 (0.39–1.13) | 28% vs 27%, OR 1.16 (0.38–3.5) | 1.69 | — | p = 0.23 | null still null |
| Individuals, co-author groups, labs, institutions, facilities (entity scan, exact stratified, BH within class) | Nothing at q < 0.1 | EM: smallest q is 0.15 (person), 0.19 (non-last author), 0.20 (co-author group), 0.30 (lab), 0.33 (institution), 0.38 (any facility). No lab has more than 1 case study. | same | EM + X-ray, studies merged across methods: smallest person q = 0.04. One institutional co-author cluster appears on 2 of its 4 studies: two capsid papers, 6AD0 (2018) and 6LHT (2019), with different last authors. | — | null still null (EM, lens definition). One institutional cluster in the all-method scan. |
| Co-authorship links between case studies of different labs | 0.85× expected | 4 vs 2.1 expected (p = 0.19); 1 vs 1.8 with collision-prone names dropped | — | — | — | null still null |
| Country of the last author / any author | Null (region p = 0.24) | China-led: 9/16 EM papers vs 22%, OR 5.5 (1.8–17). Entity scan q = 0.013. USA 25% vs 50%. X-ray 1/6. | 3.8 | In-window errors that conkit missed: 19%, OR 1.0. All in-window errors: 35%, OR 2.2 (0.96–5.0). | vs rest p = 0.006 | absent or reversed. A selection effect: case vs rest_w 9/16 vs 4/21, p = 0.036. |
| Lab persistence (same lab had an earlier error study) | 2.19 → 1.69; 1.04 if the earlier error was on another antigen | Crude 36% vs 36%; adjusted 7.4 (0.98–56), unstable; 11 / 170 | 9.4 | — | p = 0.16 | null still null (crude identical; the adjusted estimate is not interpretable) |
| Carrier (co-author on an earlier error study of another lab) | 1.02 | 1.27 (0.20–8.1) | 1.37 | — | p = 0.72 | null still null |

### Publication and timing (README §4.4, §6). Paper level, EM; 16 case papers vs 271–284 window control papers.

| Trend | Full set | Paper-strict | Incl. copies | Sensitivity | Heterogeneity | Verdict |
|---|---|---|---|---|---|---|
| Citations, log1p per year | 0.68 (0.54–0.86), between-lab only | Median 2.1 vs 2.7, OR 0.49 (0.29–0.83), lab-clustered, p = 0.008. Percentile 0.35 vs 0.62. | 0.58 (0.38–0.88) | + China 0.55 (0.34–0.90); + SARS-CoV-2 0.43; non-China papers 0.89 (n = 7); all in-window errors 0.58 (0.36–0.95). Within-lab not testable (8 labs). X-ray 0.93 (6 papers). | vs rest p = 0.17 | holds (EM). Partly China composition; between-lab only, as in the full set. |
| Authors per antibody built | 0.70 per doubling | 0.73 (0.43–1.25); + China 0.58 (0.34–1.00) | 0.62 (0.40–0.96) | — | p = 0.91 | same direction, underpowered |
| Journal group / tier | Heterogeneity p = 0.36; tier null | χ² 21.6 on 16 df, permutation p = 0.16. CNS 0.67, sister journals 0.30, Nat Commun 1.58, mega-journals 1.83, all n.s. Cell Research has 3/16 case papers. | — | — | — | null still null |
| Funders | All null; DFG OR 5 (q = 0.052, chance) | NIH 38% vs 59%, OR 0.29 (0.09–0.88) → 0.60 n.s. with China. HHMI, Gates, Wellcome, Chinese, ERC, MRC, DFG, industry all n.s. | NIH 0.41 | — | NIH vs rest p = 0.04 | null still null (the NIH dip is China composition) |
| Open access | Null | 81% vs 68%, OR 1.96 (0.54–7.1) | 0.86 | — | p = 0.07 | null still null |
| Preprint; preprint → publication; preprint within 90 days of deposition | All null (preprint ≤90 days OR 2.6, q = 0.41) | Preprint 25% vs 34%, OR 0.63 (0.19–2.2). Only 4 case papers have preprints (4/4 within 90 days vs 75%). | — | — | — | null still null |
| Deposition → publication, review time, review <60 days, acceptance → publication | Null | 0.76 / 0.56 (0.30–1.06) / 0.96 / 3.3 (0.6–18) per 100 days or per flag | — | — | — | null still null |
| Hold time; structures added after submission; release before the paper | Null | 0.75 / 1.28 / 0.91 | — | — | — | null still null |
| Letter or brief article type | 6.4% vs 4.4%, null | 12% vs 5%, OR 3.4 (0.96–12) | 3.1 | All in-window errors 1.24 | — | null still null (borderline) |
| No funding record in the PDB (pdb_meta) | OR 2.4, 2020–21 only, not paper-weighted | Entry: 41% vs 18%, OR 2.9 (1.24–6.7); paper level 2.0 (0.71–5.8) | 2.3 | All in-window errors 2.3 (1.17–4.6) | p = 0.43 | holds at entry level; same direction, underpowered per paper |

### Map, processing, size, software (README §4.4, §6; deepdive; junction). EM entries, 27 cases vs 628 window controls.

| Trend | Full set | Paper-strict | Incl. copies | Paper level / sensitivity | Heterogeneity | Verdict |
|---|---|---|---|---|---|---|
| Titan Krios before 2022 | 67% vs 78%, OR 0.53 (0.34–0.83); clustered in a few facilities | 89% vs 88%, OR 1.12 (0.33–3.8). ≤2019: 2 of 6 case entries are not on a Krios (2 Polara, 1 Tecnai F30 across all years); every 2020–21 case is on a Krios. | 0.98 | Paper level 0.74 (0.19–2.9). In-window errors that conkit missed: 56% vs 88%, OR 0.18 (0.07–0.46). | vs rest_w p = 0.019 | absent or reversed. Power is low (minimum detectable OR 0.29), but the missed half carries the deficit. |
| 300 kV | 79% vs 86%, OR 0.59 | 100% vs 91% | 3.3 | rest_w 0.24 | — | absent or reversed |
| Processing: local/focused refinement, DeepEMhancer, density modification, cryoSPARC/RELION, particles, K2, EM start-up model | All null | 1.13 / 0 of 27 / 0 of 27 / 0.23 / 3.9 / 0.77 / 0.50 / startup from PDB 0.14 (0.02–1.2), all n.s. | — | — | — | null still null |
| Half maps deposited | 62% vs 76%, OR 0.58 (n.s. once mandatory) | 15% vs 40%, OR 0.25 (0.06–1.1), p = 0.07 | 0.22 (0.05–0.93) | — | p = 0.15 | same direction, underpowered |
| Map-intrinsic local resolution at the site | 48th percentile; 11% in the worst quartile (null to better) | Only 4 case sites in 3 papers have EMDB local resolution (median 16th percentile) | — | rest_w 38 sites: 59th percentile | — | cannot test (85% of case entries have no half maps) |
| Entry size: total mass per doubling | 0.69 (p = 0.005) | 0.85 (0.32–2.3) | 0.89 | Paper level 0.65 (0.33–1.3) | p = 0.69 | same direction, underpowered |
| Antibody <15% of mass; membrane keywords; >500 kDa | 0.57; 0.47; null | 0.27 (0.07–1.05); 0.77; >500 kDa 41% vs 19%, OR 5.2 (1.06–25) | — | >500 kDa: all in-window errors 1.14 | >500 kDa vs rest p = 0.014 | same direction (first two). >500 kDa reversed by 3 capsid entries. |
| Fewer EMDB metadata fields | 371 vs 412 | 0.94 per 10 fields (0.87–1.01) | — | — | — | same direction, underpowered |
| Recorded Coot / Phenix / Rosetta / ISOLDE (fewer errors in 2018–21) | 4.7% / 7.6% / 1.5% / 0% vs 9.6% base | Coot 0.49 (0.09–2.7); Phenix 0.50 (0.17–1.5); Rosetta 0 of 27 vs 10% (exact CI 0–1.35); ISOLDE 0 of 27; any building tool 0.30 (0.05–1.9) | 0.41 / 0.66 / 0 / 0 | All in-window errors: any building tool 0.25 (0.09–0.75). X-ray: Coot 3 of 11 vs 5%, OR 3.75 (1.0–14). | p = 0.54–0.89 | same direction, underpowered (EM). X-ray reversed (n = 11). |
| Starting model: PDB template vs none; AlphaFold | Junction (CDR3/FR4): AlphaFold 0.34%, PDB 1.09%, none 1.70%; 65/70 κ error chains have none | PDB template 15% vs 27%, OR 0.43 (0.11–1.7); none recorded 85% vs 73%, OR 2.3 (0.6–8.9). AlphaFold: 0 of 1,610 window entries. | 0.68 | — | p = 0.50 | PDB vs none: same direction, underpowered. AlphaFold: cannot test (no AlphaFold starting models exist in the window). |
| Fitting protocol: rigid-body only | D−: 30% vs 11%; enzyme lens: 28% vs 10% | 11% vs 14%, OR 0.77 (0.18–3.3) | 0.62 | Only 1 D− unit in the paper set | — | cannot test the mode-specific claim (1 D− unit); pooled: absent |

## How conkit's selection bears on these trends

The paper-confirmed set is not a random half of the window's errors. Compared with the 30–32 in-window error entries
that it does not contain (`rest_w`), it is:
* more China-led (9/16 vs 4/21 EM papers, p = 0.036);
* more SARS-CoV-2 (67% vs 28% of EM entries deposited 2020+, p = 0.017; at paper level 55% vs 28%, p = 0.24);
* from less experienced teams and labs (heterogeneity p = 0.01);
* more often on a Krios (89% vs 56%);
* more often mode A, and less often CDR3/FR4 (`../local_hazards/tables/mode_distribution.csv`);
* almost never before 2018-05 (AlphaFold2 training set): 2 entries, 1KEN (2001) and 3JBF (2015), vs 8 of the
  conkit-missed set.

**Trends this selection can produce or mask.**
1. **The China excess, the low-NIH and low-USA shares, and part of the citation deficit.** These come from the
   China-led SARS-CoV-2 and capsid papers that conkit flagged. The citation OR is 0.89 in non-China papers.
2. **The stronger experience gradient.** Teams with less antibody cryo-EM experience are concentrated in the flagged
   half. Without the selection (`win_all`), the most experienced author gives 0.82 per doubling and the lab gives
   0.94, close to the full set.
3. **The Krios null.** The missed half carries the full-set Krios deficit (Tecnai Arctica and Polara capsid labs).
   Without the selection, Krios is 71% vs 88%, OR 0.35 (0.16–0.74), so the full-set trend is present in the window.
   conkit's selection masks it.
4. **SARS-CoV-2 and capsid enrichment.** The selection can make SARS-CoV-2 look enriched. All in-window errors give
   OR 1.12 for SARS-CoV-2 (0.41–3.0), which matches the full-set null. Capsid stays enriched without the selection:
   6.2 (2.4–16).
5. **Small numbers.** Errors in the AlphaFold2 training set (pre-2018-05) are almost absent, so the ≤2019 stratum
   rests on a handful of entries.

## Caveats

* **Small n.** There are 16–18 EM papers or studies, and three papers contribute 12 of the 27 EM case entries
  (7CZR–7D00 ×7, 7CWT/7CWU, 7E39/7E3C). Paper-clustered SEs and one-per-paper versions are given throughout.
  Minimum detectable ORs for binary exposures are about 3–7 at the paper level.
* **Coverage.**
  * paper_meta covers only entries deposited from 2018. Five case papers (1KEN, 3JBF, 6BPE, the two unpublished
    X-ray entries) have no publication metadata.
  * Last-author country is missing for unpublished studies.
* **Experience flags.** The flags inherit the lens's name-key issues. The `_na` flag treats collision-prone surnames
  as inexperienced, which inflates "new team" for China-led studies.
* **Multiple testing.** About 200 tests were run with no global correction. The nominal positives that are not
  replicated by a full-set trend (big complexes, letter type, Coot in X-ray) should be treated as chance or
  composition.
* **People.** People are reported only as pseudonyms (P#### as in the people_network lens, I### for institutions).

## Files (`scripts/`, `tables/`)

| Script | What it does | Outputs |
|---|---|---|
| `ww_common.py` | Case, control, rest and win_all flags at entry, paper and study level; logit, MH and exact tests, power, heterogeneity | — |
| `w01_antigen_competition.py` | Competition ±1/6/12 months, race, SARS-CoV-2, capsid, enzyme, fiducial, first-of-family, niche 2020–21, paper_meta fewer competitors | `w01_antigen_competition_tests.csv`, `w01_entry_frame.csv.gz`, `w01.log` |
| `w02_sars2_race.py` | Rank of each SARS-CoV-2 EM entry; quarterly time course; SARS-CoV-2 share of case vs rest_w | `w02_*.csv`, `w02.log` |
| `w03_people.py` | Experience (lens and verifier flags, China sensitivity), lab persistence, carrier, entity scans, co-author link permutation | `w03_people_tests.csv`, `w03_entity_enrichment_pseudonymised.csv`, `w03_entity_enrichment_summary.csv`, `w03_network_links.csv`, `w03.log` |
| `w04_paper_meta.py` | Citations, authors, journals, funders, open access, country, preprints, timing | `w04_paper_meta_tests.csv`, `w04_journal_groups.csv`, `w04_paper_frame.csv.gz`, `w04.log` |
| `w05_map_model_meta.py` | Microscope, processing, size, funding record, local resolution, software, starting model, fitting | `w05_map_model_meta_tests.csv`, `w05_local_resolution_site.csv`, `w05.log` |
| `w06_summary.py` | One row per test with all groups, heterogeneity p and minimum detectable OR | `summary_all_tests.csv` |
| `w07_capsid_enzyme_checks.py` | Capsid and enzyme by species, with the co-author cluster counted once | `w07_capsid_enzyme_checks.csv`, `w07.log` |
