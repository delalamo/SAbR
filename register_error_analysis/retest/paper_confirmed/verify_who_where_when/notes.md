# Adversarial check: paper-confirmed re-test, "who / where / when" family

Checked: `../who_where_when/` (notes.md, scripts w01–w07, tables). Own code is in `scripts/`; outputs are in `tables/`.

## 1. Case set: independently re-derived, no discrepancy

`scripts/v01_rederive_cases.py` (log: `tables/v01.log`; table: `tables/v01_rederived_cases.csv`).

**What it does differently from `build_cases.py`:**
* It reads one record per chain, in the order validation → snapshot_rest → new. It uses that record for both the
  V-fraction and the run residues. (`build_cases.py` reads the files in reverse order for the V-fraction, but no
  chain appears in more than one file, so this makes no difference.)
* It keeps every overlapping run, not just the best overlap.
* It checks every run's i0/i1 against imgt0/imgt1.

**Results:**
* 162 Table S1 rows fall on scanned single-domain V chains, and 131 of them have ≥50% of their residues in the
  V region.
* Statuses: 69 confirmed first occurrence / 16 copies / 4 near copies / 5 strict artifacts / 1 loose-ledger
  doubt / 36 not confirmed. This is identical, row by row, to `paper_errors_annotated.csv`.
* Strict set: 67 chains, 38 entries, 34 units, 25 papers. EM: 49 chains, 27 entries, 17 papers. X-ray: 18
  chains, 11 entries. Last release 2022-02-02.
* Every run index resolves (0 mismatches). No Table S1 error overlaps more than one run.

**Edge cases, all correctly excluded:**
* 12 rows of 5UK4 match only case-insensitively. These are the uppercase antigen chains, with ranges 155–179.
* 31 rows lie in constant domains or outside the scanned V record (for example 7LEX 185–199, 7MXL 141–168).

## 2. Recomputed numbers (own code: `scripts/v02_recompute.py`, `scripts/v03_composition_checks.py`)

Unit flags were rebuilt from `chain_flags` + `runs_first_occurrence` + the ledger. Tests are exact Fisher and
year-stratified CMH, plus logit with lab or paper clustering.

| Number | Investigator | Mine | Comment |
|---|---|---|---|
| China-led EM papers, case vs window controls | 9/16 vs 22%, OR 5.5 (1.8–17) | 9/16 vs 59/270; crude 4.6 (1.4–15), year-MH 4.9 (1.7–14), p = 0.004 | Reproduced. Leave-one-paper-out max p = 0.010. 8 distinct last authors. |
| China-led, case vs rest_w | 9/16 vs 4/21, p = 0.036 | Same; year-stratified p = 0.046. rest_w vs controls 18% vs 22%, OR 0.8 | Reproduced |
| Krios, EM entries | 89% vs 88%, OR 1.12 | 24/27 vs 550/628; crude 1.13 (0.33–6.0), year-MH 0.98 | Paper level 14/17 (82%) vs 88%, 0.54–0.67 (same direction). rest_w 0.17–0.18; win_all 0.34–0.35. Reproduced. |
| Capsid, EM entries | 15% vs 6%, OR 3.1 (1.04–9.1), p = 0.043 | 4/27 vs 36/616; Fisher 2.8 (0.67–8.8), p = 0.08; year-MH 3.6, p = 0.059 | The entry-level "significance" depends on the model. Paper level 4/17 vs 18/298, p = 0.023, is reproduced. |
| No funding record, EM entries | 41% vs 18%, OR 2.9 (1.24–6.7) | 11/27 vs 114/628; clustered logit 2.9 (1.25–6.6) | Reproduced at entry level, **but see §3** |
| Citations, EM papers | 0.49 (0.29–0.83) | 0.49 (0.29–0.83), p = 0.008 | Reproduced. Leave-one-out max p = 0.022. Non-China 0.89 (0.36–2.2), China-only 0.20. |
| Lab antibody experience | 0.76 (0.63–0.94), people_network measure | antigen_competition measure: entry 0.71 (0.59–0.86), paper 0.76 (0.61–0.94) | Reproduced with an independent measure. rest_w 1.13 / 1.19 (paper p = 0.02, reversed). win_all 0.94 / 1.02 (null). |
| SARS-CoV-2, EM entries from 2020 | 67% vs 28% vs rest_w, p = 0.017 | 14/21 vs 7/24, p = 0.017 | Reproduced. vs controls (year-MH) 2.65 (1.06–6.6), p = 0.047; paper level null |

## 3. Errors and overclaims found

1. **"No funding record ... holds" is an overclaim.**
   * 10 of the 11 case entries with no funding record are China-led papers. The 11th is 3JBF (2015, a US lab),
     which is not in paper_meta. Among non-China case entries that paper_meta covers, 0 of 9 lack one.
   * Adjusting for China-led last author: 2.67 → **1.39 (0.54–3.6)**.
   * Paper level: 6/17 vs 22%, p = 0.23–0.28. Dropping the 7CZR series: p = 0.07.
   * The investigator attributes the China excess to conkit's selection, so the funding excess is the same
     selection.
   * Verdict should be **attenuated**: China composition, n.s. per paper.
2. **"Krios absent or reversed" is really underpowered.**
   * The minimum detectable OR is 0.29.
   * Entry level: year-MH 0.98 (0.31–3.1). Paper level: 0.54–0.74, the same side as the full set.
   * The CI includes the full-set 0.53.
   * The substantive point stands: the in-window errors conkit missed carry a strong deficit (0.17), and the
     selection-free window estimate is 0.35.
   * Verdict should be **attenuated**: not testable in the subset.
3. **Experience "holds", but only as a product of the same selection that the China verdict discounts.**
   * The gradient is not China composition: non-China case 0.54 with my measure, 0.71 with the investigator's.
   * But the in-window errors that conkit missed go the other way (1.13–1.19), and all in-window errors are null
     for the lab measure (0.94 / 1.02).
   * The headline lists experience among trends that "hold". It should say that the gradient holds in the
     conkit-confirmed subset but is not present in the window as a whole. The label "holds" is literally correct
     for the subset.
4. **Capsid: the headline overclaims.**
   * The full-set conclusion (README 4.3) is "not a target-class effect": OR ≈ 1 after removing artifacts and
     the top co-author networks.
   * The 4 case capsid papers come from 3 networks: Hogle (3JBF), Xiamen (6AD0 and 6LHT), and IBP Wang X. (7C80).
     Three of the four involve non-human antibodies.
   * So the paper set reproduces the crude excess *and* the reason it was discounted. It is not support for a
     capsid trend.
   * The entry-level p = 0.043 depends on the model (Fisher p = 0.08).
   * The capsid excess is window-wide: rest_w OR 8, driven by the Hafenstein network that conkit missed.
5. **Enzyme, incl. copies, OR 20.6 (4.6–92) is a copy artifact.**
   * Ten of the 12 incl.-copies enzyme entries are copies or near copies from four 2021 *Science* papers
     (abe7565, abi8358 ×5, abi8532, abg4998), most likely one lab's series.
   * It should not be cited as directional support.
6. **Factual slips.**
   * "Every 2020–21 case is on a Krios" is wrong: 7KTX (2020, eLife 62611) is on a Polara 300. The
     investigator's own w05 log shows 0.92 for 2020.
   * "Three papers supply 12 of the 27 EM case entries" is also wrong. The 7CZR series, 7CWT/7CWU and 7E39/7E3C
     supply **11**. Five papers with more than one entry supply 15.
7. **Country label.**
   * The full set was null, and the subset gives a nominal new signal. The `absent or reversed` label is the
     nearest enum value, but the text should read "null not reproduced (new signal)".
   * The selection explanation is supported: rest_w 0.8, and case vs rest_w year-adjusted p = 0.046. It is not
     proven, though:
     * about 200 uncorrected tests were run;
     * the win_all estimate (1.9, 0.9–4.0) does not exclude a modest real excess.
8. **Citations "holds" is fine, but it is entirely a within-China contrast.**
   * China-led: 0.20 (case) and 0.16 (rest_w). Non-China: 0.89 (n = 7).
   * Unlike the China share, this is not created by conkit's selection: the errors conkit missed show the same
     pattern.

## 4. Verdicts I accept

* Competition ±1/6/12 months, race, SARS-CoV-2 (one series), early race (descriptive), fewer competitors, niche,
  first-of-family.
* Teams new: the lens-flag 2.64 has the same sign as the full set's ≤2021 lens-flag 2.23, and the minimum
  detectable OR is 3.8, so "null still null" and "underpowered" cannot be told apart.
* Entity scans, co-author links, persistence and carrier, authors per antibody, journal, funders (the NIH dip is
  China), open access, timing, processing.
* Local resolution and the fitting protocol (cannot test), entry size, software, starting model (the AlphaFold
  part cannot be tested).
