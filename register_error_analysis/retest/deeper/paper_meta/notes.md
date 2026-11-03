# Paper and publication metadata: do the error papers share anything?

## Data

**Papers.** Every DOI cited by an antibody entry deposited from 2018 onward: 1,576 DOIs, of which 1,161 are cryo-EM.

**Sources** (all official APIs, cached under `scratchpad/deeper/cache/paper_meta/`):
* Europe PMC core records;
* Crossref;
* PubMed efetch (received, revised and accepted dates);
* bioRxiv/medRxiv `pubs` and `details` endpoints;
* Europe PMC preprint (PPR) records.

**Coverage:**

| Field | Coverage |
|---|---|
| Europe PMC and Crossref records | 99.7% |
| Received and accepted dates | 91% |
| Author lists | 99.9% |
| Last-author country | 97% |
| Preprint found | 28% (440 papers: 379 bioRxiv, 42 Research Square) |

**Unit of analysis.** The unit is the paper's cryo-EM entries deposited from 2018 onward:
* **Error paper:** at least one of those entries has a first-occurrence error. There are 127 such papers (110 published).
* **Clean paper:** none of those entries has a run of any kind. There are 1,300 such papers (991 published).
* **Excluded:** 32 papers whose runs are all copied or artifact runs, and 10 amyloid-fibril papers (see below).

**Statistics.**
* Each feature is fitted in a logit model: `error ~ x + period + log2(n antibody V regions) + median resolution`.
* **Why the size term matters.** A paper's chance of containing at least one error grows with the number of antibodies it builds: OR 1.76 per doubling, q = 5e-7. The rate per antibody is flat at about 4%.
* **Checks on the positives:**
  * a fuller set of covariates (SARS-CoV-2, viral antigen, tool paper, CNS and CNS-sister journals);
  * a unit-level GEE clustered by paper;
  * estimates within each period.
* **Multiple testing.** One Benjamini-Hochberg correction covers all 104 tests (`all_tests_bh.csv`).

## 0. A data problem that every lens should fix: light-chain amyloid fibrils

**What they are.** Fifteen cryo-EM entries are AL amyloid fibrils of λ light chains:
* 6IC3, 6Z1O and 7NSL (Fändrich, Schönland and Hegenbart, Ulm/Heidelberg);
* 9FAA, 9FAB and 9FAC (Ricagno);
* 9ELS (Saelices);
* 9LIV, 9LIW, 9LIX, 9LIY, 9LJ0, 9LJA and 9LJB (Li D.);
* 9V91 (Huang H.).

**All 15 are flagged as first-occurrence errors.** They make up:
* 12 of the 183 cryo-EM error units;
* **77 of the 488 first-occurrence runs (16%)**. These runs span almost the whole domain, for example 6IC3 from IMGT 16 to 122 with Δ −48.

They are labelled as mode A_FR1Nterm (8 entries), F_CDR1 (2) and "other" (6). Before the exclusion, this was the strongest link between authors in the whole dataset: all 3 Fändrich/Schönland papers were "error" papers (q = 0.06).

**Why they are not errors.** These chains are not in the immunoglobulin fold. Structure-based IMGT numbering of a fibril has no meaning, so these are detector artifacts, not register errors. **They should be removed from every analysis** (`build_base.py` flags them as `amyloid_artifact`).

## 1. Speed and pressure: nothing

No difference on any measure of speed or pressure (error vs clean, published cryo-EM papers):

| Measure | Error | Clean | Adjusted p |
|---|---|---|---|
| Preprint exists | 32.7% | 30.4% | 0.79 |
| Preprint → publication (median) | 211 d | 179 d | 0.10 |
| Deposition preceded the preprint | 69% | 65% | 0.97 |
| First deposition → publication | 178 d | 194 d | 0.55 |
| Submission → acceptance (review time) | 144 d | 149 d | 0.86 |
| Review time < 60 d | 12.7% | 12.4% | 0.99 |
| Acceptance → publication | 24 d | 26 d | 0.96 |
| First structure deposited > 7 d after submission | 27% | 32% | 0.29 |
| Some structure added after submission (revision) | 32% | 35% | 0.36 |
| Some structure deposited after acceptance | 3.9% | 3.6% | 0.70 |
| Structures released before the paper | 30% | 30% | 0.77 |
| Hold time (median) | 207 d | 231 d | 0.48 |
| Letter or brief-report article type | 6.4% | 4.4% | 0.22 |

* **Within-paper comparison.** In the 53 papers that hold both error and clean cryo-EM entries, the erroneous entries are not the late additions (`tests_within_paper.csv`, conditional logit, all p > 0.18):
  * offset from the paper's first deposition: mean 1.8 d vs 8.3 d;
  * share in the last deposition batch: 20% vs 26%;
  * resolution, hold time and V chains per entry: no difference.
* **The one nominal hit is weak.** Among preprinted papers, a preprint within 90 days of deposition: 83% (30/36) vs 62.5%. Adjusted OR 2.6, p = 0.04, q = 0.41.

## 2. Competitive pressure: the opposite

For each paper:
* **Antigen:** the non-antibody chains, mapped to UniProt with SIFTS.
* **Competitors:** distinct DOIs from other labs, anywhere in the PDB, with an entry containing that UniProt and deposited within ±180 days (`competition.py`).

**Error papers had *fewer* contemporaneous competitors:**
* any competitor: **58% (70/121) vs 73% (919/1256)**, adjusted OR 0.47 (0.32–0.70), p = 1.7e-4, q = 0.009;
* median number of competitors: 1 vs 5.

**The result holds up:**
* adding SARS-CoV-2, viral, tool-paper and journal terms: OR 0.47;
* unit-level GEE: OR 0.41, p < 1e-4;
* each period: 0.61, 0.38 and 0.46;
* viral non-SARS-CoV-2 antigens: 37% vs 66%, OR 0.30, p = 0.002;
* non-viral antigens: 52% vs 66%, OR 0.59, p = 0.05.

For SARS-CoV-2 every paper had competitors (100% vs 100%), so the measure is uninformative there.

Two related measures point the same way:
* same-antigen structures released in the 180 days before publication: OR 0.88 per doubling, q = 0.05;
* first structure of the antigen in the PDB: OR 1.55, p = 0.10 (unit GEE 1.79, p = 0.02). This matches the earlier "novel antigen" signal.

The prior lens's `competitors` variable showed nothing at the paper level (p = 0.10). Errors therefore sit in *less*-crowded antigen niches, not in races.

**Caveat on accessions.** UniProt accessions are strain-specific for viral proteins, so for viruses this undercounts competition by family.

## 3. Visibility and team size: weak to moderate "profile" signals, not links

* **Citations.** Error papers are cited less:
  * median 5.8 vs 8.1 citations per year;
  * within-year citation percentile 0.49 vs 0.56;
  * adjusted OR 0.68 per log unit, q = 0.034;
  * holds with the full covariate set (0.62) and in a unit-level GEE (0.63, p = 1e-4), with the same direction in every period;
  * X-ray: same direction (0.65), but n.s. (16 error papers).
* **Authors per antibody.** Error papers have fewer authors for the number of antibodies built:
  * authors per antibody V region: 4.5 vs 6.5 (median total 14 vs 14);
  * adjusted OR 0.70 per doubling of authors, q = 0.052;
  * not seen in 2018–20.
* **These three overlap.** Competition, citations and authors are moderately correlated (r 0.17–0.25), and each keeps a partial effect in a joint model (OR 0.57, 0.72 and 0.76).
  * Taken together, error papers tend to be *lower-visibility papers on less-crowded antigens, with fewer people per structure built*.
  * This describes a kind of paper, not a shared cause.
* **DFG funding.** 5/110 vs 13/991, OR 5.0, q = 0.052. The 5 papers come from 4 unrelated labs (Marlovits, Ward ×2, Draper, Frost). This is likely chance.

## 4. Links between error papers: none

* **Shared co-authors.** Among error papers from *different* labs, pairs that share at least one author number 275, against a null of 325 from permutations within strata of period × size × resolution. That is 0.85× expected, p = 0.86. Through non-last authors only, the ratio is 0.87×.
* **Individual entities.** No author, institution, cryo-EM facility or funder is enriched at q < 0.1 (1,618 authors, 309 institutions, 7 facilities and 142 funders tested).
  * The top hits are "Wang K" (a name collision, 7/12 papers across 6 labs, q = 0.29) and the Hafenstein lab (4/5).
  * Facilities show nothing: PNCC, NCEF, SLAC, NYSBC, eBIC, MRC-LMB, EMBL and "cryo-EM facility" are all p > 0.2.
* **Lab clustering at the paper level is modest.**
  * All cryo-EM papers: dispersion 1.40×, p = 0.005. Unpublished entries of one study count as separate papers here, which inflates the figure.
  * Published papers: 1.18×, p = 0.10.
  * The earlier entry-level 3.2× is therefore mostly within-paper copying.
  * Large labs are at their expected rates: Ward 10/82 papers (9.8 expected), Kwong 2/38 (4.3), Bjorkman 6/20 (3.6).
* **Journals.** Heterogeneity across 19 journal groups: p = 0.36. eLife (5/24) and J Virol (4/26) are highest, Sci Transl Med (0/18) and Sci Adv (2/42) lowest.
* **Other strata, all null:**
  * last-author region: USA 11.0%, China 10.0%, Europe 9.9%, other 3.2%; p = 0.24;
  * preprint server: p = 0.41;
  * open access, NIH, HHMI, Gates, Wellcome, Chinese, EU/ERC, MRC, CEPI, DoD, NSF and industry funding or affiliation;
  * CNS, sister or mega journals;
  * errata, comments and published peer review;
  * title keywords: COVID, vaccine, neutralizing, broad, structural basis, nanobody, fiducial/chaperone, HIV, influenza, escape, therapeutic.

## 5. Paper type

* **Tool papers** (every antibody is an auxiliary reagent) look protective in crude terms: 19.7% vs 32.5%, period-stratified p = 0.006.
  * This is a size effect. Adjusted for the number of antibodies, OR 0.90 (p = 0.69); unit-level GEE 0.86.
* **Publication status.** Unpublished entries and preprint-only citations are not worse: adjusted OR for being published 1.46, p = 0.19.

## Files

**Scripts:**
* `fetch_meta.py`, `fetch_ppr.py`: data download;
* `parse_meta.py`: builds `paper_meta.csv` and `paper_authors.csv.gz`;
* `build_base.py`: builds `paper_base.csv`;
* `analysis.py`: builds `tests_em_papers.csv`, `journal_rates.csv` and `paper_analysis_table.csv`;
* `competition.py`: builds `competition_table.csv`, `tests_competition*.csv`;
* `within_paper.py`, `network.py`, `lab_dispersion.py`, `sensitivity.py`;
* `collect.py`: builds `all_tests_bh.csv`, `tests_xray_secondary.csv` and `error_papers.csv`.

`error_papers.csv` is a browsable per-paper listing of the error and artifact papers, with dates, intervals, citations and competitors.
