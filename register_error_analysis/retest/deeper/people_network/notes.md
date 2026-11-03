# People, labs and facilities: do the error structures share people?

**Short answer.** No person, co-author group, core-staff "hub", cryo-EM facility, institution or country links the
error structures beyond chance. The error studies are, if anything, *less* connected to one another and to the rest
of the field. The one robust people-level signal is **inexperience**: teams in which no author had deposited an
antibody cryo-EM structure before are about **twice as likely** to contain a first-occurrence error. The kind of
errors they make is the same as for experienced teams. That penalty is confined to 2023 and earlier.

## Data and definitions

* **Set.** Cryo-EM antibody entries.
  * *Error*: a first-occurrence run (`err_first`).
  * *Clean*: no run of any kind.
  * *Excluded*: copied or artifact-only runs, and the 15 AL amyloid fibril entries (a detector artifact; see `../pdb_meta`).
  * This gives 178 error and 2,574 clean entries.
* **Study.** A citation DOI. Unpublished entries are grouped by last author, with deposits up to 60 days apart.
  * This gives 127 error and 1,175 clean cryo-EM studies.
  * The study is the primary unit. Entry-level results are secondary, because a study's entries share people and
    errors.
* **People.** PDB audit authors (RCSB GraphQL), with ORCID for 57% of author slots after mapping names to ORCIDs.
  * Key: the ORCID, otherwise surname plus all initials.
  * Surnames prone to collision (common East Asian and Western surnames) are dropped in the `_na` / `noamb`
    sensitivity analyses.
  * Paper author lists and affiliations come from Europe PMC (`../paper_meta`).
* **Experience.** Counted from all ~262k entries in `entries.idx`, using entries deposited before the study's first
  deposition. "Antibody entry" uses the PDB-wide ANARCI set (`../antigen_competition`, 11,972 entries).
* **Statistics.**
  * **Entity scans:** exact stratified Fisher tests (strata are deposition-year categories).
  * **Models:** logistic regression adjusted for deposition-year category, resolution and log2(number of V chains).
    Entry-level models use SEs clustered by study.
  * **Network and linking tests:** permutation, or 2,000 year-matched samples of clean studies.
  * **Multiple testing:** BH within each entity class, and across the lens's 78 primary tests (`tables/all_tests_bh.csv`).

## 1. Over-representation of specific entities: none at the study level

Source: `tables/entity_enrichment_summary.csv`. Rows are study level unless marked "entry".

| Entity class (≥3 cryo-EM studies) | n tested | Smallest q |
|---|---|---|
| People, any author position | 503 | 0.46 |
| People, non-last authors only | 389 | 1.0 |
| Labs (last author) | 108 | 0.099 (one lab, 3/3 studies) |
| Co-author groups (pairs sharing ≥3 studies, as connected components) | 53 | 0.62 |
| Multi-lab cryo-EM "hub" people (core-staff proxy) | 46 | 1.0 |
| Institutions (affiliations) | 274 | 0.33 |
| Facility named in affiliations (PNCC, NCEF, SLAC, NYSBC, eBIC, core facilities …) | 12 | 1.0 |
| Country of the last author | 19 | 1.0 |
| Country of any author | 31 | 0.31 (Belgium; see §6) |
| Any facility mention (aggregate) | 1 | 8 observed vs 8.2 expected |
| Any hub person on the study (aggregate) | 1 | 18 vs 26.1 (MH OR 0.60, two-sided p = 0.06) |

* **The single lab near q = 0.1 is an antigen niche.** It had 3 of 3 studies with errors, all on viral capsids,
  deposited in 2016, 2017 and 2024. This repeats the capsid effect found in `../antigen_competition`.
* **The entry-level "hits" are within-study repetition.** The 16 person-level and 11 lab-level entities with
  q < 0.1 each rest on 1–2 error studies (lab-level hits: up to 3). Many entries of one study share both the authors
  and the error (`tables/entry_level_hits_by_study.csv`).

## 2. Co-authorship network: error studies are not more connected

Year-matched (deposit half-year) clean samples, 2,000 draws (`tables/network_vs_matched_clean.csv`):

| Link type (study level) | Statistic | Error | Null | Ratio | One-sided p (more / less) |
|---|---|---|---|---|---|
| Any shared audit author | edges | 120 | 132 | 0.91 | 0.65 / 0.37 |
| | largest component | 13 | 23 | 0.57 | 0.89 / 0.16 |
| Shared person, different labs | edges | 45 | 62 | 0.72 | 0.86 / 0.15 |
| Shared non-last author, different labs | share linked | 0.18 | 0.27 | 0.67 | 0.97 / 0.045 |
| … collision-prone names dropped | edges | 7 | 19.4 | **0.36** | 1.0 / **0.010** |
| Entry level, other studies | share linked | 0.53 | 0.64 | 0.82 | 1.0 / **0.003** |

* **Other null models agree.** Matching also on author-count bin, and using paper author lists (Europe PMC), give the
  same picture.
  * The paper-author network is one giant component, so it carries no information.
* **The cross-lab components of error studies are small.** The largest have 13 studies (4 labs) and 11 studies
  (8 labs), both below the null's largest component.
* **Isolation is a side-effect of inexperience.** A study whose co-authors had earlier antibody studies in *another*
  lab has fewer errors: 8.4% vs 12.1%, OR 0.68, p = 0.04. Adjusted for the new-team flag (§4), this falls to OR 0.74,
  p = 0.13 (`tables/connectedness_vs_experience.csv`).

## 3. Does the same person link error studies across labs, or carry an error from lab to lab? No

* **Static linking** (labels permuted within year × resolution × size strata; `tables/crosslab_linking_perm.csv`):

  | Statistic | Observed | Expected |
  |---|---|---|
  | People on error studies of ≥2 different labs | 37 | 36.5 |
  | Most labs linked by one person | 4 | 4.1 |
  | Non-last authors on error studies of ≥2 labs | 20 | 21.8 |

  The results are the same with collision-prone names dropped.
* **Temporal "carrier" test.** Among studies with a co-author who had worked on an earlier antibody study in another
  lab, the earlier study having an error does not raise the risk.
  * Error rate: 8.2% if that earlier study had an error vs 8.5% if not.
  * OR 1.02 (0.52–1.99); with lab experience 1.00; with collision-prone names dropped 1.18.
* **Lab persistence is project continuity, not lab habit.**
  * A lab with an earlier error study has more errors: 12.2% vs 7.5%, OR 2.19, p = 0.007.
  * This shrinks with adjustment for antigen class, nanobody, mouse and SARS-CoV-2 (OR 1.69, p = 0.08).
  * It is null when the earlier error was on a *different antigen* (OR 1.04) or ≥180 d earlier (1.39).
  * Within labs, pairs of error studies share an error mode no more often than chance: 15 vs 19.9 expected of 73
    pairs, p = 0.87 (`tables/robustness_tests.csv`, `tables/mode_concordance_within_lab.csv`).

## 4. Experience: the robust signal is a team new to antibody cryo-EM

Source: `tables/experience_tests.csv`, study level, adjusted.

| Exposure | Error | Clean | OR (95% CI) | q (lens) |
|---|---|---|---|---|
| **No audit author had a prior antibody cryo-EM entry** | **47.2%** | **33.2%** | **1.99 (1.36–2.91)** | **0.010** |
| No audit author had any prior cryo-EM entry | 29.9% | 21.5% | 1.77 (1.16–2.68) | 0.039 |
| Most experienced author, prior antibody cryo-EM entries (per doubling) | median 2 | 4 | 0.89 (0.81–0.97) | 0.039 |
| Last author, prior antibody entries (per doubling; replicates `../antigen_competition`) | 4 | 6 | 0.90 (0.83–0.98) | 0.067 |
| Last author's first antibody cryo-EM entry | 46.5% | 36.4% | 1.62 (1.11–2.38) | 0.055 |
| Last author's first cryo-EM entry | 26.0% | 22.3% | 1.31 (0.85–2.03) | n.s. |
| Last author, prior PDB entries of any kind | 32 | 41 | 0.98 | n.s. |
| Number of audit authors | 3 | 3 | 0.84 | n.s. |

* **The penalty is antibody-specific.**

  | Team experience | Error rate | OR |
  |---|---|---|
  | No cryo-EM experience | 13.1% (38/291) | 2.02 (1.31–3.12) |
  | Cryo-EM experience, but no antibody cryo-EM | 13.8% (22/159) | 1.96 (1.16–3.33) |
  | Antibody cryo-EM experience (reference) | 7.9% (67/852) | — |

* **The new-team effect survives every check** (OR 1.73–2.12 throughout; `tables/robustness_tests.csv`,
  `tables/new_team_mapq_adjusted.csv`, `tables/robustness_any_run_outcome.csv`):
  * antigen class (capsid, enzyme), nanobody, mouse, SARS-CoV-2 and first-of-family adjustment: 1.92;
  * plus last-author experience: 1.87;
  * published studies only: 2.12;
  * excluding capsid and enzyme targets: 1.94;
  * name-collision handling (flag from all name keys): 2.05;
  * teams with ≥2 non-collision-prone names: 1.93;
  * number of authors and last-author country: 2.06;
  * local map–model fit, AlphaFold-type starting model and flexible fitting: 1.94;
  * counting copied and artifact runs as errors: 1.73 (p = 0.016);
  * the X-ray direction agrees (OR 2.1, n.s., 33 errors).
* **Same errors, more often.** The mode mix of new-team errors matches experienced teams; only FR3 is nominal,
  p = 0.04 (`tables/modes_by_new_team.csv`).
* **No influx explains the 2020–21 peak.** The share of new teams is flat at 32–38% in every year
  (`tables/influx_by_year.csv`).
* **The penalty disappears from 2024 (exploratory; the split was chosen after seeing the per-year table).**
  * Studies deposited up to 2023: OR **3.05** (1.91–4.87).
  * Studies deposited from 2024: OR 0.79 (0.38–1.65).
  * Interaction p = 0.001 (`tables/new_team_by_era.csv`).

## 5. Core staff and facilities

* **Hub people.** These are PDB-wide cryo-EM people who are non-last authors for ≥5 labs (310 people, a proxy for
  facility or core staff).
  * Study level: 14% of error vs 21% of clean studies had one, OR 0.67 (0.39–1.13), p = 0.13.
  * Entry level: OR 0.42 (p = 0.003).
  * Mostly 2022+ (OR 0.31).
  * Reading: a weak protective association, not a source of errors.
* **Facility names in affiliations.** No association (8 observed vs 8.2 expected).
* **No facility field.** Neither the PDB nor the EMDB records a data-collection facility, so facility is inferred only
  from people and affiliations.

## 6. Post hoc lead (hypothesis only): a shared nanobody source

This follows up the top country hit (Belgium, any author; q = 0.31). Its p-values are not corrected for that
selection. Source: `tables/nanobody_supplier_tests.csv`.

* **The association.** Among cryo-EM studies with a nanobody, those with a VUB/VIB (Brussels) co-author had errors in
  **9/31 vs 26/353**.
  * Year-stratified OR 5.0, p = 0.0016.
  * Adjusted OR 3.8, unchanged by the new-team flag.
  * It holds when the last author is outside Belgium: 7/26.
  * Non-nanobody studies show no effect.
* **The studies are spread out.** The 9 error studies come from 8 last-author labs (2019–2025). Two VUB-affiliated
  co-authors appear on 6 and 5 of them, but also on 27 and 21 VUB-linked cryo-EM studies overall.
* **The errors are mixed.**
  * Modes: FR1, CDR2/FR3, CDR3 ±1.
  * Roles: both subject nanobodies and auxiliary ones.
  * CDR lengths and germline identity match other nanobodies.
  * CDR2/FR3 errors are concentrated: 4/62 vs 3/505 units.
* **It does not survive a full scan.** In a scan of all 79 institutions within nanobody studies, VUB ranks first, but
  at q = 0.80.
* **Reading.** A shared reagent source (many labs modelling nanobodies from one discovery pipeline) is a
  candidate link worth checking with other lenses (local resolution at the nanobody, sequence features). It is not
  established.

## Nulls (no difference)

* Any individual (audit or non-last author), co-author group, hub person, institution, named facility, or last-author
  or any-author country, at the study level after BH.
* **Network:** co-authorship connectivity of error studies vs year-matched clean studies (edges, linked share,
  components, largest component), for any shared author, cross-lab, and cross-lab middle-author links.
  * The only departures are toward *fewer* links.
* **Cross-lab linking:** people linking error studies across labs; the most labs linked by one person.
* **Carrier test:** an earlier error in a co-author's other lab.
* **Lab persistence once antigen continuity is removed.** The error mode does not recur within labs.
* **Last author's general experience:** total PDB entries, first-ever cryo-EM entry, first antibody entry.
* **Team size:** audit-author count.
* **Facility mentions in affiliations.**

## Caveats

* **Names.** Audit author lists use initials. ORCIDs cover 57% of author slots and fewer before 2019.
  * Collisions inflate experience and links for common names; splits deflate them.
  * The main results hold under both a collision-inflating and a collision-deflating definition.
* **Audit authors ≠ paper authors.** The PDB list is shorter and closer to the people who built the model, but
  depositors choose it. Facility staff are often only acknowledged.
* **Experience is PDB-visible experience.** Unpublished in-house work and X-ray-only antibody work count as
  inexperience in the antibody cryo-EM flag. The flag also cannot separate "new people" from "new to this kind of
  map".
* **Confounding and overlap with other lenses.**
  * Inexperience correlates with lower citations and fewer authors per antibody (`../paper_meta`), and with capsid
    and enzyme niches (`../antigen_competition`; adjusted here).
  * The new-team association is with *who models*, not proof of a mechanism.
* **Exploratory results.** The era interaction and the nanobody-supplier lead were found after looking at the data.
  Both need confirmation.
* **Privacy.** People are reported only in aggregate. Repo tables carry pseudonyms (P####, L####). The name maps and
  raw author keys stay in the scratchpad cache (`cache/people_network/`).

## Files

* **Scripts:**
  * `s01_fetch_orcid.py`: audit authors with ORCID;
  * `s02_build.py`: entry and study tables, PDB-wide experience;
  * `s03_enrichment.py`: entity scans;
  * `s04_network.py`;
  * `s05_experience.py`;
  * `s06_crosslab_people.py`: linking, carrier and persistence;
  * `s07_robustness.py`;
  * `s08_nanobody_supplier.py`;
  * `s09_influx.py`;
  * `s10_collect.py`: lens-wide BH and the pseudonymised study frame;
  * `common.py`.
* **Tables:** `tables/*.csv`. Logs are in `tables/*.log`.
