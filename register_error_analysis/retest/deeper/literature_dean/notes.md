# How the register-error antibodies compare with published antibody structural statistics (lens: literature_dean)

**Question.** Are the antibodies with first-occurrence register errors unusual by the yardsticks of the antibody
structural-statistics literature? That literature means mainly Charlotte Deane's group (SAbDab, ABangle, SCALOP,
ABodyBuilder/ImmuneBuilder), Dunbrack's North/PyIgClassify clusters, and the papers on model-quality validation.
What, if anything, links the errors?

## Short answer

1. **Rarity is not the link.** Measured against a SAbDab-style curated reference, the error antibodies don't carry
   rare CDR lengths overall. The reference is 3,593 unique V domains with an X-ray structure at ≤3.0 Å, first released
   ≤2019. Units with any CDR length below 5% frequency: 4.2% flagged vs 4.3% (adjusted OR 0.85, q=0.48).
2. **What links them is a pull toward the common canonical length.** We compared the CDR lengths implied by the
   deposited coordinates (SAbR, structural numbering) with the true sequence lengths (ANARCI).
   * In first-occurrence errors, the apparent length was **more common in the reference than the true length in 56 of
     107 cases.** If the slip direction were random (same size), 36.5 would be expected (p≈4e-5).
   * The apparent length **was the reference mode in 34 cases vs 19 expected** (p≈2e-5).
   * The result holds within cryo-EM (48/89 vs 31.0) and with every reference tried (p 4e-5 to 2e-4).
   * The detector itself has no such bias. In ~15,000 chains without any disagreement run, the two numberings give the
     same length 99.9% of the time (98.8% at rare lengths), and never toward a more common length.
   * So the deposited loop often looks like the commonest canonical length, with the sequence threaded one or two
     residues off.
   * This fits building from (or refining toward) a generic or common-length template. AMA-II (Teplyakov 2014) found
     that PDB errors and canonical-template choice propagate into models. Greenshields-Watson 2024 found that
     deep-learning predictors don't extrapolate to loop shapes they haven't seen.
3. **Specific common, non-modal canonical lengths are error-prone.** Each is well populated in the reference, but the
   majority form sits right next to it:
   * **Heavy CDR2 IMGT 7 (North H2-9) vs 8 (H2-10):** 8.0% vs 3.4% flagged, OR 2.44 (1.63–3.67), q=1.4e-4. Still
     2.18 after adjusting for nanobody, species, germline identity and SARS-CoV-2. 2.38 in non-nanobody human Fabs.
     Errors move both ways between H2-9 and H2-10 (8 vs 7 units), as if templates of the other length were used.
   * **λ L1-13 (IMGT 8) vs L1-14:** 11.4% vs 3.0%, OR 4.27, q=0.009. Seven of these units were built with an L1-14-like
     apparent length.
   * **κ L1-17 (IMGT 12) vs L1-11:** 10.3% vs 3.5%, OR 3.13, q=0.02.
4. **The error antibodies fall outside the data the published statistics were built on.**
   * Only 7% (13/181) of cryo-EM error units have any X-ray ≤3.0 Å structure of the same V domain in the PDB, vs 20%
     of clean units. Flagged rate 1.5% vs 5.0%, OR 0.29 (0.16–0.51), q=1.4e-4. This partly restates the known
     protective effect of templates.
   * Most structural-statistics papers are X-ray only at ≤2.8–3.0 Å: North 2011, Nowak 2016, Regep 2017, ABlooper,
     Guloglu 2023 and TAP.
   * So the canonical-form and CDR statistics in the literature have essentially never sampled these structures.
5. **The archive's composition changed sharply.** PDB-wide, the share of newly seen unique V domains that have no
   X-ray ≤3.0 Å structure rose:

   | First-release year | No X-ray ≤3.0 Å structure | Seen in any cryo-EM entry |
   |---|---|---|
   | 2012–18 | 11–24% | 7–18% |
   | 2020 | 33% | not extracted |
   | 2021 | 46% | 44% |
   | 2022 | 54% | not extracted |
   | 2025 | 69% | 66% |

   In the study set, chains from cryo-EM went from 0% (2012–14) to 76% (2021) and 89% (2025). SARS-CoV-2 made up
   36–42% of 2020–22 chains. Literature: SAbDab grew from 1,624 entries to 5,426 (Sep 2021); SAbDab-nano had 823
   structures; CoV-AbDab listed 1,402 coronavirus antibodies by Aug 2020.
6. **Machine-learning training sets.**
   * ImmuneBuilder/ABodyBuilder2 trained on SAbDab structures at ≤3.5 Å, X-ray *or* cryo-EM, with no internal missing
     residues, extracted 31 Jul 2021 (7,084 structures).
   * Among cryo-EM chains released before that date, error chains *pass* that filter more often than clean chains: 59%
     vs 35%, OR 2.74, p=5.5e-8. The error chains sit at better nominal resolution (median 3.43 vs 3.70 Å).
   * At most ~29 error-carrying entries (21 cryo-EM, 8 X-ray) fit the filter, about 0.4% of that set. This is an
     upper-bound approximation.
   * Separately, nominal resolution didn't protect in 2020–21 cryo-EM: 7.0–7.4% flagged at 2.9–3.5 Å vs 8.9% at
     3.8–4.2 Å.

## What the literature says (reference statistics and filters)

* **Quality filters.**
  * North 2011: low-resolution and high-B CDRs removed, about 300 non-redundant structures.
  * Nowak 2016: X-ray only, ≤2.8 Å.
  * Regep 2017: ≤3.0 Å.
  * Guloglu 2023: X-ray only, ≤3 Å, paired chains.
  * TAP 2019: X-ray only.
  * ABlooper: <3.0 Å, no missing CDR backbone.
  * ImmuneBuilder: ≤3.5 Å, no internal gaps; includes cryo-EM.
  * PyIgClassify2: EDIA electron-density filter, and a cluster needs ≥10 sequences.
  * Cryo-EM structures were excluded, or entered only after 2021.
* **How often antibody structures are wrong or unusual.**
  * AMA-II: PDB errors propagate into models, and canonical-template choice is a main error source.
  * PyIgClassify 2015: many cis-non-Pro CDR residues are probably low-resolution refinement errors. 71% of new
    length/cis clusters were singletons.
  * Kelow 2022: about half of the North clusters retired, some due to misfitting to density.
  * Abhinandan & Martin 2008: about 10% of Kabat entries carry numbering errors.
  * Regep 2017: more than 30% of H3 loops have a unique structure, vs under 3% for other loops.
* **Register errors in general.**
  * IUCrJ 2024 (conkit-validate): 1 in 6 entries at 3–5 Å carries a predicted register error. The rate goes from 10%
    at 3.0 Å to 20% at 3.7–5 Å, and cryo-EM is worse than X-ray at the same nominal resolution.
  * checkMySequence 2022: register shifts are among the hardest errors to see.
  * Croll 2021: errors are more common in SARS-CoV-2 structures solved under pressure.
  * Kossiakoff (Bailey 2018): Fab elbow flexibility degrades density.
* **Germline coverage (computed here).**
  * Human V genes with an X-ray ≤3 Å structure: 104, with a median of 22 unique V domains per gene (IQR 8–57).
    IGHV3-23 has 647.
  * Mouse: 244 genes, median 5.
  * The λ share of new PDB V domains is flat at 10–13% (2010–25), compared with 30–35% natural abundance (Raybould
    2024).
* **VH–VL orientation.** ABangle uses five angles and one distance (HL, HC1, LC1, HC2, LC2, dc). Not computed here;
  see the `struct_unusual` lens.

## Nulls
* Rare CDR length (<5% in the reference) for any CDR, CDR1 or CDR3, and a continuous rarity score, vs any error:
  q≥0.09.
* Rare CDR1 length vs the CDR1 mode: OR 2.5, q=0.37 (n=12).
* Rare length vs the N-terminal FR1 mode (a location-mismatch control): null.
* Whether the CDR3 slip modes (D+, D−) occur where L3−1 or L3+1 is the more common length (unit level): null.
  D− borderline, OR 1.29 per log2, q=0.09.
* CDR3 length distribution, error vs clean cryo-EM units: H p=0.08, κ p=0.99, λ p=0.27.
* The canonical pull in X-ray-only error units: 8/18 vs 5.5 expected, p=0.13 (small n).
* The canonical pull by starting-model type, fitting protocol or period: no visible difference (n too small).
* λ share among newly deposited V domains over time: flat.

## Caveats
* "Apparent length" is SAbR's structural numbering of the deposited coordinates. The pull test assumes either slip
  direction would be equally detectable.
* The North labels are IMGT-converted (L1 = IMGT+5, H1 = IMGT+5, H2 = IMGT+2). H3 and L3 are kept as IMGT lengths.
* The reference is computed here from the PDB with ANARCI. It is not SAbDab's own download: SAbDab uses Chothia CDRs,
  and SAbDab2's ML-grade criteria weren't available.
* The ImmuneBuilder-filter emulation uses V-domain internal gaps only, and doesn't check pairing.
* The Croll 2021 full text and the publisher pages weren't retrieved. Their claims come from abstracts or preprints.

## Files
* `scripts/s01–s03`: Europe PMC fetch and full-text grep.
* `scripts/s04_build.py`: reference and augmented table.
* `scripts/s05_analysis.py`: tests T1–T19 with BH.
* `scripts/s06_apparent_length.py`: the canonical-pull test and detector control.
* `scripts/s07_context.py`: filters, composition and germline coverage.
* `scripts/s08_pull_robustness.py`: robustness across references.
* `tables/`:
  * `tests_bh.csv`, `analysis.log`;
  * `cdr_length_vs_reference.csv`;
  * `canonical_pull_tests.csv`, `canonical_pull_robustness.csv`, `canonical_pull_transitions_north.csv`;
  * `apparent_length_error_units.csv`, `sabr_length_bias_clean_chains.csv`, `apparent_length.log`;
  * `context.log`, `pdbwide_composition_by_year.csv`, `study_composition_by_year.csv`;
  * `units_literature.csv.gz`, `ref_vdomains.csv.gz`;
  * `literature_facts.csv` (citations and URLs).
