# PDB and EMDB structure metadata: what sets the error entries apart

**Lens.** Entry-level PDB/EMDB metadata, compared between first-occurrence error entries (`err_first`) and clean
entries (no run of any kind), within cryo-EM.

**Data fetched.**
* RCSB GraphQL for all 4,085 entries. Fields: EM reconstruction, imaging, specimen, fitting, CTF and software;
  database status and site; entry info; validation summaries; keywords; EMDB ids; authors; revisions; funding; and
  per-chain Q-score, inclusion, unobserved, clash and cis-peptide summaries.
* EMDB REST entry JSON for all 2,828 EMDB maps: half maps, masks, additional maps, focused, consensus and composite
  references, pixel and box size, startup model and processing software.
* EMDB validation-analysis JSON (per-residue local resolution and Q-score) for every error entry plus 3 year-matched
  clean controls per error (715 entries).
* Per-instance cis-peptide and sheet positions for all 15,865 V chains, mapped to ANARCI IMGT positions.

Raw caches are in `.../scratchpad/deeper/cache/pdb_meta/`.

**Statistics.** Logistic regression per feature, adjusted for deposition-year bin (≤2018 … ≥2024), resolution and
log2(number of V chains); chain-level models also adjust for chain type. SEs are cluster-robust by paper (DOI).
BH q is computed within each table. Sensitivity (`tables/entry_sensitivity.csv`) covers period strata, one vote per
paper, +SARS-CoV-2 and +entry Q-score.

## 1. AL amyloid fibrils are in the error set and should be removed (detector artifact)

* **15 of 193 EM error entries** are cryo-EM structures of **AL amyloid fibrils**: patient λ light chains in a non-Ig
  amyloid fold (6IC3, 6Z1O, 7NSL, 9ELS, 9FAA/B/C, 9LIV/W/X/Y, 9LJ0/A/B, 9V91).
  * They make up 12 of 218 error units and **77 of 488 first-occurrence runs (16%)**.
  * They also make up **12 of 26 λ error units** and 6 of 66 mode-A units.
  * SAbR's structural numbering of a non-Ig fold is meaningless, so these are not register errors.
* They drive the apparent "antibody alone" excess: 16/193 vs 14/2,574 before exclusion, 1/178 after.
* They largely explain the earlier λ CDR1-length signal, because IGLV1-44, 1-47, 1-51 and 6-57 all have L1 = 8:
  * λ chains, L1 = 8 vs L1 = 9: **12.3% vs 1.9%** with amyloid, **2.8% vs 1.0%** without;
  * per unit: 7.9% → 4.1% vs 1.5%.
* The list is in `tables/amyloid_fibril_chains.csv`.
* Everything below excludes them: 178 error vs 2,574 clean EM entries.

## 2. The strongest link: error chains are modelled without the canonical cis-prolines (a different modelling route)

The wwPDB cis-peptide annotation was mapped onto sequence-based IMGT positions (`08_cis_sheet.py`,
`09_cis_sheet_tests.py`).

**κ Pro8 (IMGT 8) modelled cis**

| Comparison | Errors | Clean | Adjusted OR | p |
|---|---|---|---|---|
| All κ chains | **7.1% (5/70)** | 63.6% (1,891/2,975) | **0.036 (0.012–0.11)** | 2e-9 |
| Excluding chains whose error is in FR1 | 11.9% (5/42) | 63.6% | 0.071 | 2e-6 |
| Novel V regions only | 8.1% | 62.5% | 0.047 | 5e-6 |
| + adjusted for chain Q-score and starting-model type | | | 0.069 | 6e-6 |

**Other cis-prolines away from the error**

| Site | Errors | Clean | OR | p |
|---|---|---|---|---|
| κ Pro115 (CDR-L3, Kabat 95), chains with no CDR3/FR4 error | 12.5% | 62.1% | 0.079 | 0.026 |
| Any cis peptide in the Cκ domain | **20.7% (6/29)** | 80.8% | 0.039 | 0.001 |
| Any cis peptide in the CH1 domain | 41.9% | 72.3% | 0.24 | 0.054 |

**As a risk marker (EM κ chains with Pro8)**
* Error rate when Pro8 is trans: **5.7% (65/1,149)**; when cis: **0.26% (5/1,896)**.
* In 2020–21: **13.0% vs 0.2%** (1 of 467).
* So almost every κ-chain error sits in a model that lacks the cis-prolines found in any κ Fab crystal template.

**Spread and scope**
* The error chains span 22 labs and 26 papers.
* It is **not a stable lab habit**: a lab's leave-own-paper-out cis rate does not predict its errors (low-cis labs
  OR 0.27).
* It marks the **model or entry**: entries whose *clean* κ chains lack cis-Pro8 have more heavy-chain errors
  (51.7% vs 37.0%, OR 2.1, p = 0.016).

**Interpretation.** The error chains were most likely traced or threaded (built residue by residue, mutated onto a
non-matching template, or taken from an all-trans model), not docked from a correct κ Fab structure. That route also
leaves gaps that must be counted across.

## 3. Global β-sheet deficit in the error V domains, away from the error

* PROMOTIF sheet fraction of V-domain residues more than 5 residues from any error run: **0.39 vs 0.44** (adjusted OR
  0.54 per 0.08, p = 3e-12).
* It holds in each chain type (H, κ, λ), in novel V regions only, and after adjusting for chain Q-score and
  starting-model type (OR 0.56, p = 2e-10).
* Within the same entry, error chains have less sheet than clean V chains of the same type (52 pairs, 70%,
  p = 4e-5).
* The constant domain is barely different (H 0.461 vs 0.465, p = 0.03; κ n.s.).
* So the whole V domain has poorer β-geometry in error models. It is a symptom of the modelling route and a
  candidate detection flag, not an independent cause.

## 4. Entry-wide model quality is lower; the clashes are not specific to the antibody

* Clashscore **8.7 vs 6.8** (OR 1.40 per 7.3, p = 1e-6, q = 1e-4).
* Clashes in non-antibody chains: 0.19 vs 0.15 per residue (OR 1.28, p = 0.004).
* Ramachandran outliers: OR 1.23, p = 0.006.
* Within an error entry, the error V chain has **no more clashes than its clean partner chains** (p = 0.08), so the
  excess is entry-wide (modelling and refinement practice).
* Strongest in 2020–21 (clashscore OR 2.0, p = 7e-6); not significant from 2022 (1.11).
* Robust to SARS-CoV-2, entry Q-score and paper weighting (1.26, p = 9e-4).

## 5. Error sites are *not* in worse map: the map-intrinsic measure disagrees with Q-score

This uses EMDB validation per-residue local resolution, which is a map property, against the same IMGT window in
clean chains of year-matched controls.

* **Local resolution:** error sites sit at the **48th percentile**, and only 11% are in the worst quartile (25%
  expected); 134 sites, p = 0.05 toward *better*.
* **Q-score at the same sites:** **58th–61st percentile**, 30% in the worst quartile (p = 8e-8). The paired
  difference between the two measures is significant (p = 1e-4).
* **Chain level:** V-domain local resolution minus map resolution is 0.45 vs 0.41 Å (OR 0.99, p = 0.9). Relative to
  the non-antibody parts of the map: n.s.
* **Implication:** the low Q-score and RSCC at error sites (prior finding, and `ab_q`/`dq_ab_minus_non` here) are
  largely a *consequence* of the misplaced model, not evidence of poor density.
* Caveat: only entries with half maps have local resolution (≈39% of the fetched set), and local resolution is
  smooth over a few Å.

## 6. "Antibody is a small part of a huge complex": rejected, if anything the opposite

* Error entries are smaller: total mass OR 0.69 per doubling (p = 0.005; paper-weighted 0.62, p = 6e-5).
* Antibody <15% of mass: 18.0% vs 25.3% (OR 0.57, p = 0.04).
* Membrane-protein or receptor keywords: 19.1% vs 32.4% (OR 0.47, p = 0.008). Fab fiducials on membrane proteins are
  clean, matching the auxiliary-reagent finding.
* No difference for >500 kDa (14.0% vs 12.7%) or >1 MDa (1.1% vs 1.2%).

## 7. Microscope tier (weak; clustered)

* Krios **67.4% vs 78.1%** (OR 0.53, p = 0.005, q = 0.06).
* The effect is in ≤2019 (0.19) and 2020–21 (0.29), and is n.s. from 2022 (0.79).
* Related: 300 kV OR 0.59 (p = 0.05); pixel size 1.07 vs 1.02 Å (OR 1.31, p = 0.005).
* In 2020–21, **8 of 13** Tecnai Arctica entries are errors. These are mostly one group (FMDV scFv, Lou Z., ×5) and
  echovirus (Feng R., ×2). Polara: 3 of 3.
* It survives paper weighting (OR 0.57, p = 0.01), but it is a facility/lab proxy.

## 8. Sparser depositions (weak)

* No funding record: 13.5% vs 4.6% (OR 2.4, p = 0.01; 2020–21 only).
* Fewer EMDB metadata fields: 371 vs 412 (p = 0.014).
* Half maps deposited: 62% vs 76% (OR 0.58, p = 0.03; n.s. once they became mandatory).
* None of these survives paper weighting.
* Fewer total revisions (p = 0.007). This is probably remediation-driven and uninterpretable; major coordinate
  revisions show no difference (5.6% vs 5.9%).

## NO difference (EM, adjusted)

| Area | Features tested |
|---|---|
| Local, focused and composite maps | Local or focused refinement (13.4% vs 13.7%); EMDB focused or consensus references; composite flag or keyword; additional maps; masks; multiple maps per entry |
| Post-processing and processing | DeepEMhancer; density modification; LocScale or EMReady; sharpening; non-uniform refinement; symmetry expansion; 3D variability |
| Particles and symmetry | Particle numbers (final, per ASU, per kDa, picked, retention); point symmetry |
| Software and startup model | Final reconstruction software (cryoSPARC 74% vs 75%, RELION); EM startup model type |
| Sites | Processing site (RCSB, PDBe, PDBj; PDBc 3.9% vs 9.6%, p = 0.09); deposition site; cross-site processing; deposit groups. **Annotator identity is not public in PDB/EMDB metadata and could not be tested** |
| Imaging and specimen | Detector K2; energy filter; CTF; dose (marginal); box size; concentration; pH |
| Map-level quality | Entry Q-score (0.40 vs 0.43, p = 0.11); calculated FSC resolution and its gap to the author's value; non-antibody Q and inclusion |
| Unmodelled fraction | Entry-wide and non-antibody unmodelled fraction (if anything *fewer* unmodelled residues in error entries); antibody-chain unobserved coverage |
| Revisions | Major coordinate revisions |
| Authors and funding | Funding country or agency (US, China, UK, NIH, Gates); author count (marginal, fewer) |
| Composition | Viral keywords; >500 kDa or >1 MDa complexes; antibody-alone (after the amyloid fibrils are removed) |
| Local resolution | V-chain local resolution relative to the map and to the rest of the complex |

## Files

| Kind | Files |
|---|---|
| Scripts (in order) | `scripts/01_fetch_rcsb.py` … `13_local_resolution.py`; `statlib.py` |
| Main tables | `tables/entry_tests_em.csv` (also `_with_amyloid`), `entry_sensitivity.csv`, `chain_tests_em.csv`, `within_entry_pairs.csv`, `cis_sheet_tests_em.csv`, `local_resolution_tests.csv`, `local_resolution_sites.csv`, `amyloid_fibril_chains.csv` |
| Feature tables | `entry_features_rcsb.csv.gz`, `entry_features_emdb.csv.gz`, `emdb_features.csv.gz`, `chain_features_rcsb.csv.gz`, `chain_cis_sheet.csv.gz`, `chain_novel_v.csv.gz`, `em_entry_analysis_frame.csv.gz` |
