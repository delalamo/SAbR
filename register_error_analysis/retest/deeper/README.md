# What links the antibody register errors? Synthesis of nine investigation lenses

This document synthesises nine independent lenses (`*/notes.md`) and the adversarial checks of their strongest
findings (`verify/*/notes.md`). Each checked finding got one confounding check and, for most, one data-integrity
check. A small amount of new work was done for the synthesis (`synthesis/`): a combined list of detector artifacts,
and rechecks of literature_dean results that had not been verified (canonical pull and three CDR-length tests).

Throughout, "associated with" means a statistical association in observational data. None of these results shows
that a factor *causes* an error.

## 1. The question

In PDB antibody V domains at 3–5 Å, SAbR (structure-based IMGT numbering) and ANARCI (sequence-based) sometimes
disagree over three or more consecutive residues: the sequence has been threaded one or more positions out of
register.

- **Data.** 15,865 single-domain V chains in 4,085 entries.
- **Errors.** 218 antibody-in-paper units (183 cryo-EM) carry a *first-occurrence* error: one that was not copied
  from an earlier structure and is not a known detector artifact. These sit in 227 entries (193 cryo-EM).
- **Already known.** About 90% of errors are cryo-EM, with a peak in 2020–21. Errors sit next to unbuilt residues.
  Atypical antibodies at the error location are over-represented. Software, lab, germline lineage and copying do
  not explain the errors.

The user is convinced that something links these structures and asked us to look at everything:

- structure metadata;
- paper metadata;
- the antigen, and whether other labs were publishing the same antigen at the same time (competitive pressure);
- people;
- whether the antibodies are unusual by published antibody structural statistics (the Deane group and others).

## 2. What was examined

| Lens | What it looked at |
|---|---|
| `pdb_meta` | About 100 PDB/EMDB fields per entry: validation scores, cis-peptides, secondary structure, EMDB processing, microscope, local resolution, complex size, deposition completeness |
| `paper_meta` | Preprints, review and deposition timing, journal, funders, country, citations, author counts, co-author links, same-antigen competition (UniProt via SIFTS) |
| `antigen_competition` | Antigen identity and class, other labs depositing the same antigen within ±1/6/12 months, the early SARS-CoV-2 race, lab antibody experience |
| `seq_novelty` | Distance to earlier PDB antibodies, first-time germlines, SEQRES truncation, rare IMGT columns, species |
| `struct_unusual` | Deane-group structural statistics (ABangle VH–VL orientation, canonical CDRs, CDR-H3 base, ImmuneBuilder comparison) and MolProbity geometry |
| `people_network` | Individual authors, co-author network, facility and core-staff "hubs", institutions, countries, team experience |
| `literature_dean` | Comparison with curated X-ray reference statistics (SAbDab, PyIgClassify style): CDR-length rarity, canonical lengths, training-set coverage |
| `case_files_ABCF` | 29 curated case files with 45 matched controls for FR1 N-terminal (A), CDR2/FR3 (B), FR3 (C) and CDR1 (F) slips: templates, methods text, high-resolution re-checks |
| `case_files_DE` | 27 case files for CDR3/FR4 slips (D+, D−, E), with a framework-superposition check that does not use SAbR |
| `verify/` | 22 findings re-run adversarially: 38 checks in total (confounding and data integrity); 12 confirmed, 25 weakened, 1 refuted |

## 3. First, a substantial part of the "errors" are not register errors

Five verified findings show that the error set itself is contaminated. The combined list
(`synthesis/s01_artifact_ledger.py`, run-level table `synthesis/artifact_ledger_runs.csv`) removes 38 of 218
first-occurrence units (17%; 34 of 183 cryo-EM units) as confirmed artifacts or copies. Adding probable and
unconfirmed cases raises this to 51 units (23%; 47 cryo-EM).

| Source | Units removed (strict) | Evidence | Verdicts |
|---|---|---|---|
| **AL amyloid fibrils** (15 cryo-EM entries, 77 of 488 runs) | 12 | Patient λ light chains in a flat, non-Ig fibril fold. Cα "thickness" is 1–3 Å vs ≥4.8 Å for every Ig chain, with no overlap. The two numberings disagree at 97% of residues. All 17 scanned AL fibrils are flagged. | confirmed ×4 |
| **FR3/FR1 indel gap placement** (gap at IMGT 84, 80+84, 84+85; FR1 gap at 26; FR3 85A–G; H 4A; CDR2/FR3 66–75 insertions) | 16 | The same disagreement appears in 15 of 16 X-ray structures at ≤2.75 Å and in ImmuneBuilder/IgFold models, which cannot contain a deposition error. SAbR's `correct_de_loop` always puts a one-residue FR3 deletion at 81, while ANARCI's HMM puts it elsewhere. | ARTIFACT_INDEL confirmed ×2; FR3START_B **refuted**; FR3_C and SN3 weakened |
| **CDR3/FR4 chain-end artifact** (models ending at IMGT 118–119) | 5 (11 runs) | Truncating 80 correct chains after 118 or 119 produces a "D+" run in 80/80 and 75/75; after ≥121 it produces 0. Every scanned chain ending at 118–119 is flagged. | confirmed ×2 |
| **Near-identical copies** missed by the exact-sequence copy rule (7JHG/7JHH/7M74 ← 6CMO; 8SNC/8SND ← 7SP7; 8XLK ← 7TE9; and others) | 4 | Same mode, same shift, ≥90% identity | — |
| Conformational NCS variant (9OM5) | 1 | Run present in 3 of 6 identical copies, which are not shifted relative to each other | — |
| *Loose list only:* D/E runs not confirmed by superposition, two probable indel artifacts | +13 | Mostly next to unbuilt FR4 residues; may still be real | — |

**Consequences for earlier results.**

- **Mode A and λ CDR1 signal.** The amyloid fibrils produced much of the earlier "atypical λ CDR1 length" signal
  and inflated the mode-A and "other" counts.
- **λ L1-13 replication.** literature_dean's replication of λ L1-13 (IMGT L1 = 8) goes from OR 4.3 (p=0.003) to
  OR 2.6 (p=0.13) once the 12 amyloid units are removed (`synthesis/length_tests_recheck.csv`).
- **"Next to an unbuilt loop" origin.** Amyloid inflated this origin from 34.8% to 44.5% of runs.
- **Recent error rate.** Amyloid also inflated the 2024–25 error rate. Without it, the fall after 2021 is
  steeper (2025: 5.8% → 4.1%).
- **Mode C.** Up to a third of the C (FR3) units are gap-placement artifacts: 6 confirmed, up to about 10 of 31.
  - The "same error in different labs" impression in FR3 comes from these artifacts.
  - Several labelled "human IGHV3-53/3-66" are actually rabbit VH, misassigned by human/mouse-only germline HMMs.
- **Mode D+.** About a fifth to a quarter of D+ units are the chain-end artifact. This explains the earlier
  "Fv-only / FR4 truncated" D+ association in part.
- **Lenses that still include amyloid.** struct_unusual, literature_dean and seq_novelty did not exclude amyloid.
  Where this was checked, their main results did not change (§4, §6).

## 4. What links the genuine first-occurrence errors (survived verification)

Ranked by how well each held up. "Weakened" means the association reproduced but its size, specificity or
interpretation had to be narrowed.

### 4.1 Strongest: they are poorly built antibody models, and the problem belongs to the whole entry or paper, not to the slipped strand

Several independent signals in the deposited coordinates point the same way.

| Signal | Error vs control | Effect (at the most conservative unit) | Verification |
|---|---|---|---|
| **κ Pro8 modelled cis** (canonical in κ V domains) | 7% (5/70) vs 64% (1,891/2,975) of clean cryo-EM κ chains | Chain OR 0.036. Paper-weighted OR 0.115 (0.04–0.31). Excluding FR1 errors, per antibody-in-paper, OR 0.14 (0.05–0.37), p=9e-5. Coordinates match the annotation in 580/580 chains checked. | Weakened (confounding), confirmed (integrity) |
| **κ CDR-L3 cis-Pro95** (sequence implies cis; ImmuneBuilder predicts cis in 67/67) | 23% (6/26) vs 76% (48/63) | OR 0.077. Marks the same all-trans V model as Pro8. | Not separately verified; consistent with Pro8 |
| **Backbone geometry outside the error site** (Ramachandran "allowed", run ±3 excluded) | 9.5 vs 5.7 per 100 residues | +3.75 per 100 (p=3e-7). One case per paper: +2.6. Replicates in X-ray (+3.7) and with a new control set (+4.5). Holds beyond 20 Å from the site. | Confirmed (integrity), weakened (confounding) |
| **Antibody vs rest of the same model** (Ramachandran "allowed" gap, V minus non-V) | +2.2 percentage points vs clean entries, population-wide | p=4e-5. Within paper +2.3. In V chains with no disagreement at all, +1.1 (p=0.009). | Weakened ×2 (size inflated by a zero-fill bug; clash part is entry-wide) |
| **Framework Cα deviation from a reference** (run excluded) | 1.56 vs 1.02 Å | OR 3.2 per SD; within paper OR 10 | Weakened (confounding), confirmed (integrity) |
| **VH error predicts VL error in the same entry** | 34% vs 1.3% when VH has a CDR3/FR4 slip | Paper/antibody-level OR 15–25. Any VH error predicts any VL error, OR about 13–18. | Confirmed (confounding), weakened (integrity) |
| **Less β-sheet across the V domain**, >5 residues from the error | Median 0.39 vs 0.44 | OR 0.54 per 0.08; error chain below its clean partner in 70% of 52 pairs | Not verified; overlaps the error definition |
| **Deposited non-H3 CDRs off their canonical (predicted) conformation**, also in error-free chains of the same Fv | Backbone-bin agreement 0.76 vs 0.89 | −0.13; −0.09 in error-free chains | Not verified |

**What these signals say together.**

1. **The signal belongs to the entry or paper, not to the slipped chain.**
   - Within the same entry, the error chain is no worse than the entry's error-free V chains: Ramachandran +0.8
     (p=0.2–0.23) and framework RMSD +0.17 Å (p=0.20).
   - The cis-Pro8 deficit is largely a paper-level habit. Within paper, error chains are less often cis in 5
     papers, more often in 1, and tied in 9 (sign test p=0.22). In 7 papers every κ chain is trans.
   - The framework distortion and the VH→VL link are not specific to CDR3/FR4 slips. Other error modes show the
     same thing (framework OR 3.1–3.5; D/E vs other modes p=0.17–0.32).
2. **The excess is concentrated in a subset.** Entries in which every V chain is wrong reach about 14–16%
   non-favoured residues. Dropping the most extreme 23% of cases removes the beyond-site geometry effect.
3. **Antigen chains are mostly fine.** The non-antibody chains of error entries have normal Ramachandran
   statistics. Their clash excess is real but paper- or lab-level: it vanishes within paper.
4. **Interpretation.** The pattern fits antibody models that were traced or threaded without antibody-specific
   priors and refined loosely (no cis-prolines, loose backbone), rather than docked from a correct antibody
   template and refined. This cannot be tested from the metadata: 65 of 70 κ error chains record no starting
   model.
5. **The constant-domain part does not hold.** The cis deficit in the constant domains (Cκ 21% vs 81%) rests
   mainly on one 2021 paper (12 entries, 7CZP–7D0D) that contains no cis peptides anywhere, in the spike
   included. Without it, CH1 is null and Cκ is weak (verdict: weakened ×2).
6. **Practical use: a flag, not a diagnosis.** A trans κ Pro8 raises the error rate from 0.26% to 5.7%. The flag
   is sensitive but not specific, since 36% of clean cryo-EM κ chains are also all-trans.

### 4.2 Moderate: local hazards at the slipped strand

These describe *where* a poorly built model slips. Each is a hazard, not a determinant: controls with the same
hazard were often built correctly.

- **Length mismatch between the starting point and the antibody.**
  - *Case files.* Templates with a different CDR1, CDR2, CDR3 or FR3 length often give a shift equal to the
    length difference: 3JBF −3, 7WVL +1, 9DM7 +1, 7K8X, 7V2A, 6ORP, 12TY (nanobody re-sequenced from a 17- to a
    15-residue CDR3, +2).
  - *Case-control comparison.* At the case-file level, a template length mismatch at the site is present in 24%
    of cases vs 18% of controls (p=0.56). The mechanism is clear in individual cases but does not separate the
    groups.
- **"Canonical pull" (literature_dean; no adversarial check).** In error chains, the CDR length implied by the
  coordinates is more common in the curated reference than the true sequence length.
  - 56/107 vs 36.5 expected.
  - The detector shows no such bias in about 15,000 clean chains.
  - **The synthesis recheck survives artifact removal:** 46/97 vs 31.5 without amyloid; 39/84 vs 27.0
    (p=1e-4) without all strict artifacts; cryo-EM 31/66 vs 21.5 (p=0.001) (`synthesis/canonical_pull_recheck.csv`).
  - This fits building or refining toward a common-length template.
  - Not excluded: misregistrations toward a common length could simply be easier to detect.
- **Chain ends and unbuilt neighbours.**
  - Mode-A slips have a truncated SEQRES N-terminus in 32% vs 7.8% of controls (OR 5.3).
  - Verifiers found this is nested inside the known "N-terminus not built" factor. Adjusting for where the model
    starts leaves OR 1.2–3.8, and the effect is absent in λ chains.
  - FR4-truncated SEQRES likewise disappears once "model ends inside FR4" is adjusted for.
  - Verdicts: weakened ×3. Treat these as one known factor, not new links.
- **Common but non-modal CDR lengths and long-L1 κ chains.**
  - Heavy CDR2 IMGT 7 (North H2-9) vs 8: 8.0% vs 3.4%, OR 2.45. The synthesis recheck gives the same OR after
    removing artifact units (p=7e-5).
  - κ L1 IMGT 12 (L1-17): OR 3.1–3.6.
  - Mouse IGKV8-30 (long L1): 4/41 vs 4/333, post hoc.
  - These are common forms next to a dominant mode, not rare ones. Not adversarially verified.

### 4.3 Weak to moderate: who and what was being built

- **Teams new to antibody cryo-EM** (no audit author with a prior antibody cryo-EM entry): 13.3% vs 7.9% error
  rate, OR 2.0, robust to about 25 adjustments.
  - The effect is confined to 2022–23 (OR 4.7–8.4) and is absent ≤2021 and ≥2024. It does *not* explain the
    2020–21 peak.
  - It is absent in nanobody studies (OR 0.82).
  - With one study per lab, median OR 1.6 (p=0.11).
  - It depends on how common surnames are handled.
  - Verdict: weakened ×2. A period-specific association, not a general one.
- **Capsid and enzyme targets.**
  - *Capsids:* entry OR 4.7, but 7–13 of the 32 capsid error entries are numbering artifacts. The "12 labs" are 7
    co-author groups. After removing artifacts and the two largest groups, OR ≈ 1.0. No effect within labs, and
    none for human antibodies (OR 0.98).
  - *Enzymes:* entry OR 3.8. It holds up better (leave-one-lab-out OR 3.1–3.6) but depends on a residual class
    definition: an EC-number definition gives OR 1.4–1.9 (n.s.). The whole excess sits in chains with unbuilt V
    residues.
  - Verdict: weakened ×4. A few specialist groups with mostly non-human antibodies, not a target-class effect.

### 4.4 Weak descriptors of the kind of paper (survive, but not mechanisms)

| Finding | Values | Verdict / caveat |
|---|---|---|
| Fewer citations | Within-year 5.8 vs 8.1 per year, OR 0.68 | Weakened: between-lab only (within-lab OR 0.88, n.s.); crude difference null; lab-clustered q≈0.2 |
| Fewer authors per antibody built | 4.5 vs 6.5, OR 0.70 | Not verified; only after adjusting for paper size |
| Non-human/mouse germlines (alpaca, rabbit, bovine) | 28% vs 13%, OR 3.0 | Bovine is mostly one lab. Rabbit OR 3.3 rests on 7 units, one of them an artifact. |
| Lower germline identity (<85%) | OR 1.8 | Disappears after adjusting for species and nanobody status (seq_novelty M2 OR 0.94) |
| Fewer Titan Krios / 300 kV instruments before 2022 | 67% vs 78% | Clustered in a few labs or facilities |
| Smaller complexes, fewer membrane proteins | OR 0.69 per doubling of mass | Not verified; the opposite of a "tiny part of a huge complex" explanation |

## 5. Findings that were weakened or refuted

| Finding (lens) | Claimed | After verification |
|---|---|---|
| FR3START_INDEL_B (case_files_ABCF) | CDR2/FR3 indel marks mode B, OR 222 | **Refuted**: gap-placement artifact. Correctly built high-resolution structures and predicted models reproduce the run. |
| FR3_INDEL_C (case_files_ABCF) | FR3 indel marks mode C, OR 52 | Weakened to 2–4 genuine units (OR about 6–18). Most exposed units are artifacts or copies. |
| Rare IMGT column in the run window (seq_novelty) | O/E 14.9 | Weakened: the run *is* the insertion numbering. 1 confirmed deviation outside one bovine lab (O/E 1.1). |
| SEQRES N- and C-truncation (seq_novelty) | OR 5.3 and 14 | Weakened: a restatement of "the model lacks its terminal residues" |
| Fewer competitors (paper_meta) | OR 0.47 | Weakened: target class explains a third of it, and strain-specific UniProt accessions cause the rest. Family-level OR 0.75 (n.s.). |
| No competitive pressure (antigen_competition) | "Strong" null, if anything protective | Weakened: the protective direction is confounding (adjusted OR 1.07 per doubling). The data cannot exclude modest harm at SARS-CoV-2 levels (upper bound about 3× over 7 doublings). |
| Capsid target, enzyme target (antigen_competition) | OR 4.7, 3.8, "strong" | Weakened (§4.3) |
| New team (people_network) | OR 2.0, "strong" | Weakened (§4.3) |
| cis deficit in constant domains and Pro115 (pdb_meta) | Cκ OR 0.04 | Weakened: one all-trans paper; CH1 null; Pro115 marks the same all-trans V model |
| Partner-chain clashes (struct_unusual) | +5.2 per 100 | Weakened: entry-level, not specific to the partner; Ramachandran value misreported |
| Lower citations (paper_meta) | OR 0.68, q=0.03 | Weakened (§4.4) |

## 6. What showed no association

**Competition and speed**
- Other labs depositing the same antigen within ±1, ±6 or ±12 months, after adjustment: OR 1.0–1.07.
- The earliest SARS-CoV-2 entries: the first 25 were clean, but only 0.6 errors were expected, and entries
  26–50 contained 10 errors (7 from one paper).
- SARS-CoV-2 vs other antigens: 6.7% vs 6.4%.
- Preprints, preprint→publication, deposition→publication, review time, acceptance→publication, hold time.
- Structures added after submission, and later-deposited entries within the same paper.

**People and places**
- 503 individual authors, 389 non-last authors, 53 co-author groups, 46 multi-lab core-staff hubs, 274
  institutions, 12 named facilities, 19 last-author and 31 any-author countries: none reached q<0.1.
- Co-authorship links between error studies: 0.85× expected. People linking error studies across labs: 37 vs
  36.5 expected. A co-author carrying errors from another lab: OR 1.02.
- Lab persistence once the earlier error is on a different antigen (OR 1.04). The same error mode recurring
  within a lab (15 vs 19.9 expected pairs).

**Publication**
- Journal (19 groups), journal tier, open access, article type, funders (NIH, NIAID, HHMI, Gates, Wellcome,
  Chinese funders, ERC, MRC/UKRI and others), industry affiliation, title keywords.
- Tool-paper status once paper size is adjusted for.

**Map and processing**
- Map-intrinsic local resolution at the error site (48th percentile of clean chains; site minus domain 0.00 vs
  0.01 Å).
- Local, focused or composite refinement; DeepEMhancer, density modification and other post-processing.
- Particle numbers and symmetry; final reconstruction software; EM start-up model.
- Detector, energy filter, dose; processing and deposition site.
- Entry Q-score; huge complexes (>500 kDa); unmodelled fraction.

**Antibody sequence**
- Nearest earlier antibody identity after adjustment (OR 0.90).
- First-time V, J or V–J germline (V–J 3.8% vs 3.5%).
- Somatic mutation; humanisation; rare residues at canonical, Vernier and VH–VL interface positions.
- J-gene usage; rare CDR lengths overall (4.2% vs 4.3%); CDR3 length.

**Structure**
- Mean VH–VL orientation (all six ABangle parameters).
- Orientation predicted from sequence; predicted CDR-H3 base for VH.
- Ramachandran outliers proper; non-Pro cis peptides outside the site.
- Recorded starting-model type or fitting protocol as an explanation of the geometry excess.

**Methods text** (case files, open access only)
- Homology tool, AlphaFold, rigid-body fitting, ISOLDE, Rosetta, ModelAngelo, Fv-only, "poor density".

## 7. How the erroneous antibodies compare with general antibody structural statistics (Deane-group norms)

In brief: **the antibodies are ordinary; the models are not.**

| Yardstick | Erroneous antibodies | Verdict |
|---|---|---|
| VH–VL orientation (ABangle, Dunbar 2013 frame) | Mean angles and distance unchanged. More outliers (36% vs 16% outside the 99% ellipsoid), entirely explained by distorted domain cores. Orientation predicted from sequence is normal (p=0.16). | Ordinary antibody, distorted model |
| CDR length rarity (SAbDab/PyIgClassify-style X-ray ≤3 Å reference) | Rare lengths no more common (4.2% vs 4.3%, OR 0.85) | Ordinary |
| Specific canonical lengths | H2-9 (OR 2.45) and κ L1-17 (OR 3.1–3.6) over-represented: common forms that sit next to the dominant mode. λ L1-13 n.s. without amyloid. | Mildly atypical, unverified |
| Canonical CDR conformations | Deposited loops agree less with the predicted canonical backbone (0.76 vs 0.89), also in error-free chains | Model, not antibody |
| CDR-H3 base (kinked vs extended) | Predicted kinked for VH as often as controls (97% vs 98%). Deposited extended bases in errors are a product of the slip. Nanobodies predicted extended more often (OR 0.31, weak). | Ordinary, except possibly some VHH |
| κ L3 cis-Pro95 | ImmuneBuilder predicts cis for every sequence; error models built it trans in 20/24 | Model, not antibody |
| ImmuneBuilder ensemble uncertainty | Slightly higher at the error site (0.46 vs 0.32 Å), n.s. within 2020–22 | Weak intrinsic signal |
| Germline novelty and distance to earlier PDB antibodies | Not different after species, nanobody and same-antibody adjustment | Ordinary |
| Species | More non-human/mouse (nanobodies, rabbit, bovine); ANARCI germline sets are sparse for these | Atypical, but partly one lab and partly a detector issue |
| Representation in curated statistics | 7% of error units have any X-ray ≤3 Å structure of the same V domain, vs 21% of clean units. From 2021, most newly deposited V domains (46–69%) have no such structure. | The literature's statistics have essentially never sampled these antibodies (partly circular with the first-occurrence definition) |

## 8. Conclusion: is there something linking them?

**Yes, but it is not a person, a lab network, a facility, an antigen race, a journal or an unusual antibody.**

The common thread is visible in the deposited coordinates. The genuine errors sit in antibody models that were
built and refined in a way that leaves an antibody-specific fingerprint:

- the canonical κ cis-prolines are missing;
- backbone geometry in the antibody chains, but not in the antigen, is worse;
- frameworks are distorted;
- non-H3 CDRs are off their canonical conformations;
- loops are pulled toward common template lengths;
- when one chain of a Fab is wrong, the other is often wrong too.

This fingerprint belongs to the entry or the paper, not to the single slipped strand. It is spread widely: the κ
error chains alone span 26 labs and 31 papers, mostly in 2020–23. The slip then happens where this kind of model is most fragile:

- next to unbuilt termini or loops;
- where the antibody's loop or framework length differs from the template or from the most common length (H2-9,
  long-L1 κ, CDR3 length changes, rabbit and other non-human frameworks).

Three things did not pan out, and they matter for the user's hypothesis:

1. Competitive pressure is not supported in either direction once target class is accounted for, although the
   data cannot rule out a modest effect at SARS-CoV-2 levels of crowding.
2. No shared people, institutions, facilities or co-author links exist beyond chance.
3. The antibodies themselves look ordinary by Deane-group yardsticks.

About one in five flagged units (17–23%) are not register errors at all. They are amyloid fibrils, gap-placement
conventions at rare indels, chain-end numbering artifacts, or missed copies. Several apparent "links" came from
these artifacts: the λ CDR1 length signal, the "same FR3 error across labs" pattern, the bovine and capsid
clusters, and the CDR2/FR3 indel signal.

These are associations. We cannot say which building tools or workflows produce the fingerprint, because
starting-model and software records are sparse: 65 of 70 κ error chains record no starting model. The next step
would be to test the fingerprint prospectively, using all-trans κ Pro8 and an antibody-vs-rest Ramachandran gap
as screening flags (a high negative predictive value, low positive predictive value), and to ask depositors about
their workflow for a sample of flagged entries.

## 9. Caveats

- **Many tests.** Hundreds of tests were run across lenses. BH q-values were computed within each lens or
  verifier family, not jointly. Several weak positives would not survive a global correction.
- **Small numbers.** About 150–180 genuine error units, 110–127 error papers, and ≤40 for most mode-specific
  tests. Period-specific estimates before 2020 are imprecise.
- **Overlap with the error definition.** A misregistered strand lowers sheet content, raises local geometry
  outliers and lowers Q at the site. Away-from-site, error-free-chain and within-entry versions reduce this
  overlap but cannot remove it.
- **Control selection.** Controls are entries with no disagreement run of any kind, which excludes entries whose
  runs were all copied or artifacts. This can favour carefully built entries and inflate some odds ratios (for
  example FR3 indels: OR 52 → 31 with all non-C units as controls).
- **Artifact list.** The combined list in `synthesis/` merges flags from different verifiers. It was not itself
  re-verified. The "loose" category includes D/E runs next to unbuilt FR4 that may be real. Other indel patterns
  (FR3 gaps at 85, 89, 94, 66–68, 74–75; κ CDR2 59–62) are unconfirmed and could be further artifacts.
- **Unverified findings.** Canonical pull, specific CDR lengths, canonical-CDR deviation, sheet deficit, complex
  size and local resolution were not adversarially verified. The synthesis rechecks cover only amyloid and
  artifact removal.
- **Missing metadata.** Starting model, fitting protocol and software are missing for most entries. Annotator
  identity is not public. Local resolution exists only for entries with half maps (about 39%).
- **Name matching.** Author identity relies on name keys (ORCID for 57% of author slots), so collisions and
  splits affect the people-level results. People-level tables are pseudonymised.
- **Causation.** None of these analyses establishes causation.

## 10. Open gaps

A completeness critic reviewed this document afterwards (`completeness_critic/gaps.md`). Its main points:

1. About 70 remaining units (modes B, C, F, D0, other) have never been confirmed by a test independent of SAbR,
   and these are the modes where gap-placement artifacts were found.
2. The headline associations were not re-run on a single cleaned error set.
3. The all-trans κ Pro8 fingerprint is increasingly common in clean models (30% of κ units before 2020, 44% from
   2024), while the error rate among all-trans models fell from 12.9% to 3.5%. It is a near-necessary marker with
   a falling positive predictive value; what separates the all-trans models that slipped has not been examined.
4. The 2020–21 peak is not explained, and much of each half-year bump comes from one or two labs.
5. Methods text was coded only for the case files; error registers in antigen chains, map-level checks of
   individual errors and same-antibody natural experiments were not done.

Eight verifiers returned a verdict without writing `notes.md`; their notes were written afterwards from the
verdicts recorded in the workflow log.

## 11. Files

**Synthesis (this document)**
- `synthesis/s01_artifact_ledger.py` → `artifact_ledger_runs.csv`, `artifact_ledger_summary.txt`: combined
  artifact and copy flags per first-occurrence run, and unit counts after removal.
- `synthesis/s02_canonical_pull_recheck.py` → `canonical_pull_recheck.csv`: canonical pull without amyloid and
  artifacts.
- `synthesis/s03_length_tests_recheck.py` → `length_tests_recheck.csv`: H2-9, λ L1-13 and κ L1-17 tests without
  amyloid and artifacts.

**Lens write-ups** (each has scripts and tables alongside)
- `pdb_meta/notes.md`
- `paper_meta/notes.md`
- `antigen_competition/notes.md`
- `seq_novelty/notes.md`
- `struct_unusual/notes.md`
- `people_network/notes.md`
- `literature_dean/notes.md`
- `case_files_ABCF/notes.md` and `case_files_ABCF/case_files.md`
- `case_files_DE/notes.md` and `case_files_DE/tables/case_files_DE.txt`

**Adversarial checks**
- `verify/<lens>_<finding>_{confounding,integrity}/notes.md`, with scripts and BH tables.

**Key artifact evidence**
- `pdb_meta/tables/amyloid_fibril_chains.csv`
- `case_files_ABCF/tables/hires_gap_check.csv` and `artifact_runs.csv`
- `verify/case_files_ABCF_ARTIFACT_INDEL_integrity/`
- `verify/case_files_DE_DE1_cterm_artifact_{confounding,integrity}/tables/` (truncation experiment)
- `verify/case_files_ABCF_FR3START_INDEL_B_confounding/tables/v08_feature_units_evidence.csv`
