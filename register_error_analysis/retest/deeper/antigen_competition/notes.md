# Antigens and competitive pressure

**Question.** Are register errors linked by what the antibody binds, or by how many other labs were racing
on the same antigen at the time?

**Answer.** There is no sign of competitive pressure. Errors are *less* common on crowded antigens, and the
early SARS-CoV-2 race was clean. What does link errors is the kind of target: **virus capsids** and
**enzymes** carry 4–5× the error rate of other antigens. This holds across many labs, and it explains all of
the clustering of errors by antigen.

## Data built for this lens

**Antibody chains across the whole PDB.** Every SEQRES chain that passed a V-domain motif prefilter (28,361
unique sequences) was run through ANARCI. This found 39,154 antibody chains in **11,972 entries**, at all
resolutions and methods (`s01`–`s03`). The prefilter keeps 99.3% of the antibody sequences already known to
the project.

**Entry metadata.** For every one of these entries, RCSB GraphQL supplied (`s04`, `s05`):
* the polymer entities, with UniProt accession, organism, Pfam annotation and 30% identity cluster;
* the assembly-1 interfaces, with buried area per entity pair;
* UniProt keywords for 3,161 accessions.

**Antigen.** Light chains were paired with their heavy chain. The antigen is any non-antibody entity whose
interface with that antibody is at least 150 Å² (`s06`).

**Status of each entry:**
* **bound**;
* **fiducial-only**: every antibody touches only G-protein subunits, BRIL, MBP, T4L, GFP or a megabody
  scaffold. The entry's target is then its subject protein, for example the GPCR;
* **unbound**: antibody alone;
* **fusions**: a nanobody fused to its receptor counts as bound.

**Target class** comes from UniProt keywords and names. The classes are viral surface glycoprotein, viral
capsid, other viral protein (in practice viral enzymes), GPCR, channel/transporter, enzyme, toxin,
cytokine, receptor/membrane protein, and other.

**Family** is the connected component linking each UniProt accession to its 30% identity cluster.

**Competition** is counted from all PDB antibody entries (`s07`), for each target:
* the number of *other* labs (normalised last author) that deposited antibody structures containing the same
  accession or the same family within ±1, ±6 and ±12 months;
* whether the entry was within 30 days of the first antibody structure of that antigen or family;
* the antibody entries, and all PDB entries (SIFTS), of that antigen deposited in the preceding 12 months.

**Comparison.**
* **Error:** an entry with a first-occurrence error (212 entries; 178 cryo-EM).
* **Clean:** an entry with no run of any kind (3,782).
* **Excluded:**
  * copied-only and artifact-only entries;
  * the **15 AL amyloid fibril entries**. These are λ light chains that are not in an Ig fold, so they are
    detector artifacts. The fibrils were found independently here as "antibody alone" entries, with 17 of 42
    unbound cryo-EM entries flagged.

**Statistics.**
* **Models:** logit models with deposition year and resolution, standard errors clustered by paper, mostly
  within cryo-EM. Paper-level models add log(number of entries).
* **Sparse cells:** Fisher's exact test.
* **Multiple testing:** one Benjamini–Hochberg correction over all 241 tests (`tables/all_tests_bh_lens.csv`).

## Findings (cryo-EM unless stated)

### 1. Competitive pressure does not raise error rates

Measured as other labs on the same antigen within ±6 or ±12 months, errors are no more common. If anything,
they are less common.

| Measure (error vs clean, adjusted OR per doubling) | All EM | EM, no SARS-CoV-2 | X-ray |
|---|---|---|---|
| Other labs, same accession, ±12 months | 0.94 (p = 0.22) | 0.82 (p = 0.10) | 1.03 (p = 0.69) |
| Other labs, same family, ±12 months | 0.90 (q = 0.08) | **0.82 (q = 0.014)** | 0.99 |
| Another lab within ±1 month ("race") | OR 0.90 (p = 0.67) | 0.96 (p = 0.89) | 1.21 (p = 0.64) |
| All PDB structures of the antigen in the prior year | 0.95 (p = 0.18) | 0.88 (p = 0.08) | 1.05 (p = 0.52) |

**SARS-CoV-2.** SARS-CoV-2 was the most contested antigen in PDB history: a median of 143 other labs within
±12 months. Its spike family has an ordinary error rate, 6.7% vs 6.4% (OR 0.91, p = 0.74). The first 25
SARS-CoV-2 antibody EM entries (April–August 2020) were all clean.

### 2. "Niche" antigens in 2020–21, not crowded ones

Paper-level, non-SARS-CoV-2 cryo-EM papers whose target no other lab touched within ±12 months:

| Period | Niche papers with an error | Contested papers with an error | Adjusted OR |
|---|---|---|---|
| 2020–21 | **21% (19/89)** | 4% (4/100) | **8.3** (2.2–31), q = 0.013 |
| 2022+ | 10% | 8% | 1.06 (p = 0.83) |
| ≤2019 | 12% | 8% | n.s. |

* The 2020–21 estimate is adjusted for capsid/enzyme target, lab antibody experience and new V regions.
* The 19 error papers are listed in `tables/niche_by_period.log`. Ten of them are capsid or enzyme targets.
* "First antibody structure of the family": OR 2.2 (q = 0.014). This falls to 1.3–1.4 (n.s.) once lab
  experience and new V regions are added.
* **Read this as a pandemic-period niche effect, not a race effect.** It is confined to one period.

### 3. Capsids and enzymes are where errors concentrate

| Target class (EM) | Entries with an error | Papers with an error | Adjusted OR (entry) |
|---|---|---|---|
| **Viral capsid** | **20.8% (32/154)** | **29% (18/62), 12 labs** | **4.7** (2.5–8.8), q = 2e-5 |
| **Enzyme** | **18.5% (15/81)** | **31% (15/49), with viral enzymes** | **3.8** (1.8–8.0), q = 0.003 |
| Viral enzyme ("viral other") | 50% (8/16) | 2 labs (Grimes, Zimmer) | 11.3, q = 7e-4 |
| Viral surface glycoprotein | 5.9% | 9.3% | 0.71 (n.s.) |
| GPCR | 3.9% | 5.6% | 0.59 (n.s.) |
| Channel/transporter | **2.7%** | 5.5% | **0.39** (q = 0.07) |
| Fiducial-only (Nb35, scFv16, BAG2...) | **1.0% (2/203)** | | **0.17** (q = 0.002) |

**Robustness.**
* **Paper level:** capsid OR 4.4 (q = 3e-5) and enzyme OR 4.7 (q = 6e-5).
* **Leave one lab out:** the capsid OR stays at 3.9–5.0 and the enzyme OR at 4.5–4.9. With the two largest
  error labs removed, they are 3.2 and 4.0.
* **Covariates:** the effects survive local map fit (`lq_z`), nanobody, fitting protocol, mouse origin,
  germline identity, lab antibody experience and new V regions. The capsid OR is 4.8–5.7 and the enzyme OR
  4.5–7.1.
* **Not the same as other lenses:** capsid and enzyme are *not* more often new V regions (70% and 65% vs
  65%), and their maps are no worse on average.

**What differs:**
* **Capsids:**
  * the antibodies are mostly mouse (56% vs 18%);
  * the errors are dominated by **CDR1 boundary errors (9 of the 14 EM F_CDR1 entries)** and in-CDR3 D0
    runs (7 of 10);
  * within capsid entries, the error chains fit their maps worse (lq_z −0.11 vs +0.21, p = 0.007).
* **Enzymes:**
  * the antibodies are more often nanobodies (43% vs 27%), with lower germline identity;
  * rigid-body-only fitting is more common (28% vs 10%);
  * CDR2/FR3 (B) errors are over-represented (7 entries).

**Antigen clustering.** Errors cluster by antigen family beyond year and resolution:
* entry level: dispersion 2.1×, p = 0.002;
* paper level: 1.35×, p = 0.016;
* paper level, once capsid/enzyme is in the null model: **1.18× (p = 0.14); 1.12× without SARS-CoV-2**.

No other antigen stands out:
* HIV Env is *lower*: 3.4% vs 6.8%, OR 0.42 (q = 0.07);
* influenza HA is at baseline (6.6%);
* the single "Capsid protein" family with OR 10 is one paper (Zhu P., 6 of 15 entries).

### 4. Lab experience, a small effect

* Each doubling of the lab's earlier antibody structures gives OR **0.88** (q = 0.013).
* Labs with ≥20 prior antibody entries: 3–4%. Labs with 1–4: 9–15%.
* Labs with *no* prior antibody structure are not worse (OR 1.06).
* Lab experience does not explain the capsid, enzyme or 2020–21 niche effects.

### 5. Auxiliary vs subject antibodies

* **Entry level:** fiducial-only entries have 1.0% errors.
* **Unit level, by contacts:** fiducial binders 1.4% (2/139) vs subject antibodies 4.2%. OR 0.42, p = 0.13,
  not significant.
* **Antibodies that bind no antigen** in a bound entry: 4.1%, the same as subject antibodies.
* **Custom tool antibodies:** 597 antibodies that the earlier `aux_reagent` text rule called "auxiliary"
  actually bind a non-fiducial antigen. Many enzyme and membrane-protein Fabs or nanobodies are custom tools,
  and those are the ones with high rates.

## What showed no difference

* Other labs on the same accession within ±6 or ±12 months: all EM, non-SARS-CoV-2 EM, 2022+, ≤2019, and
  X-ray.
* "Race": another lab within ±1 month.
* All-PDB structures of the antigen in the prior year.
* The first-ever PDB structure of the antigen (first_allpdb): p = 0.17. The first antibody structure of the
  accession is not significant after adjusting for class (OR 1.12).
* SARS-CoV-2 spike, influenza HA, GPCRs, receptors, toxins, cytokines and peptides.
* Unbound antibody-alone entries, after the amyloid fibrils are removed.
* Bacterial, viral and other-eukaryote target organisms. Human targets are lower (OR 0.49, q = 0.04), but
  that is a class mix of membrane proteins.
* New antibody lab (no prior antibody structure).
* X-ray: no class or competition effect (31–34 errors, underpowered).

## Caveats

* **Heuristics:**
  * the class rules are heuristic (UniProt keywords and names);
  * polyprotein accessions merge capsid with polymerase domains (picornaviruses and flaviviruses);
  * class A GPCRs fall into one 30% family.
* **Interfaces:** the antigen comes from RCSB assembly-1 interfaces, and 40 entries lack interface data.
* **Labs:** a lab is the normalised last audit author, so name variants or shared PIs can blur it.
* **Dates:** competition uses PDB deposition dates, not preprint or submission dates.
* **Small numbers:** the 2020–21 niche result rests on 23 error papers, and it is a period-specific subgroup.
* **Overlap with other lenses:** capsid and enzyme entries overlap with antibody-level features (mouse
  antibodies, nanobodies). Entry-level adjustment does not remove the effects, but chain-level residual
  confounding is possible.
* **Not usable:** the Fv-only check could not be done, because the SAbR records hold V-region residues only.

## Files

* **Scripts:** `s01`–`s15`, run in order.
* **Tables (`tables/`):**
  * per-entry antigen and competition features, PDB-wide (`pdbwide_ab_entries_antigen.csv.gz`) and for the
    focal set (`focal_entries_antigen_features.csv.gz`);
  * per-chain antibody role (`pdbwide_ab_chain_roles.csv.gz`, `units_antibody_role.csv.gz`);
  * test tables (`competition_tests.csv`, `class_organism_tests.csv`, `antigen_family_tests_em.csv`,
    `followup_tests.csv`, `lab_experience_tests.csv`, `niche_by_period.csv`, `novel_v_checks.csv`,
    `extra_checks.csv`);
  * all tests with lens-wide q values (`all_tests_bh_lens.csv`);
  * **one row per error paper**, with antigen, class, competitors and first-structure status
    (`error_papers_antigen_competition.csv`);
  * logs (`*.log`).
* **Cache:** downloads are under `scratchpad/deeper/cache/antigen_competition/`.
