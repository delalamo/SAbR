# Verification of paper_meta PM0_amyloid_artifact (confounding lens)

**Verdict: confirmed, with small corrections.** The 15 AL light-chain fibril entries are detector artifacts.

None of the following explains the 15/15 flag rate:
* year;
* method;
* resolution;
* chain type (λ);
* SARS-CoV-2 status;
* paper clustering;
* poor local map;
* unbuilt residues.

The fold evidence does not depend on SAbR at all.

## Reproduction (`v01_counts.py`)
* A broad title regex, `amyloid(?!-beta…)|fibril`, finds exactly the 15 entries. All are cryo-EM λ chains, and all are `err_first`. No X-ray entry and no clean entry matches.
* The following counts reproduce exactly:
  * 12/183 cryo-EM error units;
  * 77/488 first-occurrence runs (15.8%; 18.0% of the 428 cryo-EM runs);
  * 6IC3 at IMGT 16–122, Δ −48.
* **Correction 1:** the modes by entry are A_FR1Nterm **7** (not 8), F_CDR1 2 and other 6, for a total of 15.
  * The labels are arbitrary. 9LIV and 9LIW are the same patient and the same sequence, but they are labelled F_CDR1 and A_FR1Nterm.
  * The amyloid entries make up 6 of the 12 cryo-EM "other"-mode entries and 35 of the 47 "other" runs.
* **Correction 2:** the "10 amyloid papers" in paper_base are really **7 papers from 6 labs**. 9FAB, 9LJA and 9LJB have no DOI, so each became its own `nodoi:` pseudo-paper. The same split affects every unpublished entry in the paper-level analyses.

## Co-author claim (`v02_author_enrichment_with_amyloid.py`)
I put the amyloid papers back in as error papers and re-ran network.py's per-author test.
* The Fändrich/Schönland group (Radamaker, Huhn, Fändrich, Schönland, Schmidt, Hegenbart) is reproduced: 3/3 papers, 0.18 expected, p = 2.0e-4, **q = 0.056**.
* But Wang K (7/12 papers, 6 labs, no amyloid) is essentially tied: p = 2.5e-4, q = 0.058.
* So "top co-author hit" holds only by a hair, and neither hit reaches q < 0.05.
* The amyloid set is the strongest link in effect size (15/15 entries against about 0.8 expected), not in this co-author test.

## Fold test, independent of SAbR (`v03_fold_geometry.py`)
I took the CA atoms of residues numbered by ANARCI and measured the thickness of the CA cloud (SD along the third principal axis).
* AL fibril chains: 1.0–3.0 Å (median 1.3; n = 77 chains from 15 entries).
* All non-AL cryo-EM chains: 5.0–7.8 Å. This covers one chain from each of the 171 non-amyloid error units plus 180 clean chains.
* The two groups do not overlap at all (Mann-Whitney p = 5e-11 by entry).
* Radius of gyration: 21.5–28 Å for fibril chains vs 12–23 Å for the rest.
* The four non-amyloid chains where SAbR and ANARCI disagree at more than 50% of residues are 3JCX H, 7C80 H, 9PV3 L and 9YC6 L. All four are Ig-folded (thickness 5.4–6.7 Å), so they are candidate whole-chain register shifts, not fold artifacts.
* The title-based exclusion is therefore complete for non-Ig folds in this set.

## Confounders (`v04_confounding.py`, BH q over 10 tests)
* **Expected count.** A model fitted on non-amyloid cryo-EM entries (`err ~ year bin + res + log2 nV`, clustered by paper) predicts **0.77** flagged entries among the 15. All 15 are flagged (p = 3e-20).
  * Adding λ chain and SARS-CoV-2 status: 0.88 expected.
  * Also adding "no heavy chain": 2.6 expected (p = 3e-12).
  * By paper: 7/7 flagged vs 0.73–2.2 expected (p ≤ 9e-5).
* **Matched λ units** (cryo-EM, 2018+, 3.0–3.95 Å): AL fibrils 12/12 flagged vs 9/390 (2.3%) for other λ units (p = 9e-18). The gap holds in every period.
* **Map fit.** The AL entries fit their maps *better* than clean entries: lq_z median 0.90 vs 0.40, adjusted +0.62 (0.41 to 0.82), p = 5e-9. Real errors fit worse (−0.27, p = 0.017). Poor density cannot explain the flags.
* **Extent of disagreement.** SAbR and ANARCI disagree at more than 50% of V residues in 77/77 AL chains, against 4/322 non-AL cryo-EM error chains. AL runs cover the whole domain, so 96% of them count as "next to a chain end or unbuilt loop" almost by definition.

## What leaving them in does (why every lens must exclude them)
* **λ vs κ, units, cryo-EM 2018+.** With AL included, λ looks riskier: 5.1% vs 3.2% (OR 1.6, p = 0.09). Without AL: 2.6% vs 3.2% (OR 0.8, p = 0.64).
* **Error rate by year.** AL inflates the late years: 2024 is 4.8% with AL and 4.1% without; 2025 is 5.8% with and 4.1% without.
* **"Next to unbuilt" share of cryo-EM runs:** 46% with AL vs 35% without. The prior "48% next to unbuilt residues" figure should be checked for this.
* **Other skews:** the "other" mode, the "antibody alone" role, and IGLV1-44 / IGLV3-1 / IGLV2-14.

## Coverage caveat
* Two AL fibrils inside the 3–5 Å window, **6HUD and 8CPE** (both 4.0 Å, 77 modelled residues), were scanned by SAbR but are missing from `chains.csv`. SAbR and ANARCI disagree at 90–100% of their residues. They are also flat (thickness 1.9 Å).
* **6Z1I** (3.4 Å, the conformation B partner of 6Z1O) was never scanned. It is also flat (1.3 Å).
* A lens with a looser residue filter would pick these up, so the exclusion should be by fold (or by PDB keyword PROTEIN FIBRIL plus Ig light chain), not by a title regex.
