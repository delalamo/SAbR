# Case files: modes A (FR1 N-terminus), B (CDR2 to FR3), C (FR3) and F (CDR1)

There are 29 first-occurrence error entries (mostly cryo-EM; one per lab) and 45 matched, correctly modelled controls.
The controls are of three kinds: same antigen, same period and target class, or a "near miss" that shares the local hazard.

**Column key:**
* **fit**: site map fit, as a percentile within the V domain.
* **unbuilt**: unbuilt residues within ±5 of the site.
* **contact**: the site contacts a non-antibody chain.
* **indel**: an atypical length or indel at the site, read from SEQRES.
* **template**: the closest antibody template named in the paper or PDB record. *mismatch* means it differs in length at the site.

Scripts are `s01`–`s12`; the tables are in `tables/`. The method-text snippets are in `methods_snippets.txt`.

## A_FR1Nterm

### 6ad0 chain L: Cheng, T., 2018, EM 3.90 Å
* **Antigen:** Genome polyprotein.
* **Run:** IMGT 1–32, sequence `DVLMTQTPLSLPVSLGHQASISCRSSQSIVHS`.
* **Site:** fit 0.37; 0 unbuilt residues; first built residue IMGT 1 (SEQRES starts at 1); antigen contact 1.
* **Antibody:** IGKV1-117*01 (identity 0.95, mouse); indel at site 0; gap-placement artifact 0.
* **Template:** 5xs7; mismatch 0. Open-access text: True.
* **Same V region elsewhere in the PDB:** 0 before, 0 after.
* **Same antigen, other labs, ±6 months (in the scan):** 0 entries, of which 0 have errors.
* **Clue:** Homology model of the Fab from an earlier CVA6 immune complex (5XS7; Discovery Studio), docked into a segmented capsid map; only the V domains were built. The mouse IGKV1-117 light chain has an 11-residue CDR-L1, the template light chain a 6-residue one. The run (IMGT 1-32, +1) covers FR1 and stops in CDR1, where the lengths differ.
* **Control (period/class): 6aj7/L, Li, S.W., 2018.** Window `DIQMIQSPSSMFASLGDRVSLSCRASQDI`; fit 0.45; 7 unbuilt; first built residue 2; indel 0; template none named (mismatch 0).
* **Control (near-miss (L)): 6mjz/L, Kwong, P.D., 2018.** Window `QMTQSPPSLSAYVGDRVTITCRASQAI`; fit 0.50; 0 unbuilt; first built residue 3; indel 1; template none named (mismatch 0).

### 7czu chain K: Zhou, Q., 2020, EM 3.40 Å
* **Antigen:** Spike glycoprotein.
* **Run:** IMGT 1–9, sequence `DIQMTQSPS`.
* **Site:** fit 0.24; 0 unbuilt residues; first built residue IMGT 1 (SEQRES starts at 1); antigen contact 1.
* **Antibody:** IGKV1-33*01 (identity 1.00, human); indel at site 0; gap-placement artifact 0.
* **Template:** 7c2l; mismatch 0. Open-access text: True.
* **Same V region elsewhere in the PDB:** 1 before, 2 after.
* **Same antigen, other labs, ±6 months (in the scan):** 133 entries, of which 9 have errors.
* **Clue:** Chainsaw model from 7C2L (a different antibody), MDFF into the whole-spike map and the focused RBD-Fab map. The N-terminus is built from IMGT 1 but the strand is one residue off (1-9, -1), with poor site fit (24th percentile). The same slip is in the sibling entries 7czr/7czs/7czv/7d00 of the same paper.
* **Control (same antigen): 7cyp/D, Zhu, L., 2020.** Window `IVLTQSP`; fit 0.43; 0 unbuilt; first built residue 2; indel 1; template none named (mismatch 0).

### 7jhg chain L: Melcher, K., 2020, EM 3.47 Å
* **Antigen:** 5'-AMP-activated protein kinase catalytic subunit alpha-1.
* **Run:** IMGT 3–10, sequence `QMTQSPSS`.
* **Site:** fit 0.26; 2 unbuilt residues; first built residue IMGT 3 (SEQRES starts at 1); antigen contact 0.
* **Antibody:** IGKV1-39*01 (identity 0.91, human); indel at site 0; gap-placement artifact 0.
* **Template:** 6cmo; mismatch 0. Open-access text: False.
* **Same V region elsewhere in the PDB:** 0 before, 1 after.
* **Same antigen, other labs, ±6 months (in the scan):** 0 entries, of which 0 have errors.
* **Clue:** Recorded starting models include 6CMO, a Fab from the Xu H.E. lab at the same institute. Its light chain is 98% identical and carries the same N-terminal slip (-2). This is a near-identical template copy, missed by the identical-sequence copy rule. The same applies to 7jhh and 7m74.
* **Control (period/class): 7chh/E, Wang, G., 2020.** Window `VMTQSPLS`; fit 0.58; 0 unbuilt; first built residue 1; indel 0; template 7chb (mismatch 0).
* **Control (near-miss (D)): 6ww5/D, Wang, D.N., 2020.** Window `QMTQSPSS`; fit 0.47; 3 unbuilt; first built residue 4; indel 0; template 5eii (mismatch 0).

### 7tz5 chain N: Mousa, J.J., 2022, EM 3.41 Å
* **Antigen:** Hemagglutinin.
* **Run:** IMGT 2–31, sequence `SVLTQPPSVSGAPGQRVTISCTGSSSNIG`.
* **Site:** fit 0.35; 0 unbuilt residues; first built residue IMGT 2 (SEQRES starts at 2); antigen contact 0.
* **Antibody:** IGLV1-40*01 (identity 0.97, human); indel at site 1; gap-placement artifact 0.
* **Template:** none named; mismatch 0. Open-access text: False.
* **Same V region elsewhere in the PDB:** 0 before, 0 after.
* **Same antigen, other labs, ±6 months (in the scan):** 0 entries, of which 0 have errors.
* **Clue:** No starting model recorded and no open-access text. SEQRES starts at IMGT 2. The whole FR1 is shifted -1 (2-31). The period control 7whz (another lab) has an identical IGLV1-40 window sequence and was built correctly.
* **Control (period/class): 7whz/L, Wang, S., 2022.** Window `SVLTQPPSVSGAPGQRVTISCTGSSSNIG`; fit 0.49; 0 unbuilt; first built residue 1; indel 0; template none named (mismatch 0).
* **Control (near-miss (J)): 7wti/J, Fu, W., 2022.** Window `ALTQPASVSGSPGQSITVSCTGTSSDLG`; fit 0.59; 0 unbuilt; first built residue 3; indel 1; template none named (mismatch 0).

### 8e8l chain H: Charnesky, A.J., 2022, EM 3.13 Å
* **Antigen:** Genome polyprotein.
* **Run:** IMGT 4–16, sequence `LVQSGAELKKPG`.
* **Site:** fit 0.25; 0 unbuilt residues; first built residue IMGT 4 (SEQRES starts at 4); antigen contact 0.
* **Antibody:** IGHV1-46*01 (identity 0.81, human); indel at site 1; gap-placement artifact 0.
* **Template:** none named; mismatch 0. Open-access text: True.
* **Same V region elsewhere in the PDB:** 0 before, 4 after.
* **Same antigen, other labs, ±6 months (in the scan):** 0 entries, of which 0 have errors.
* **Clue:** ABodyBuilder model, built in ISOLDE then Phenix in an icosahedral poliovirus map. The construct starts at IMGT 4 (SEQRES truncated), so the strand has no N-terminal anchor (4-16, -1).
* **Control (period/class): 7x49/H, Xia, N., 2022.** Window `LQQSAAELARPG`; fit 0.41; 1 unbuilt; first built residue 2; indel 0; template none named (mismatch 0).
* **Control (near-miss (N)): 7yub/N, Duan, Y., 2022.** Window `LVESGGGLVQPG`; fit 0.43; 2 unbuilt; first built residue 3; indel 0; template none named (mismatch 0).

### 8rvn chain L: Bowden, T.A., 2024, EM 3.50 Å
* **Antigen:** Fusion glycoprotein F0.
* **Run:** IMGT 4–17, sequence `LTQTPASVEAAVGG`.
* **Site:** fit 0.29; 3 unbuilt residues; first built residue IMGT 4 (SEQRES starts at 1); antigen contact 0.
* **Antibody:** IGKV1-9*01 (identity 0.67, human); indel at site 0; gap-placement artifact 0.
* **Template:** 6t3f; mismatch 0. Open-access text: False.
* **Same V region elsewhere in the PDB:** 0 before, 0 after.
* **Same antigen, other labs, ±6 months (in the scan):** 2 entries, of which 0 have errors.
* **Clue:** Starting model 6T3F (a different Fab, 81% identity, same lengths at the site). Three N-terminal residues are unbuilt, so the built strand starts at IMGT 4 (4-17, -1). The V region is far from germline (67%).
* **Control (same antigen): 9cin/K, Bajic, G., 2024.** Window `LTQSPATLSLSPGE`; fit 0.51; 0 unbuilt; first built residue 2; indel 1; template none named (mismatch 0).
* **Control (near-miss (D)): 8vr9/D, Koide, S., 2024.** Window `MTQSPSSLSASVGD`; fit 0.35; 3 unbuilt; first built residue 4; indel 0; template none named (mismatch 0).

### 9mnb chain E: Rohou, A.L., 2024, EM 3.00 Å
* **Antigen:** Tryptase alpha/beta-1.
* **Run:** IMGT 3–9, sequence `SVEESRG`.
* **Site:** fit 0.20; 0 unbuilt residues; first built residue IMGT 3 (SEQRES starts at 3); antigen contact 0.
* **Antibody:** IGHV4-59*07 (identity 0.60, human); indel at site 1; gap-placement artifact 0.
* **Template:** 6vvu; mismatch 1. Open-access text: True.
* **Same V region elsewhere in the PDB:** 0 before, 0 after.
* **Same antigen, other labs, ±6 months (in the scan):** 0 entries, of which 0 have errors.
* **Clue:** An AlphaFold2 model of the rabbit Fab E82.AS was the starting model, then ISOLDE and Phenix. The rabbit VH sequence starts at IMGT 3 (QSVEESRG), but the strand was built one residue off (3-9, -1; 20th-percentile fit). An AlphaFold start did not prevent an N-terminal slip.
* **Control (period/class): 8rla/K, Gilbert, R.J.C., 2024.** Window `QLVENGG`; fit 0.55; 0 unbuilt; first built residue 1; indel 0; template 7zmv (mismatch 0).
* **Control (near-miss (H)): 9ml7/H, Bjorkman, P.J., 2024.** Window `SLEESGG`; fit 0.28; 0 unbuilt; first built residue 3; indel 1; template 7sc1 (mismatch 1).

### 6bpe chain F: Tham, W.H., 2017, X-ray 3.34 Å
* **Antigen:** Reticulocyte-binding protein 2b.
* **Run:** IMGT 3–8, sequence `VMTQSQ`.
* **Site:** fit 0.14; 2 unbuilt residues; first built residue IMGT 3 (SEQRES starts at 1); antigen contact 0.
* **Antibody:** IGKV6-15*01 (identity 0.95, mouse); indel at site 0; gap-placement artifact 0.
* **Template:** 5en2; mismatch 0. Open-access text: False.
* **Same V region elsewhere in the PDB:** 0 before, 0 after.
* **Same antigen, other labs, ±6 months (in the scan):** 0 entries, of which 0 have errors.
* **Clue:** X-ray molecular replacement with 5W53/5EN2. Two N-terminal residues are unbuilt. Run 3-8 (-1) with RSCC at the 14th percentile and a site B-factor z of +1.9.
* **Control (period/class): 6b0e/A, Julien, J.P., 2017.** Window `VMTQSP`; fit 0.58; 0 unbuilt; first built residue 1; indel 0; template 5d1x (mismatch 0).
* **Control (near-miss (L)): 6bkd/L, Law, M., 2017.** Window `ELTQSP`; fit 0.42; 0 unbuilt; first built residue 3; indel 1; template 6bkb (mismatch 1).

## B_CDR2FR3

### 7e39 chain C: Liu, C., 2021, EM 3.70 Å
* **Antigen:** Spike glycoprotein.
* **Run:** IMGT 62–93, sequence `GSNKYYADSVKGRFTISRDNSKNTLYLQMNS`.
* **Site:** fit 0.39; 3 unbuilt residues; first built residue IMGT 1 (SEQRES starts at 1); antigen contact 0.
* **Antibody:** IGHV3-33*01 (identity 0.98, human); indel at site 0; gap-placement artifact 0.
* **Template:** none named; mismatch 0. Open-access text: True.
* **Same V region elsewhere in the PDB:** 0 before, 0 after.
* **Same antigen, other labs, ±6 months (in the scan):** 197 entries, of which 22 have errors.
* **Clue:** No starting model recorded; symmetry expansion plus focused classification. The run spans CDR2 to the end of FR3 (62-93, +1) next to 3 unbuilt residues. The light chain of the same paper (7e3c) also slipped.
* **Control (same antigen): 7e3l/D, Yang, H., 2021.** Window `SGNTNYAQKFQERVTLTRDMSTRTAYMELSS`; fit 0.56; 0 unbuilt; first built residue 1; indel 0; template 7e3k (mismatch 0).
* **Control (near-miss (F)): 7nka/F, Grimes, J.M., 2021.** Window `GSTNFADSVKDRFTISRDATENTVVLQMNS`; fit 0.49; 5 unbuilt; first built residue 1; indel 0; template 6qpg (mismatch 1).

### 12iu chain Y: Ward, A.B., 2026, EM 3.06 Å
* **Antigen:** Hemagglutinin.
* **Run:** IMGT 58–72, sequence `SNLENLEAGVP`.
* **Site:** fit 0.19; 0 unbuilt residues; first built residue IMGT 1 (SEQRES starts at 1); antigen contact 1.
* **Antibody:** IGKV1-33*01 (identity 0.86, human); indel at site 1; gap-placement artifact 0.
* **Template:** none named; mismatch 0. Open-access text: False.
* **Same V region elsewhere in the PDB:** 0 before, 0 after.
* **Same antigen, other labs, ±6 months (in the scan):** 0 entries, of which 0 have errors.
* **Clue:** Unpublished (2026). Kappa CDR2/FR3 run 58-72 (+3) in a chain whose ANARCI gap pattern at IMGT 59-62 disagrees with SAbR in all 3 scanned chains. This is possibly a gap-placement artifact (no high-resolution example).
* **Control (period/class): 10gh/B, Klein, D., 2026.** Window `SSLYSGVP`; fit 0.55; 0 unbuilt; first built residue 1; indel 0; template none named (mismatch 0).
* **Control (near-miss (L)): 9zrt/L, Schmidt, H.R., 2025.** Window `NRLEGGVP`; fit ; 10 unbuilt; first built residue 2; indel 0; template none named (mismatch 0).

### 8cr9 chain A: Marlovits, T.C., 2023, EM 4.20 Å
* **Antigen:** Type III secretion protein PcrV; Maltose/maltodextrin-binding periplasmic protein.
* **Run:** IMGT 50–63, sequence `LEWVAGIIGNGS`.
* **Site:** fit 0.48; 0 unbuilt residues; first built residue IMGT 1 (SEQRES starts at 1); antigen contact 1.
* **Antibody:** IGHV3-23*01 (identity 0.83, human); indel at site 0; gap-placement artifact 0.
* **Template:** none named; mismatch 0. Open-access text: False.
* **Same V region elsewhere in the PDB:** 0 before, 0 after.
* **Same antigen, other labs, ±6 months (in the scan):** 0 entries, of which 0 have errors.
* **Clue:** No open-access text. FR2-CDR2 run 50-63 (-3) at the antigen contact. The same entry has FR3 (C) and kappa runs. Map at 4.2 A.
* **Control (period/class): 8i0z/D, Shukla, A.K., 2023.** Window `LEWVASISSYYG`; fit 0.55; 0 unbuilt; first built residue 2; indel 0; template 8gp3 (mismatch 0).

### 7oh0 chain C: Zanotti, G., 2021, EM 3.90 Å
* **Antigen:** Tetanus toxin.
* **Run:** IMGT 63–69, sequence `GVTYYNS`.
* **Site:** fit 0.30; 0 unbuilt residues; first built residue IMGT 1 (SEQRES starts at 1); antigen contact 1.
* **Antibody:** IGHV4-39*01 (identity 0.81, human); indel at site 0; gap-placement artifact 0.
* **Template:** none named; mismatch 0. Open-access text: False.
* **Same V region elsewhere in the PDB:** 0 before, 0 after.
* **Same antigen, other labs, ±6 months (in the scan):** 0 entries, of which 0 have errors.
* **Clue:** No open-access text. CDR2 run 63-69 (-1) at the antigen interface, with a clash excess at the site.
* **Control (period/class): 7m8l/M, Ward, A.B., 2021.** Window `GRTYYNP`; fit 0.80; 0 unbuilt; first built residue 1; indel 0; template none named (mismatch 0).
* **Control (near-miss (H)): 7r7n/H, Seattle Structural Genomics Center for Infectious Disease (SSGCID), 2021.** Window `GTANYAQ`; fit 0.42; 4 unbuilt; first built residue 2; indel 0; template none named (mismatch 0).

### 7mkl chain H: Center for Structural Genomics of Infectious Diseases (CSGID), 2021, EM 3.20 Å
* **Antigen:** Spike glycoprotein.
* **Run:** IMGT 54–59, sequence `GVIWAD`.
* **Site:** fit 0.31; 0 unbuilt residues; first built residue IMGT 1 (SEQRES starts at 1); antigen contact 1.
* **Antibody:** IGHV2-9*02 (identity 0.95, mouse); indel at site 0; gap-placement artifact 0.
* **Template:** 1kiq; mismatch 0. Open-access text: True.
* **Same V region elsewhere in the PDB:** 0 before, 1 after.
* **Same antigen, other labs, ±6 months (in the scan):** 177 entries, of which 11 have errors.
* **Clue:** Templates chosen by BLAST: 1KIQ for VH and 5XJM for VL. They were docked into a locally refined Fv/RBD map and refined in Coot, ISOLDE and Phenix. The CDR2 slip (54-59, +1) sits at the antigen interface. The template has the same CDR2 length, so this slip happened without a length mismatch. The entry has an "Atomic model" revision but is still flagged.
* **Control (same antigen): 7mjj/D, Subramaniam, S., 2021.** Window `SVIYSG`; fit 0.54; 0 unbuilt; first built residue 1; indel 0; template 6wgj (mismatch 1).

### 8dg8 chain L: Pancera, M., 2022, EM 3.62 Å
* **Antigen:** Fusion glycoprotein F0.
* **Run:** IMGT 57–70, sequence `KNNRPSV`.
* **Site:** fit 0.48; 2 unbuilt residues; first built residue IMGT 3 (SEQRES starts at 1); antigen contact 0.
* **Antibody:** IGLV3-19*01 (identity 0.90, human); indel at site 0; gap-placement artifact 0.
* **Template:** none named; mismatch 0. Open-access text: True.
* **Same V region elsewhere in the PDB:** 0 before, 0 after.
* **Same antigen, other labs, ±6 months (in the scan):** 0 entries, of which 0 have errors.
* **Clue:** No starting model stated in the text. Lambda CDR2 run 57-70 (+1) next to unbuilt residues; the first built residue is IMGT 3.
* **Control (period/class): 7xmx/F, Li, X., 2022.** Window `NNKRPSG`; fit 0.46; 0 unbuilt; first built residue 1; indel 0; template 7wpf (mismatch 0).

### 7wvl chain H: Nair, D.T., 2022, X-ray 3.00 Å
* **Antigen:** Spike glycoprotein.
* **Run:** IMGT 62–82, sequence `TGDTNFNEKFKSKATLTVDK`.
* **Site:** fit 0.27; 0 unbuilt residues; first built residue IMGT 1 (SEQRES starts at 1); antigen contact 0.
* **Antibody:** IGHV1-47-14*01 (identity 0.90, mouse); indel at site 0; gap-placement artifact 0.
* **Template:** 7b3o; mismatch 1. Open-access text: True.
* **Same V region elsewhere in the PDB:** 0 before, 0 after.
* **Same antigen, other labs, ±6 months (in the scan):** 258 entries, of which 9 have errors.
* **Clue:** X-ray. The model was started from the STEC90-C11 Fab (7B3O) and "the sequence ... was slowly changed to that of P4A2". The template CDR-H2 is 7 residues, P4A2 has 8, and the run 62-82 is shifted +1, so the extra residue was absorbed by sliding FR3. This is a textbook template-threading error.
* **Control (same antigen): 7wue/C, Mohapatra, A., 2022.** Window `IGDTTYNQKFKGKATLTVDT`; fit 0.52; 0 unbuilt; first built residue 1; indel 0; template 7c01 (mismatch 1).
* **Control (near-miss (C)): 7u0c/C, Bjorkman, P.J., 2022.** Window `LGIGNYAEKFQGRVTITADN`; fit 0.44; 0 unbuilt; first built residue 2; indel 0; template 6med (mismatch 1).

## C_FR3

### 9dm7 chain H: Liu, Y., 2024, EM 3.02 Å
* **Antigen:** Mannosyltransferase.
* **Run:** IMGT 72–92, sequence `KGRFTISADTSKNTAYLQMN`.
* **Site:** fit 0.25; 2 unbuilt residues; first built residue IMGT 1 (SEQRES starts at 1); antigen contact 1.
* **Antibody:** IGHV3-23*04 (identity 0.82, human); indel at site 1; gap-placement artifact 0.
* **Template:** 5ucb; mismatch 1. Open-access text: True.
* **Same V region elsewhere in the PDB:** 0 before, 0 after.
* **Same antigen, other labs, ±6 months (in the scan):** 1 entries, of which 0 have errors.
* **Clue:** "Most of the Fab ... was built based on a preexisting model at high resolution (5UCB)". The Fab E6 FR3 is two residues shorter than 5UCB (ANARCI gaps 94-95; FR3 36 vs 38), and FR3 72-92 is shifted +1 (25th-percentile fit). The length mismatch is exactly at the slipped strand.
* **Control (period/class): 9l1l/C, Yonekura, K., 2024.** Window `KNRISITRDTSKNQFFLKLN`; fit 0.51; 0 unbuilt; first built residue 1; indel 0; template none named (mismatch 0).
* **Control (near-miss (N)): 9jrt/N, Xu, E.H., 2024.** Window `QGRFTVSRDDAKTVYLQMN`; fit 0.61; 8 unbuilt; first built residue 1; indel 1; template 8tb7 (mismatch 0).

### 7e23 chain C: Dou, C., 2021, EM 3.30 Å
* **Antigen:** nan.
* **Run:** IMGT 76–93, sequence `FSGSGSGTDFTLTISS`.
* **Site:** fit 0.30; 0 unbuilt residues; first built residue IMGT 2 (SEQRES starts at 1); antigen contact 0.
* **Antibody:** IGKV3-11*01 (identity 1.00, human); indel at site 0; gap-placement artifact 0.
* **Template:** none named; mismatch 0. Open-access text: True.
* **Same V region elsewhere in the PDB:** 15 before, 9 after.
* **Same antigen, other labs, ±6 months (in the scan):** 0 entries, of which 0 have errors.
* **Clue:** No starting model stated. The kappa FR3 FSGSGSGTDF slipped by -2 (76-93), exactly the period of the SG repeat. The same -2 kappa FR3 slip is in 7stf (Miller M.). The germline IGKV3-11 light chain had been solved at high resolution 14 times before.
* **Control (period/class): 7nda/L, Stuart, D., 2021.** Window `FSGSGSGTDFTLTISR`; fit 0.62; 0 unbuilt; first built residue 1; indel 0; template 6yla (mismatch 0).

### 8w5e chain H: Zhu, P., 2023, EM 3.80 Å
* **Antigen:** Capsid protein.
* **Run:** IMGT 71–89, sequence `VTGRFTISRENAKNTLYL`.
* **Site:** fit 0.42; 6 unbuilt residues; first built residue IMGT 2 (SEQRES starts at 1); antigen contact 0.
* **Antibody:** IGHV5-15*01 (identity 0.99, mouse); indel at site 0; gap-placement artifact 0.
* **Template:** none named; mismatch 0. Open-access text: False.
* **Same V region elsewhere in the PDB:** 0 before, 0 after.
* **Same antigen, other labs, ±6 months (in the scan):** 1 entries, of which 0 have errors.
* **Clue:** Mouse IGHV5-15 (mouse IGHV5 is enriched in mode C; see ../deepdive). Six residues are unbuilt near the run (71-89, +1). No open-access text.
* **Control (period/class): 8wnu/D, Luo, D., 2023.** Window `VKGRFTISRDNSKNTLYL`; fit 0.51; 0 unbuilt; first built residue 1; indel 0; template 7php (mismatch 0).
* **Control (near-miss (D)): 8tzy/D, Mariuzza, R.A., 2023.** Window `FQDRVAITADESTSTAYM`; fit 0.44; 5 unbuilt; first built residue 7; indel 0; template 6mei (mismatch 0).

### 7kra chain I: Frost, A., 2020, EM 3.20 Å
* **Antigen:** ER membrane protein complex subunit 1.
* **Run:** IMGT 80–93, sequence `VDTSKNQFSLKLSS`.
* **Site:** fit 0.45; 9 unbuilt residues; first built residue IMGT 1 (SEQRES starts at 1); antigen contact 1.
* **Antibody:** IGHV4-34*01 (identity 1.00, human); indel at site 0; gap-placement artifact 0.
* **Template:** 1m71; mismatch 0. Open-access text: True.
* **Same V region elsewhere in the PDB:** 2 before, 6 after.
* **Same antigen, other labs, ±6 months (in the scan):** 0 entries, of which 0 have errors.
* **Clue:** The yeast EMC was built de novo by hand, with landmark residues used to assign sequence. FR3 run 80-93 (-2) next to 9 unbuilt residues. The same germline V region had two earlier high-resolution structures, which were not used.
* **Control (period/class): 6x87/A, Julien, J.P., 2020.** Window `ADTSSSTAYMQISS`; fit 0.54; 0 unbuilt; first built residue 3; indel 0; template none named (mismatch 0).

### 7mhz chain O: Jiang, Y., 2021, EM 3.20 Å
* **Antigen:** Protein-cysteine N-palmitoyltransferase HHAT.
* **Run:** IMGT 87–95, sequence `AYLQINNLK`.
* **Site:** fit 0.37; 4 unbuilt residues; first built residue IMGT 2 (SEQRES starts at 1); antigen contact 0.
* **Antibody:** IGHV9-1*02 (identity 0.98, mouse); indel at site 0; gap-placement artifact 0.
* **Template:** none named; mismatch 0. Open-access text: False.
* **Same V region elsewhere in the PDB:** 0 before, 2 after.
* **Same antigen, other labs, ±6 months (in the scan):** 0 entries, of which 0 have errors.
* **Clue:** Mouse Fab, 4 unbuilt residues next to FR3 run 87-95 (-2). An "Atomic model" revision was later deposited, but the run is still present.
* **Control (same antigen): 7q6z/B, Siebold, C., 2021.** Window `VNLQMNSLK`; fit 0.38; 0 unbuilt; first built residue 17; indel 0; template 6xux (mismatch 0).

### 9yfu chain N: Han, S., 2025, EM 3.08 Å
* **Antigen:** G protein-coupled receptor 61.
* **Run:** IMGT 81–84, sequence `DDAK`.
* **Site:** fit 0.24; 0 unbuilt residues; first built residue IMGT 1 (SEQRES starts at 1); antigen contact 0.
* **Antibody:** IGHV3-23*01 (identity 0.72, human); indel at site 1; gap-placement artifact 0.
* **Template:** none named; mismatch 0. Open-access text: False.
* **Same V region elsewhere in the PDB:** 3 before, 0 after.
* **Same antigen, other labs, ±6 months (in the scan):** 0 entries, of which 0 have errors.
* **Clue:** One-residue FR3 deletion (ANARCI gap at 85 instead of the usual 81). The 81-84 (+1) run lies between the two possible gap positions. Probably a gap-placement artifact (pattern seen in 2 chains, no high-resolution example).
* **Control (period/class): 9lyc/N, Chang, H., 2025.** Window `DNAK`; fit 0.25; 0 unbuilt; first built residue 1; indel 0; template none named (mismatch 0).
* **Control (near-miss (C)): 9ot6/C, Burke, J.E., 2025.** Window `DNAK`; fit 0.49; 0 unbuilt; first built residue 1; indel 0; template none named (mismatch 0).

### 2x7l chain A: Grimes, J.M., 2010, X-ray 3.17 Å
* **Antigen:** Protein Rev.
* **Run:** IMGT 81–83, sequence `DTS`.
* **Site:** fit 0.14; 0 unbuilt residues; first built residue IMGT 1 (SEQRES starts at 1); antigen contact 0.
* **Antibody:** IGHV3-66*01 (identity 0.59, human); indel at site 1; gap-placement artifact 1.
* **Template:** 3dvg; mismatch 1. Open-access text: False.
* **Same V region elsewhere in the PDB:** 0 before, 5 after.
* **Same antigen, other labs, ±6 months (in the scan):** 0 entries, of which 0 have errors.
* **Clue:** X-ray. FR3 deletion with ANARCI gaps at 80 and 84. SAbR shows the identical disagreement (D81/80, T82/83, S83/84) on the 2.3 A structure 5DHV, so this is a numbering artifact, not a register error.
* **Control (period/class): 3q6f/B, Kong, X.P., 2010.** Window `DNS`; fit 0.43; 0 unbuilt; first built residue 1; indel 0; template none named (mismatch 0).

## F_CDR1

### 3jbf chain 7: Hogle, J.M., 2015, EM 4.80 Å
* **Antigen:** Genome polyprotein.
* **Run:** IMGT 37–48, sequence `YCMGWYRQVRGK`.
* **Site:** fit 0.43; 0 unbuilt residues; first built residue IMGT 1 (SEQRES starts at 1); antigen contact 0.
* **Antibody:** IGHV3-30*02 (identity 0.63, human); indel at site 1; gap-placement artifact 0.
* **Template:** 1i3u; mismatch 1. Open-access text: False.
* **Same V region elsewhere in the PDB:** 0 before, 0 after.
* **Same antigen, other labs, ±6 months (in the scan):** 0 entries, of which 0 have errors.
* **Clue:** VHH modelled from the llama VHH 1I3U. The template CDR1 is 8 residues, PVSP19B has 5, and FR2 is shifted -3 (37-48), the exact length difference. 4.8 A map, 2015.

### 6lht chain L: Li, S.W., 2019, EM 3.67 Å
* **Antigen:** Genome polyprotein.
* **Run:** IMGT 31–57, sequence `YSNNQKNYLAWYQQKPGQSPQLLIYWA`.
* **Site:** fit 0.47; 0 unbuilt residues; first built residue IMGT 1 (SEQRES starts at 1); antigen contact 1.
* **Antibody:** IGKV8-30*01 (identity 0.96, mouse); indel at site 0; gap-placement artifact 0.
* **Template:** none named; mismatch 0. Open-access text: False.
* **Same V region elsewhere in the PDB:** 1 before, 0 after.
* **Same antigen, other labs, ±6 months (in the scan):** 0 entries, of which 0 have errors.
* **Clue:** Localized reconstruction of a CVA16 virion with a mouse Fab. The IGKV8-30 light chain has a 12-residue CDR-L1, and CDR1-FR2-CDR2 is shifted +5 (31-57). This is consistent with a short-L1 starting model whose missing residues were absorbed downstream. No open-access text.
* **Control (period/class): 6jhr/D, Wang, X., 2019.** Window `SYMHWYQQKSGTSPKRWIYDT`; fit 0.44; 0 unbuilt; first built residue 1; indel 1; template 6jhq (mismatch 0).
* **Control (near-miss (L)): 7jic/L, Blacklow, S.C., 2020.** Window `NYMHWYQQKPGTSPRRWIYDT`; fit 0.65; 0 unbuilt; first built residue 1; indel 1; template 6ani (mismatch 0).

### 7t6x chain D: Lander, G.C., 2021, EM 3.83 Å
* **Antigen:** Genome polyprotein.
* **Run:** IMGT 44–47, sequence `QKPG`.
* **Site:** fit 0.47; 0 unbuilt residues; first built residue IMGT 1 (SEQRES starts at 1); antigen contact 0.
* **Antibody:** IGKV3-15*01 (identity 0.88, human); indel at site 0; gap-placement artifact 0.
* **Template:** none named; mismatch 0. Open-access text: False.
* **Same V region elsewhere in the PDB:** 0 before, 0 after.
* **Same antigen, other labs, ±6 months (in the scan):** 0 entries, of which 0 have errors.
* **Clue:** Short FR2 run 44-47 (-1, QKPG) in an HCV E1E2-Fab map. No open-access text.
* **Control (period/class): 7m30/L, McLellan, J.S., 2021.** Window `QKPG`; fit 0.62; 0 unbuilt; first built residue 1; indel 0; template none named (mismatch 0).

### 8gse chain H: Fu, W., 2022, EM 3.70 Å
* **Antigen:** Genome polyprotein.
* **Run:** IMGT 27–35, sequence `FYTFS`.
* **Site:** fit 0.29; 0 unbuilt residues; first built residue IMGT 1 (SEQRES starts at 1); antigen contact 0.
* **Antibody:** IGHV1-47-27*01 (identity 0.81, mouse); indel at site 1; gap-placement artifact 1.
* **Template:** 1nmc; mismatch 0. Open-access text: True.
* **Same V region elsewhere in the PDB:** 0 before, 3 after.
* **Same antigen, other labs, ±6 months (in the scan):** 0 entries, of which 0 have errors.
* **Clue:** Starting models 5HDQ and 1NMC. The mouse 6D10 VH has an FR1 deletion (SCQA-FYTF; ANARCI gap at 26). SAbR shows the same 27-35 disagreement on the 2.49 A structure 5AAM, so this is a numbering artifact. A different lab's anti-echovirus-3 Fab 5G3 (7eaj/7eak) has the same rare FR1 deletion and the same artifact.
* **Control (period/class): 7x2t/H, Xia, N., 2022.** Window `GYTFT`; fit 0.54; 0 unbuilt; first built residue 2; indel 0; template none named (mismatch 0).

### 9ml5 chain M: Bjorkman, P.J., 2024, EM 3.40 Å
* **Antigen:** Spike glycoprotein.
* **Run:** IMGT 31–41, sequence `SSSYWICW`.
* **Site:** fit 0.42; 0 unbuilt residues; first built residue IMGT 1 (SEQRES starts at 1); antigen contact 1.
* **Antibody:** IGHV3-53*04 (identity 0.65, human); indel at site 1; gap-placement artifact 0.
* **Template:** 7sc1; mismatch 1. Open-access text: True.
* **Same V region elsewhere in the PDB:** 0 before, 0 after.
* **Same antigen, other labs, ±6 months (in the scan):** 52 entries, of which 5 have errors.
* **Clue:** Rabbit Fab M8b-B8. The model was started by docking human Fab-RBD structures (7UZD, 7UZ4) into a locally refined map. The rabbit CDR-H1 (SSSYWICW, 9 residues, with an extra Cys) differs in length from the templates (8 or 10), and CDR1 is shifted +1 (31-41). The V-domain local resolution is 6.7 A against 3.4 A global. The same chain also has an FR3 run on an artifact-prone FR3 gap (84).
* **Control (same antigen): 9l6c/D, Bing, Z., 2024.** Window `SSNYMNW`; fit 0.68; 0 unbuilt; first built residue 2; indel 0; template 7cdi (mismatch 0).
* **Control (near-miss (E)): 9kvd/E, Zheng, Q., 2024.** Window `SSFLWNW`; fit 0.63; 2 unbuilt; first built residue 2; indel 0; template none named (mismatch 0).

### 9yc6 chain L: Ploug, M., 2025, EM 4.80 Å
* **Antigen:** Urokinase plasminogen activator surface receptor.
* **Run:** IMGT 31–127, sequence `SNNGKTYLEWYLQKPGQSPKLLIYKVSNRFSGVPDRFSGSGSGTDFTLKISRVEAEDLGVYYCFQGSHVPWTFGGGTKLEVK`.
* **Site:** fit 0.48; 0 unbuilt residues; first built residue IMGT 1 (SEQRES starts at 1); antigen contact 1.
* **Antibody:** IGKV1-117*01 (identity 0.94, mouse); indel at site 0; gap-placement artifact 0.
* **Template:** none named; mismatch 0. Open-access text: False.
* **Same V region elsewhere in the PDB:** 0 before, 1 after.
* **Same antigen, other labs, ±6 months (in the scan):** 0 entries, of which 0 have errors.
* **Clue:** The recorded starting model 3U74 is uPAR alone, so there was no Fab template. The mouse IGKV1-117 light chain has an 11-residue CDR-L1, and the whole chain from CDR1 to the end is shifted +5 (31-127). The CDR-L1 insertion was absorbed downstream. 4.8 A map; ISOLDE was used.
* **Control (period/class): 9uhq/B, Mao, X., 2025.** Window `NNFLNWYQQKPGKAPKLLIYDASNLETGVPSRFSGSGSGTDFTFTISSLQPEDIATYYCHQYDNLPRTFGQGTKVEIK`; fit 0.51; 0 unbuilt; first built residue 1; indel 0; template none named (mismatch 0).

### 7e5y chain B: Yang, X.M., 2021, X-ray 3.59 Å
* **Antigen:** Spike glycoprotein.
* **Run:** IMGT 45–80, sequence `QVPGSTPKLVIYNDRLRPSGVPDRFSGSK`.
* **Site:** fit 0.33; 0 unbuilt residues; first built residue IMGT 2 (SEQRES starts at 1); antigen contact 1.
* **Antibody:** IGLV1-44*01 (identity 0.81, human); indel at site 0; gap-placement artifact 0.
* **Template:** 7c01; mismatch 1. Open-access text: True.
* **Same V region elsewhere in the PDB:** 0 before, 0 after.
* **Same antigen, other labs, ±6 months (in the scan):** 198 entries, of which 23 have errors.
* **Clue:** X-ray molecular replacement with CB6 (7C01, whose kappa light chain has CDR-L1 of 6) for a lambda chain with CDR-L1 of 8. FR2-CDR2-FR3 is shifted +1 (45-80). The upstream length difference was absorbed downstream.
* **Control (same antigen): 7lq7/D, Wilson, I.A., 2021.** Window `HPGKAPKLMIYEVNKRPSGVPDRFSGSK`; fit 0.49; 0 unbuilt; first built residue 2; indel 0; template 7jmw (mismatch 1).
* **Control (near-miss (E)): 7pi3/E, Higgins, M.K., 2021.** Window `KSPGSAPVTVIYENNKRPSDIPSRFSGSK`; fit 0.39; 0 unbuilt; first built residue 3; indel 1; template 6rcs (mismatch 1).
