# Are antibody V domains more error-prone per residue?

**Question.** Antibody V domains are short (about 112 residues). The 2024 study's "1 in 6 entries
flagged" is per *entry*, and the median entry has 1,266 residues. Could antibodies be worse than
other proteins once length is controlled?

**Data.** The 2024 conkit-validate flags (Table S1), mapped onto ECOD domains with their residue
lengths. The denominator is entries released by 5 April 2022. 73 domains with invalid ranges were
dropped (wrap-around author numbering; none flagged). Comparator classes:
* **Non-Ig:** every domain outside the Ig/E-set T-group and the Ig-like β-sandwich X-group.
* **Other Ig-like:** TCR, MHC, receptor domains and other Ig/E-set domains.

## Answer

| | Antibody V vs non-Ig, per residue (HR, 95% CI) |
|---|---|
| All years, all methods | **0.58** (0.44–0.76): antibodies lower |
| All years, domains of 100–130 residues only | **0.43** (0.32–0.57) |
| Deposited ≤2017 | 0.10 (0.04–0.24) |
| Deposited 2018–19 | 0.35 (0.17–0.71) |
| Deposited 2020–22 | **1.41** (1.02–1.95): antibodies higher |
| Cryo-EM, 2020–22 | **1.59** (1.12–2.25) |
| 2020–22, proteins new to the PDB since the AlphaFold2 training cutoff | 1.19 (0.80–1.77), n.s. |
| Cryo-EM 2020–22, proteins new to the PDB | 1.21 (0.78–1.87), n.s. |

HR = per-residue hazard ratio from a complementary log-log model with offset log(length),
P(flag) = 1 − exp(−h·L). It is adjusted for method × resolution bin and era, with standard errors
clustered by entry. A Poisson model of error counts gives the same answers (`tables/hazard_ratios_vs_abV.csv`).

1. **Units explain most of the apparent gap between the headline numbers.** With the same detector,
   the rates are:
   * 16.2% of entries;
   * 5.4% of chains (antibody chains 2.0%);
   * 3.75% of non-Ig domains;
   * 1.55% of antibody V domains.

   Per 10,000 residues, there are 2.9 errors in non-Ig domains (2.7–3.1) and 1.7 in antibody V
   domains (1.2–2.1).
2. **Length does not rescue the pooled comparison.** Flag risk in non-Ig domains scales as
   length^0.75. Among domains of the same length (100–130 residues), 3.9% of non-Ig domains are
   flagged against 1.55% of antibody V domains.
3. **The pooled comparison is dominated by structures AlphaFold2 had seen.** conkit-validate
   cannot see errors in those (`../retest`, H3). For 2020–22 depositions, antibody V domains have
   1.4× the per-residue hazard of non-Ig domains, and 1.6× in cryo-EM.
   * Within the same cryo-EM maps (2020–22 entries containing antibodies), antibody V domains carry
     3.6 errors per 10,000 residues against 1.4 for the non-antibody domains.
   * The Mantel–Haenszel rate ratio stratified by entry is 1.6.
4. **About half of that 2020–22 excess is the detector's blind spot.**
   * **Which proteins AlphaFold2 had seen.** 72% of 2020–22 cryo-EM non-Ig domains belong to
     proteins already in the PDB before the training cutoff. The test is UniProt accession or exact
     sequence, in an entry deposited before 30 April 2018. Only 21% of antibody V domains do, by
     exact V-region sequence.
   * **Effect on the flags.** Unseen non-Ig domains are flagged more than seen ones (2.5 vs 2.0
     per 10,000 residues). Seen antibodies are flagged *more* than unseen ones (5.8 vs 3.0), as
     expected if they were copied from erroneous templates.
   * **Restricted to proteins new to the PDB,** antibody V domains are 1.2× non-Ig domains per
     residue, which is not significant.
5. **Other Ig-like domains look like antibodies.** Length-matched, their HR is 0.85 vs antibody V
   (n.s.).

## What this cannot settle

conkit-validate under-detects antibody errors even after the cutoff. On the same V domains, SAbR
vs ANARCI flags 8.2% of cryo-EM 2020–22 domains, against 3.6% for conkit; 97 of conkit's 107
antibody flags are also SAbR flags. Two limiting cases follow:
* **If conkit misses non-Ig errors at the same rate,** the ratios above stand: antibodies are about
  1.2–1.6× non-Ig per residue, not significant for new proteins.
* **If conkit catches nearly all non-Ig errors,** the true antibody excess could be up to about
  2.3 × 1.2 ≈ 2.7× per residue.

Settling this needs an AlphaFold-independent register check run on all proteins, for example a
map-based check such as checkMySequence. The SAbR/ANARCI detector only works on antibodies.

## Files

* `scripts/length_norm.py`: units, length dependence, length matching and standardisation, hazard
  and Poisson models, within-entry comparison, whole chains. Log: `tables/length_norm.log`.
* `scripts/length_norm_af2.py`: the training-set control. Log: `tables/length_norm_af2.log`.
* `scripts/length_fig.py`: `figures/length_normalised.png`.
* `tables/`: all result tables as CSV.
