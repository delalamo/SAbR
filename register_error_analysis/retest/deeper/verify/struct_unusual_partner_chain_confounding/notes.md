# Verification: struct_unusual / partner_chain (confounding)

The verifier returned this verdict without writing a notes file. The text below is its structured return, copied unchanged from the workflow log.

**Verdict: weakened**

## Reason

The clash result reproduces exactly and holds up against the standard confounders. The rest of the finding does not hold as stated.

**1. Misreported value.** The Ramachandran 'allowed' error value of 10.5 per 100 is wrong for this comparison. It is the mean over all 128 case partners, including 43 that carry their own disagreement runs (those 43 average 16.7). Among the 85 error-free partners actually tested the mean is 7.36, against 5.92 in controls.

**2. The Ramachandran 'allowed' trend is not robust.**
- Base model: p=0.065.
- Binomial GLM: OR 1.19, p=0.10.
- Median regression: p=0.18.
- log(1+y): p=0.14.
- Winsorised: p=0.12–0.19.
- Non-SARS-CoV-2 only: p=0.11.
- 2020–22 only: p=0.43.
- Dropping the 5 highest case partners: +0.33, p=0.58.

**3. The clash excess survives confounding adjustment.** Base: +5.23 (95% CI 1.26–9.19), p=0.0098, n=85/309. Holding up under:
- cluster by paper (p=0.012) or by last author (p=0.021);
- continuous deposition date (p=0.014); SARS-CoV-2 (p=0.009); non-SARS-CoV-2 only (+6.6, p=0.006);
- partner Q (+4.6, p=0.027); partner Q plus entry Q (p=0.017);
- clash rate of the non-V rest of the same entry (+3.9, p=0.004);
- software flags, fitting protocol and starting-model type (+4.4, p=0.028);
- partner unbuilt residues and side-chain truncation (p=0.010);
- excluding partners with any 1–2-residue disagreement, i.e. the error definition (+5.7, p=0.008);
- paper-level means (p=0.007); matched-set Wilcoxon (p=0.010); binomial GLM (OR 1.36, p=0.015);
- symmetric trimming or winsorising of both groups (+3.8 to +4.7, p=0.007–0.026).

**4. The clash excess is fragile in subsets.** Interaction terms are not significant (period p=0.48, resolution p=0.29), so this is lost power rather than clear heterogeneity.
- 2020–22 only: +4.2, p=0.20 (n=41/131).
- VH partners of VL error chains: +2.9, p=0.32 (n=30/152).
- Resolution below 3.5 Å: +2.4, p=0.30 (n=40/155).

**5. Not explained by contact with the misbuilt chain.** Clashes are classified by the counterpart atom in the validation XML:
- intra-partner clashes: +4.7 per 100;
- clashes with the error chain: +0.68;
- clashes with the error site ±5 residues: +0.23 (p=0.10);
- clashes with other chains: +0.21.

**6. Not specific to the partner, and not independent evidence.**
- Within the same case entries, the partner is no worse than the other error-free V chains of the same class: clash −0.74 (n=42, p=0.83), core RMSD +0.04 Å (p=0.89).
- The within-entry difference-in-differences against control entries is not significant.
- Case partners in entries that do have other clean same-class chains show only +2.4 (p=0.29) against controls; those without show +7.8 (p=0.009).
- Adjusting for the error chain's own clash rate outside the error site removes the partner effect entirely: −0.01 (−1.89 to 1.88), p=0.99. Spearman rho is 0.91 in cases and 0.77 in controls.
- All error-free V chains in error entries are worse than chains in control entries: core RMSD +0.19 Å (p<1e-4), clash +3.7 (p=0.06).

So the partner result restates the entry-level, antibody-wide signal already reported as the beyond-site finding; it does not confirm it independently.

**7. Core RMSD detail.** The "intermediate" core RMSD reproduces. Partner medians are 0.68 Å (VL) and 0.76 Å (VH); controls 0.56–0.57 Å; error chains 0.91–0.95 Å. Adjusted effect +0.20 Å (0.11–0.30), p=4e-5; still significant with partner Q added and in 2020–22 only. It is equally entry-wide, as the within-entry test shows.

**8. Partner map fit.** Partner Q-score is lower in cases (0.332 vs 0.371), but the gap disappears once entry Q is adjusted for (−0.006, p=0.64). It reflects map quality of the whole entry.

**Verdict.** The clash association survives confounding, but strength should drop to weak–moderate and it should be described as entry-level, not specific to the partner. The Ramachandran value and trend should be corrected or dropped.

**Files written** (all under /home/user/SAbR/register_error_analysis/retest/deeper/verify/struct_unusual_partner_chain_confounding/):
- Scripts: v01_partner_clash_decomp.py, v02_stats.py, v03_robustness.py, v04_within_entry_core.py, v05_software.py.
- Tables (tables/): partner_decomp.csv, confounding_tests.csv (BH q within the table), strata.csv, robustness.csv, tails.csv, within_entry.csv, within_entry_summary.csv, core_rmsd.csv, core_all_chains.csv, core_tests.csv, software_adjust.csv.

**Caveats.**
- The within-entry tests cover only the 42 case partners whose entries contain other clean same-class chains.
- Software metadata are sparse.
- Controls come from entries with no disagreement run anywhere, which may favour carefully built entries. The within-entry contrast is not exposed to this.

## Recomputed values

Reproduced exactly: partner clash residues per 100 were 23.16 vs 17.51, adjusted +5.23 (1.26–9.19), p=0.0098, n=85/309. Ramachandran 'allowed' was 7.36 vs 5.92, not 10.5, with +1.20 (−0.08 to 2.49), p=0.065.

Confounding-adjusted clash effects:
- cluster by paper: +5.23, p=0.012;
- full model (date, SARS-CoV-2, partner and entry Q, unbuilt residues, side-chain truncation, rest-of-entry clash): +3.87 (1.39–6.35), p=0.002;
- software, fitting protocol and starting model: +4.41, p=0.028;
- 2020–22 only: +4.21, p=0.20;
- VH partners: +2.90, p=0.32;
- resolution below 3.5 Å: +2.38, p=0.30.

Specificity checks:
- within entry, partner minus other clean same-class chains: clash −0.74 (n=42, p=0.83), core RMSD +0.04 Å (p=0.89);
- adjusted for the error chain's own clash rate outside the site: −0.01, p=0.99.

Partner core RMSD: +0.20 Å (0.11–0.30), p=4e-5.
