# Verification: people_network / new_team_ab_cryoem (integrity)

The verifier returned this verdict without writing a notes file. The text below is its structured return, copied unchanged from the workflow log.

**Verdict: weakened**

## Reason

The pooled number reproduces exactly and holds up across definitions. But it comes from one 2-year window, not a general property, and the headline shares are inflated by how names were handled.

What reproduces:
- Joins are clean. The outcome matches the investigator's study frame 100% (1,303 studies). My new-team flag, rebuilt independently, matches theirs for 99.7% of studies.
- My rebuild uses entries.idx names for both the study and its history, an author parser that handles 'Jr.', and a second antibody set built without ANARCI. It gives OR 2.03 (p=0.0003); refitting their exact flag gives OR 1.99, 47.2% vs 33.2%, error rate 13.3% vs 7.9%.
- The pooled estimate barely moves:
  - name rules: all names counted 2.06, surname plus first initial 1.95, surname only 1.97 (p=0.045), last author only 1.84;
  - a co-author-confirmed identity rule for common surnames gives 2.01;
  - the non-ANARCI antibody set gives 1.87;
  - copied/artifact runs counted as errors gives 1.66 (p=0.03);
  - entry level 2.18, published studies only 2.20;
  - adjusting for calendar year or half-year gives 2.13 and 2.27;
  - studies whose antibodies were all new sequences give 2.79;
  - leave-one-lab-out keeps p at or below 0.01.
- The lens-wide q of 0.010 reproduces within their 78-test family.

What weakens it:
1. **The effect is confined to 2022-23.** The new-team effect differs by year (interaction test p=0.0025 with all names, 0.006 with their rule, 0.02 with their flag).
   - Deposited 2022-23: OR 4.7-8.4 under every definition, p<1e-4. Counts with all names: 15/40 new-team studies had errors vs 23/352 others. This rests on 38 error studies from 15 different labs.
   - Deposited 2021 or earlier: OR 0.96 with all names, 1.15-1.26 with the co-author rule; both null.
   - Deposited 2024 onward: OR 0.65-1.15, null.
   - Dropping 2022-23 leaves no significant effect under any definition (their flag 1.41, p=0.15; all names 1.02, p=0.94; co-author rule 1.30, p=0.30).
   - Dropping 2023 alone kills the all-names version (OR 1.34, p=0.30).
   - So it does not explain the 2020-21 error peak. The investigator noted only that it vanishes from 2024.
2. **Common surnames are new teams by construction.** The primary flag ignores collision-prone surnames (Wang, Li, Kim, Smith and similar), so an all-common-surname team counts as new automatically: 98.9% of 187 such studies are flagged.
   - 66% of studies whose last author has a common surname are 'new', vs 19% otherwise.
   - Counting all names, new teams are 22.0% of error vs 12.4% of clean studies (rates 16.1% vs 8.8%). With the co-author rule: 36.2% vs 23.3%. So the 47.2% vs 33.2% shares are definition-dependent.
   - The investigator-flag effect before 2022 comes from this rule. In 2020-21, restricting to last authors with uncommon surnames gives OR 1.81 (p=0.23); adjusting for China/USA gives 2.16 (p=0.049).
3. **The effect is small in absolute terms.**
   - About 8-13% of error studies are explained: 28 new-team error studies observed vs 17.4 expected with all names, 59 vs 43 with their rule.
   - The penalty is zero vs any experience. Among experienced teams, more experience does not lower the error rate further (OR 0.96 per doubling, p=0.52).
4. **Statistical fragility.**
   - Permuting the flag within year and resolution strata gives p=0.026 for the all-names flag, against a parametric p of 0.0026. The exact year-stratified test gives 0.006.
   - Within the same last author's lab, the comparison is too small to tell learning from lab type: conditional logit OR 1.7-2.5, p=0.09-0.38.
   - Pooling all 4,135 of the lens's tests, including the entity scans, puts q near 0.06.
   - The X-ray support is weak: OR 2.2 (p=0.16) with all names, 1.17 under their primary rule.

Bottom line: a real but era-specific association (2022-23). 'Strong' and 'about twice as likely' overstate it as a general link.

## Recomputed values

Reproduced: OR 1.99 (1.36-2.91), p=0.00037; 47.2% vs 33.2%; rate 13.3% vs 7.9%; n=127 error / 1,176 clean studies. Independent rebuild (their rule): 2.03 (p=0.0003).
All names: OR 2.06 (1.29-3.29), p=0.0026; 22.0% vs 12.4%.
Co-author-confirmed: 2.01, p=0.0006; 36.2% vs 23.3%.
Year interaction (LR) p: 0.0025 / 0.006 / 0.02.
By era (all names / co-author / their flag):
- 2021 or earlier: 0.96 / 1.15 / 2.23;
- 2022-23: 8.4 / 5.1 / 4.7;
- 2024 onward: 1.15 / 1.15 / 0.79.
Excluding 2022-23: 1.02 (p=0.94) / 1.30 (p=0.30) / 1.41 (p=0.15).
Permutation within year x resolution, all names: p=0.026. Exact year-stratified: p=0.006.
Within-lab conditional logit: OR 2.5, p=0.09.
X-ray: 2.2 (p=0.16) with all names; 1.17 under their rule.
Files are in /home/user/SAbR/register_error_analysis/retest/deeper/verify/people_network_new_team_ab_cryoem_integrity/:
- scripts: v01_rebuild.py, v02_tests.py, v03_followups.py, v04_era_xray.py;
- tables: verify_tests.csv, verify_followups.csv, verify_era_xray.csv, per_year_all_name_flag.csv, leave_one_out.csv.
