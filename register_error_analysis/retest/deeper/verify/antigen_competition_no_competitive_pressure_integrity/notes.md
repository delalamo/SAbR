# Verification: antigen_competition / no_competitive_pressure (integrity)

The verifier returned this verdict without writing a notes file. The text below is its structured return, copied unchanged from the workflow log.

**Verdict: weakened**

## Reason

CORE NULL IS REAL AND REPRODUCES EXACTLY, BUT "STRONG" OVERSTATES IT AND SOME SUB-CLAIMS NEED SOFTENING.

What reproduces:
(1) Analysis set. I rebuilt it from entries.csv: amyloid entries removed, cases are err_first, controls are entries with no run of any kind, and only bound or fiducial-only entries are kept. It matches the investigator's focal table exactly (3,994 entries; cryo-EM 2,707 with 174 errors, 124 error papers, 100 error labs; control n = 2,533). There are no duplicate-key joins and no mismatches in y or in labs_acc_12m.
(2) Headline numbers. All reproduce exactly:
- other labs, same accession ±12 months: OR 0.943 (0.859–1.035), p = 0.219;
- same, ±6 months: OR 0.947;
- race (±1 month): OR 0.90 (0.55–1.46), p = 0.67;
- same family ±12 months: OR 0.895, lens-wide BH q = 0.084; without SARS-CoV-2, OR 0.817, q = 0.0143;
- X-ray: OR 1.03, p = 0.69;
- medians 0.5 vs 1 and means 32.0 vs 33.5;
- SARS-CoV-2 cryo-EM median 143 other labs;
- spike family 6.7% vs 6.4%, OR 0.91, p = 0.74.
(3) Independent rebuild of the exposure. I did not use the GraphQL or interface data. Antibody entries came from my own V-motif regex scan of pdb_seqres (11,289 of 12,068 entries agree with the ANARCI list). The antigen is all non-Ig, non-fiducial SIFTS accessions in the entry. Lab and date come from entries.idx. This exposure has Spearman 0.93 with the investigator's. It gives OR 0.938 per doubling (0.854–1.030, p = 0.18) in cryo-EM and 0.83 without SARS-CoV-2.
(4) Robustness. The OR stays between 0.92 and 0.97 under every variant: full-year, quarter or spline date adjustment; clustering by lab; bound entries only; outcome including copied or artifact runs (0.93); paper level (0.93, p = 0.064). Leave-one-error-paper-out gives OR 0.918–0.954 and leave-one-lab-out 0.919–0.958. No positive association appears in any cryo-EM specification. The upper 95% bound of 1.035 per doubling means at most OR 1.28 across 0 to 127 competing labs.

Why "weakened" rather than "confirmed":
(a) The log-linear per-doubling test is misspecified. The dose-response is U-shaped: 8.3% errors with 0 other labs, 4.8%, 3.8% and 2.8% across 1–49 labs, then 6.8% at ≥50 labs (96% of the ≥50 group is SARS-CoV-2). The quadratic term has p = 0.005, or p = 0.08 once target class is added. A within-stratum permutation (year × SARS-CoV-2) is significant in the protective direction (p = 0.014). So the p = 0.22 null averages over opposite slopes.
(b) "SARS-CoV-2 has an ordinary error rate" depends on the reference group:
- versus all other cryo-EM entries, the year-adjusted OR is 0.82 (0.47–1.43); in 2020–21 the rates are 9.6% vs 10.1%;
- versus other *contested* antigens (≥1 other lab) in 2020–21, SARS-CoV-2 is 9.7% vs 2.2%: OR 5.3 (1.6–17.9), p = 0.007; bound entries only, OR 3.4, p = 0.04;
- that comparison group holds only 3–4 errors and is mostly GPCRs (0/61) and channels, so it is fragile and confounded by class and lab experience. Against viral glycoproteins only, OR 5.9 (0.65–54), p = 0.11. In 2022+ there is no difference (OR 0.81).
(c) "If anything fewer" is target-class mix, not competition:
- adjusted for class and fiducial-only status, OR 0.98 per doubling (p = 0.70);
- any other lab: OR 0.76 (p = 0.22);
- the family-level result without SARS-CoV-2 drops to OR 0.91 (p = 0.33) once capsid and enzyme targets are excluded.
(d) The race and X-ray nulls are underpowered:
- race: CI 0.55–1.46; without SARS-CoV-2, 0.51–1.79 with 25 exposed errors;
- X-ray: 31 errors; my exposure gives OR 1.11 (0.96–1.30), p = 0.17;
- X-ray SARS-CoV-2 entries have 5.4% errors vs 2.5% (Fisher p = 0.12).
(e) "The 25 earliest were all clean" is literally true; in fact the first 32 entries, through 2020-08-29, were clean. But the 25 come from only 12 papers, and P(all clean) at the period paper rate is 0.20, so it says nothing either way. The full April–August window has 35 entries including 3 errors (one Wang X. paper, deposited 2020-08-31). In September–December 2020, SARS-CoV-2 reached 18.8% (9/48), level with non-SARS-CoV-2 at 17.6%.
(f) Minor bug: 69 cryo-EM entries have fiducial accessions (BRIL, GFP and similar) fused into their target keys, which inflates their competitor counts. Removing them leaves the result unchanged (OR 0.942, p = 0.22).

Suggested rewording: "No monotone increase in errors with the number of competing labs (cryo-EM OR 0.94 per doubling, upper bound 1.035)." Rate it moderate-to-strong for the cryo-EM trend, and weak for race and X-ray. Describe "fewer" as the capsid/enzyme niche effect. Say SARS-CoV-2 matched its period's average rate, but in 2020–21 was above other contested antigens (fragile, n.s. against viral glycoproteins).

Files (all in /home/user/SAbR/register_error_analysis/retest/deeper/verify/antigen_competition_no_competitive_pressure_integrity/):
- v01_reproduce.py and .log, v01_reproduce_robustness.csv, v01_leave_one_out.csv
- v02_independent_exposure.py and .log, v02_independent_exposure.csv, v02_independent_exposure_tests.csv
- v03_sars2_claims.py and .log
- v04_shape_and_bh.py and .log, v04_shape_tests.csv
- v05_family_fewer.py, v06_fiducial_keys.py, v05_misc_checks.log

## Recomputed values

EM cryo-EM, error 174 / control 2533 entries, 124 error papers:
- same accession ±12m: OR 0.943 (0.859-1.035), p=0.219 (exact); ±6m: 0.947, p=0.29; race ±1m: 0.90 (0.55-1.46), p=0.67;
- same family ±12m: 0.895, q=0.084; non-SARS2: 0.817, q=0.0143;
- X-ray: 1.035, p=0.69.
Independent SIFTS/entries.idx/regex exposure: 0.938 (0.854-1.030), p=0.18; non-SARS2 0.83 (0.64-1.08); race 0.95 (0.56-1.61); X-ray 1.11 (0.96-1.30), p=0.17.
Variants:
- quarter categories 0.919 (p=0.059); spline date + resolution 0.935; unadjusted 0.965;
- bound only 0.931; +SARS2 indicator 0.855 (p=0.07); outcome err_any 0.929 (p=0.08);
- paper level 0.927 (p=0.064); class+fiducial adjusted 0.982 (p=0.70).
Binary any other lab ±12m: 0.58 (0.36-0.91), p=0.017; with class+fiducial 0.76, p=0.22.
Leave-one-out: drop one error paper, OR 0.918-0.954; drop one error lab, 0.919-0.958.
Permutation (year × SARS2 strata, papers): p=0.014, negative direction.
Quadratic term: p=0.005 (year+res); p=0.08 with class.
Error rate by other labs: 0: 8.3% (87/1047); 1-2: 4.8%; 3-9: 3.8%; 10-49: 2.8%; ≥50: 6.8% (44/644).
SARS2 vs other cryo-EM, year+res: OR 0.82 (0.47-1.43); 2020-21: 9.6% vs 10.1%.
SARS2 vs contested non-SARS2:
- 2020-21: OR 5.3 (1.6-17.9), p=0.007, 9.7% vs 2.2%; bound only 3.4, p=0.04; vs viral glycoproteins only 5.9 (0.65-54), p=0.11;
- 2022+: OR 0.81 (0.37-1.79).
Earliest SARS2 cryo-EM entries: first 32 clean (to 2020-08-29); 35 entries to 2020-08-31 include 3 errors (one paper). First 25 = 12 papers; P(all clean) at period paper rate = 0.20. Sep-Dec 2020: SARS2 18.8% (9/48) vs non-SARS2 17.6% (9/51).
Fiducial-key bug: 69 entries affected; excluding them, OR 0.942, p=0.22.
