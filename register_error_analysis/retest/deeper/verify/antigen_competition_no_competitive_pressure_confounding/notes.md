# Verification: antigen_competition / no_competitive_pressure (confounding)

The verifier returned this verdict without writing a notes file. The text below is its structured return, copied unchanged from the workflow log.

**Verdict: weakened**

## Reason

The numbers reproduce exactly from the investigator's tables (EM, deposit year + resolution, clustered by paper). Other labs on the same accession within ±12 months give OR 0.943 (0.86–1.04, p=0.22, n=2707, 174 errors). The same family gives 0.895 (p=0.021), and the ±1-month race gives 0.90 (p=0.67). The core null holds: under no adjustment, stratum, unit or outcome definition does crowding significantly raise errors after BH (214 tests). But three parts of the claim do not hold up, and "strong" overstates it.

(1) The "if anything fewer" direction is confounding, not competition. Niche antigens are where the known high-risk targets and antibodies sit: capsid and enzyme targets, nanobodies, mouse antibodies and low germline identity. Adjusting for capsid/enzyme alone moves the ±12-month accession OR from 0.94 to 1.03. Antibody features alone (nanobody, mouse, min germline identity, atypical CDR, lambda) move it to 1.00. Adjusting for pre-treatment confounders (target class, fiducial-only, antibody features) gives 1.07 (0.98–1.18, p=0.14), and adding modelling covariates with half-year fixed effects gives 1.07 (p=0.17). The family measure goes from 0.90 to 1.02–1.03. The investigator's headline protective result, same family in non-SARS-CoV-2 EM (OR 0.82, q=0.014), becomes 0.93 (p=0.38) after full adjustment. The same pattern holds at paper level (0.93 → 1.03), at unit level (0.92 → 1.03), with the heavy chain alone (0.915, p=0.044 → 1.02), for the first appearance of each unique antibody (0.915 → 1.03) and with any run as the outcome (0.93 → 1.07). Clustering by lab instead of paper changes nothing. The within-lab conditional logit gives 1.03 (p=0.61). The only protective effect that survives is the investigator's separate 2020–21 non-SARS-CoV-2 "niche" effect (adjusted OR 0.31 per doubling, n=347). That is a niche or period effect, not evidence that crowding protects.

(2) Once confounders are adjusted, the short-window race measures point toward MORE errors. Log2 of other labs within ±1 month: OR 1.16 (1.01–1.34, p=0.03). The race indicator: 1.50 (0.98–2.30, p=0.06). The ±6-month count: 1.09 (p=0.09). In the fully adjusted dose-response, the 50+ labs category (SARS-CoV-2) has OR 1.72 (0.91–3.25, p=0.10). Leave-one-lab-out p values range from 0.024 to 0.14. None of these survive BH (q ≥ 0.27). The signal is carried by SARS-CoV-2: adding a SARS-CoV-2 indicator gives OR 1.17 (p=0.30), and within non-SARS-CoV-2 entries it is 1.25 (p=0.19). The adjusted upper bound of 1.18 per doubling is compatible with up to about 3× more errors at SARS-CoV-2 levels of competition (about 7 doublings). So the data do not show "no effect" strongly; they fail to show an effect.

(3) "SARS-CoV-2 has an ordinary error rate" is true crudely but hides an excess over what its antibodies predict. SARS-CoV-2 antibodies are easy cases: fewer nanobodies (13% vs 32%), fewer mouse antibodies (11% vs 24%), higher germline identity (0.89 vs 0.83), and no capsid/enzyme or fiducial-only entries. A model fitted on non-SARS-CoV-2 EM entries (half-year, resolution, antibody features, modelling, class) predicts 23.5 errors; 42 were observed (O/E 1.78, unclustered z-test p=1e-4, q=0.019). Paper-clustered estimates are weaker: pre-treatment-adjusted OR 1.72 (0.99–2.97, p=0.054); fully adjusted 1.60 (p=0.10), and 2.24 within 2020–21 (p=0.11). Against "ordinary" antigens only (non-capsid/enzyme, non-fiducial), SARS-CoV-2 is 6.4% vs 4.8% overall and 9.7% vs 5.6% in 2020–21. The adjusted OR there is 1.0–1.3 (n.s.): 1.30 (p=0.37) with antibody features, 0.96–1.34 at paper level. The excess is fully explained by the poorer local map fit of SARS-CoV-2 complexes. Among clean entries, local fit (minimum lq_z per entry) has a median of −0.82 vs +0.35 (p=5e-53), and adjusting for it gives OR 0.94. A simpler known factor (poor local density around Fabs bound to the flexible RBD) therefore accounts for it, not competition as such.

(4) The "first 25 SARS-CoV-2 EM entries all clean" point is uninformative and selectively framed. Under the model, 0.57 errors were expected (P(0 errors)=0.56). Entries 26–50 (late August to September 2020) contain 10 errors, so the earliest 50 run at 20%. Seven of those 10 come from one paper (Zhou, Q., Cell Res 2021). In 2020Q3, SARS-CoV-2 (25.5%) and non-SARS-CoV-2 (25.7%) EM entries spiked together, which points to period rather than race. SARS-CoV-2 error papers peaked in 2021H1 (18%).

(5) Two candidate artifacts did not matter:
- Excluding copied-only entries: their share is not linked to competition (OR 0.88, p=0.13).
- The Omicron race window (Dec 2021–Jun 2022): Omicron-titled entries are not elevated (6.5% vs 7.6% for non-SARS-CoV-2 EM, Fisher p=1.0).

Net: the absence of a positive competition effect survives, but the claim's protective direction and its strength do not. Read it as "no detectable effect, with the CI including moderate harm at SARS-CoV-2-level crowding", not as a strong null.

Files are in /home/user/SAbR/register_error_analysis/retest/deeper/verify/antigen_competition_no_competitive_pressure_confounding/:
- scripts: v01_build.py, v02_confounding.py, v03_race_sars2.py, v04_sensitivity.py
- logs: v02.log, v03.log, v04.log
- tables: v02_tests.csv, v03_tests.csv, all_tests_bh.csv (214 tests, BH q), v03_sars2_timecourse.csv, entries_frame.csv.gz, units_frame.csv.gz

Caveats:
- Antibody covariates are entry-level aggregates of chain features.
- Local map fit could be partly downstream of errors; to limit that, the SARS-CoV-2 map comparison used clean entries only.
- The O/E test ignores paper clustering.
- The race positives depend on adjustment choices and do not survive BH.

## Recomputed values

Reproduced (EM, year+res, paper-clustered): labs same accession ±12 m OR 0.943 (0.86–1.04) p=0.219, n=2707/174 err; same family ±12 m 0.895 p=0.021; race ±1 m 0.90 p=0.67. After adjustment:
- Pre-treatment adjustment (capsid/enzyme, fiducial-only, nanobody, mouse, germline identity, atypical CDR, lambda): accession ±12 m 1.07 (0.98–1.18) p=0.14; family 1.03 p=0.62; race 1.50 (0.98–2.30) p=0.062; log2 labs ±1 m 1.16 (1.01–1.34) p=0.032 (q=0.27); SARS-CoV-2 1.72 (0.99–2.97) p=0.054.
- Full adjustment (+ half-year, modelling): accession 1.07 p=0.17; family 1.02 p=0.64.
- Non-SARS-CoV-2 family 0.82 → 0.93 (p=0.38).
- Robustness: paper level 1.03 (p=0.49); unit level 1.03 (p=0.49); first appearance 1.03 (p=0.57); any-run outcome 1.07 (p=0.15); within-lab conditional logit 1.03 (p=0.61); resolution strata 1.06–1.09 (n.s.); X-ray 1.12 (p=0.21).
- SARS-CoV-2: O/E 42/23.5 = 1.78 (unclustered p=1e-4). Against ordinary antigens: OR 1.30 (p=0.37), and 0.94 after local map fit. Clean-entry local fit median −0.82 vs +0.35.
- Earliest 25 SARS-CoV-2 entries: expected 0.57 errors (P0=0.56); entries 26–50 contain 10 errors (7 from one paper).
