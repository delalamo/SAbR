# Verification: paper_meta / PM1_less_competition (confounding)

The verifier returned this verdict without writing a notes file. The text below is its structured return, copied unchanged from the workflow log.

**Verdict: weakened**

## Reason

The headline reproduces exactly from competition_table.csv: 58% (70/121) vs 73% (919/1256), adjusted OR 0.469 (0.316-0.696), p=1.7e-4. It is robust to many controls: lab-clustered SEs, finer time adjustment, complex size, lab experience, new-V share and SARS-CoV-2/viral status. Two things it is not robust to:

(1) Target-class confounding.
- Papers with no competitor are much more often capsid targets (10.3% vs 2.2%) and enzyme targets (7.5% vs 2.5%), and almost never fiducial-only (0.3% vs 11.1%).
- Capsid and enzyme targets carry high error rates; fiducial-only targets carry low ones.
- Adjusting for these classes moves the OR from 0.47 to about 0.60, with p near 0.05 (parsimonious model 0.60 (0.35-1.01), p=0.057, q=0.10).
- Every other adjusted model is borderline at best: the full model gives 0.62 (0.35-1.09), p=0.096, and the Mantel-Haenszel class-by-period estimate 0.66 (0.42-1.03), p=0.064.
- Within enzyme targets there is no association at all (28% vs 28%).

(2) A measurement artifact: UniProt accessions are strain-specific for viral proteins.
- Re-defining the antigen with RCSB 30% sequence clusters (same PDB-wide ±180-day rule) shrinks the gap to 82% vs 87%: base OR 0.58 (p=0.035), class-adjusted 0.75 (p=0.33).
- The headline viral non-SARS-CoV-2 subgroup (37% vs 66%, OR 0.30) becomes 87.5% vs 91.4% (OR 0.69, p=0.42).

Further points:
- X-ray shows no association (OR 1.26, p=0.59, 21 error papers).
- Among papers from experienced labs (10 or more prior antibody entries) there is none either (adjusted OR 1.17).
- A competitor deposited in the 180 days after the paper predicts errors as well as one deposited in the 180 days before. This marks a stable "niche antigen" property, not timing.
- Local map fit and unbuilt residues do not explain it: papers with no competitor actually have better local fit, and adjusting for local fit strengthens the OR to 0.41.
- What survives: the direction is consistent everywhere, and the log-count versions stay weakly negative (class-adjusted OR 0.86-0.89 per doubling, p 0.04-0.07). At unit level the class-adjusted OR is 0.49 (0.30-0.81), p=0.005, lab-clustered.

Net: about a third of the log-OR is target class. The binary headline, and especially the viral subgroup, depend on strain-specific accessions. The investigator's q=0.009 does not hold after adjustment. Strength should drop from moderate to weak.

## Recomputed values

FINDING: PM1_less_competition (confounding lens). Cryo-EM papers deposited 2018 onward, error (n=121) vs clean (n=1256). Paper level, logit with lab-clustered SEs unless stated.

REPRODUCED
- any_comp (investigator model): exact, OR 0.469 (0.316-0.696), p=1.7e-4.
- log2 count: OR 0.895 per doubling, p=0.0013.

SURVIVES (OR for any_comp)
- Lab-clustered SE: 0.469, p=0.001.
- Deposit year categorical: 0.464. Deposit-date spline: 0.471.
- Plus log2 number of antigen accessions and log2 number of entries: 0.468.
- Plus lab antibody experience: 0.482. Plus new-V-region share: 0.473. Plus SARS-CoV-2 and viral target: 0.473.
- Plus paper median lq_z: 0.411, p=5e-4. No-competitor papers have better local fit (0.33 vs 0.05).
- Plus share of V chains with unbuilt residues: 0.48.
- Contacted antigen only (partner and fiducial accessions removed): 0.47.
- Only papers with a complete +180 d window (deposited before 2025): 0.41.
- Copied-only papers added as controls: 0.47.
- Leave-one-error-lab-out (parsimonious model): OR 0.51-0.63, p 0.010-0.086.
- Unit level (3869 antibody units, 159 errors), parsimonious: 0.49 (0.30-0.81), p=0.005; q=0.017 within this verification.

ATTENUATED
- Adding target class:
  - capsid only: 0.52;
  - enzyme only: 0.52;
  - capsid + enzyme: 0.60 (0.37-0.97), p=0.037;
  - plus fiducial-only: 0.62, p=0.050;
  - 13-class mode: 0.57, p=0.028.
- Parsimonious model (period, size, resolution, capsid, enzyme, fiducial-only, SARS2): 0.60 (0.35-1.01), p=0.057, q=0.10. log2 count 0.86, p=0.044.
- Full model (spline time, size, class, lab experience, new V, SARS2, viral): 0.62 (0.35-1.09), p=0.096.
- Mantel-Haenszel, class (6) x period (2): 0.66 (0.42-1.03), p=0.064.
- Plus nanobody, mouse and germline identity: 0.52, p=0.027.
- Error rate by competitor status within class:

| Class | No competitor | Competitor |
|---|---|---|
| Capsid | 30% (12/40) | 14% (3/22) |
| Enzyme | 28% | 28% |
| Non-viral other | 8.6% | 4.9% |
| Viral other | 11.5% | 7.2% |

- Family-level antigen, RCSB 30% clusters, PDB-wide ±180 d:
  - any competitor 82% vs 87%;
  - base OR 0.58, p=0.035;
  - parsimonious 0.75, p=0.33;
  - log2 count 0.88 (p=0.004) base, 0.89 (p=0.065) parsimonious.
- Family-level antigen, 70% clusters: base 0.63 (p=0.07); parsimonious 0.80 (p=0.44).
- Viral non-SARS2 subgroup:
  - accession level: 37.5% vs 59.8%;
  - 30% cluster: 87.5% vs 91.4%, OR 0.69, p=0.42;
  - 70% cluster: OR 0.69, p=0.39.
- Antibody-structure-only competition (antigen_competition lens, ±6 or ±12 months, accession or family): base 0.46-0.58 (significant); parsimonious 0.65-0.76 (p 0.17-0.33).

NULL
- X-ray, 21 error vs 452 clean papers: OR 1.26 (0.55-2.90), p=0.59.
- Experienced labs (10 or more prior antibody entries), adjusted: 1.17, p=0.77.
- Non-viral targets, adjusted: 0.80, p=0.54.
- Deposited 2018-21, adjusted: 0.83, p=0.75.
- All-time antigen popularity in the full model: OR 0.97, p=0.68.
- Competitor-before and competitor-after windows are equally predictive (base 0.54 vs 0.50; full 0.69 vs 0.69). This points to a niche property, not a race.

FILES (all under /home/user/SAbR/register_error_analysis/retest/deeper/verify/paper_meta_PM1_less_competition_confounding/)
- Scripts: v1_build.py, v2_models.py, v3_more.py, v4_alt_measures.py, v5_family_pdbwide.py, v6_antibody_covs.py.
- Tables: paper_table_verify.csv, paper_table_model.csv, paper_table_model2.csv, family_competition.csv, leave_one_lab_out.csv, tests_confounding.csv, tests_confounding_2.csv, tests_confounding_3.csv, tests_confounding_4.csv, tests_confounding_5.csv, tests_all_bh.csv (139 tests, one BH correction).
- Cache: RCSB cluster files under /tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad/deeper/cache/verify_PM1/.

CAVEATS
- Target class, the GraphQL entity map and lab experience come from the antigen_competition lens (heuristic classes).
- 30% clusters can merge unrelated proteins, and can split multi-domain viral polyproteins.
- Paper-level n_err=121 limits the adjusted models (wide CIs).
- Unit-level models weight large papers more heavily.
