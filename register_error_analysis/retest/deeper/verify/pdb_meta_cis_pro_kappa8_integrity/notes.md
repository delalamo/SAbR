# Verification: pdb_meta / cis_pro_kappa8 (integrity)

The verifier returned this verdict without writing a notes file. The text below is its structured return, copied unchanged from the workflow log.

**Verdict: confirmed**

## Reason

The association is real. I rebuilt the frame myself from chains.csv.gz, the SAbR/ANARCI per-residue output and the cached RCSB CIS-PEPTIDE features. Every headline number reproduces exactly. I also checked the annotation against the coordinates and it is not an artifact. However, part of the effect is inflated by paper clustering. The within-paper evidence is weak. Some wording overstates it.

CHECKS THAT PASSED
- Index join is correct. In sabr_scan.py, seq_index = pdbx_poly_seq_scheme seq_id − 1, which is the same frame as RCSB beg/end_seq_id. RCSB end_seq_id is the Pro (every cis feature spans i to i+1).
- Coordinates agree with the annotation. I recomputed the IMGT7–8 omega from ModelServer coordinates for all 70 error chains, 450 random clean EM chains and 60 X-ray chains. All 580 matched the wwPDB annotation. Annotated-trans error chains have a median |omega| of 174°, so they are genuinely trans. Whole-chain cis counts from coordinates and annotation were identical (846 vs 846).
- Missing residues are not the cause. Residue 7 is modelled in 70/70 error chains with Pro8.
- The FR1 circularity is handled. Dropping error chains with any FR1 disagreement run (including copied or artifact runs) gives 4/41 cis, OR 0.056, p=1e-6.
- Germline and species do not explain it. Error cis rates are about 0–11% in every kappa family, against 53–67% in clean chains. Adding family and species gives OR 0.031 (all) and 0.073 (non-FR1). Adding entries-per-paper and SARS-CoV-2 gives non-FR1 OR 0.072.
- Leave-one-error-paper-out: the worst case is OR 0.049 (p=3e-8) for all errors and 0.099 (p=2e-5) for non-FR1. Dropping the two largest papers (Zhou Q. Cell Res 2021, 17 chains; Hafenstein eLife 2020, 10 chains) gives 0.066 (all) and 0.128 (non-FR1, p=6e-5).
- Paper cluster bootstrap (1000 reps): OR 0.036 (0.010–0.10) for all errors and 0.071 (0.016–0.20) for non-FR1. No replicate crossed OR=1.
- Entry level: OR 0.068 (45 error vs 1,518 entries), p=4e-7. Unit level: OR 0.082, p=9e-7.
- The heavy-chain entry claim reproduces: 30/58 (51.7%) vs 526/1,419 (37.1%), adjusted OR 2.08, p=0.016. It is weak and was not corrected across the investigator's many entry tests.

WEAKENING POINTS
1. Paper clustering inflates the chain-level effect about 3x. Giving each paper one vote per class yields 16% vs 59% cis, OR 0.115 (0.044–0.31), p=1.4e-5. For non-FR1 errors it yields 21% vs 59%, OR 0.16 (0.06–0.44), p=3.5e-4. At paper level (paper has any cis-Pro8 kappa chain) the result is OR 0.22, p=2e-4. The 2.2e-9 rests on 70 chains from 31 papers, one of which contributes 17 chains.
2. Within-paper evidence is weak. 15 error papers also contain clean kappa chains, which are 52% cis by chain. By paper, the error chains are less cis in 5 papers, more cis in 1, and tied in 9 (sign test p=0.22). In 7 of those 15 papers every kappa chain, error or clean, is trans. Within-entry Mantel-Haenszel uses only 5 entries and is not significant (p=0.43). Much of the signal is a paper-level modelling practice rather than chain-specific.
3. The marker is non-specific. 36% of clean EM kappa chains are also trans, and the error rate among trans chains is only 5.7%. Cis status is a high-NPV marker, not something that singles out which structures fail.
4. "Any kappa Fab crystal template has" cis-Pro8 is overstated. Clean X-ray kappa chains at 3–5 Å are 86% cis: IGKV4 60%, mouse 77%.
5. Counting error is minor. The error chains span 31 paper clusters and 26 last authors, not 26 papers and 22 labs.
6. Within resolution strata the effect sits at <4 Å (0/27 and 3/12 error chains cis). At ≥4 Å only 3 error chains remain (2 cis), which is uninformative.
7. The interpretation (traced or threaded, not docked) cannot be tested from metadata: 65/70 error chains have no recorded starting model. Within "none recorded" the error rate is 6.9% when trans vs 0.2% when cis.

Overall, the descriptive claim and the p-value reproduce and are not an annotation, join or circularity artifact. The chain-level effect size and "strong" rating should be read with the paper-weighted OR (0.115–0.16) and the weak within-paper contrast in mind.

FILES (all in /home/user/SAbR/register_error_analysis/retest/deeper/verify/pdb_meta_cis_pro_kappa8_integrity/)
- Scripts: 01_rebuild.py, 02_fetch_coords.py, 03_stats.py, 04_omega.py, 05_paper_weighted.py, 06_hchain_entry.py
- Tables: frame.csv.gz, coord_sample.csv, stats.csv (with BH q), leave_one_paper_out.csv, within_paper.csv, omega_check.csv, paper_weighted.csv
- Coordinate cache: /tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad/deeper/cache/verify_cis8/coords/

## Recomputed values

EM, non-amyloid, kappa chains with Pro8 modelled.

Exact matches with the investigator:
- Cis-Pro8 rate: 5/70 (7.1%) error vs 1,891/2,975 (63.6%) clean. Adjusted OR 0.0363 (0.0122–0.108), p=2.24e-9.
- Non-FR1 errors: 5/42 (11.9%), OR 0.0708, p=1.8e-6.
- Risk: 65/1,149 (5.7%) error when trans vs 5/1,896 (0.26%) when cis. In 2020–21: 36/276 (13.0%) vs 1/467 (0.2%).
- Heavy-chain entry claim: 51.7% vs 37.1%, OR 2.08, p=0.016.

Coordinates vs annotation: 580/580 omega(7–8) values from coordinates agree with the wwPDB annotation.

Robustness:

| Analysis | All errors | Non-FR1 errors |
|---|---|---|
| Excluding any FR1 run | — | OR 0.056 |
| + germline family and species | OR 0.031 | OR 0.073 |
| Leave-one-paper-out, worst case | OR 0.049 | OR 0.099 |
| Drop two largest papers | OR 0.066 | OR 0.128 |
| Paper cluster bootstrap | 0.036 (0.010–0.10) | 0.071 (0.016–0.20) |
| Paper-weighted GLM | 0.115 (0.044–0.31), p=1.4e-5 | 0.16 (0.06–0.44), p=3.5e-4 |

- Paper level, any cis: OR 0.22, p=2e-4.
- Entry level: OR 0.068. Unit level: OR 0.082.
- Within-paper sign test: 5 lower / 1 higher / 9 ties (7 all-trans), p=0.22.
- Within-entry Mantel-Haenszel: 5 entries, p=0.43.
- Clean X-ray kappa chains: 86% cis.
- Counts: 31 papers and 26 labs (not 26 and 22).
