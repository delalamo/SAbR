# What were the "unexplained" errors?

In `../README.md`, 38–39% of flagged antibody V domains had no identifiable origin. This follow-up
looks at all 271 such disagreement runs.

## They were not random: they sit at the edges of the hypervariable loops

| | Explained runs | Unexplained runs |
|---|---|---|
| Where the run starts | heavy FR3 61% / FR1 28%; light FR1 83–90% | **CDR3 31–52% and FR4 19–25%** in all chain types |
| Shift | mixed (+1 43%, −1 23%) | **+1 in 69%** |

The most common signatures recur across *different* antibodies:
* heavy IMGT 120, +1: 35 runs, 9 antibodies
* heavy IMGT 112, +1: 28 runs, 11 antibodies
* kappa IMGT 120, +1: 17 runs, 9 antibodies

## Three patterns

1. **CDR3–FR4 junction (121 runs).** SAbR finds that the conserved FR4 anchor, Trp118 (heavy) or
   Phe118 (light), is not at its structural position. The model's CDR3 occupies one residue too few
   or too many before the FR4 strand, so the strand slides by one. Example: 7V2A heavy chains, with
   a 24-residue CDR3 compressed by one position.
2. **FR4 bulge after the W/F-G motif (55 runs).** The two methods agree through Trp/Phe118 and
   Gly119, then SAbR numbers the next residue 121: the strand jumps a position right after the
   glycine.
3. **CDR1 and CDR2 boundaries (54 runs).** These are strongly associated with atypical CDR2 length.
   A run touching CDR2 is 9× more likely when the CDR2 length is rare for that chain type (adjusted
   OR 9.2, 3.7–22.9; heavy 2.6% vs 0.4%).

## They are real modelling errors, not numbering artifacts

* **The anchor residue is pulled out of the core.** This is a structure-only test, independent of
  maps and of both numbering tools.

  | ANARCI's Trp/Phe 118 exposed (<14 Cα within 10 Å) | |
  |---|---|
  | CDR3–FR4 junction runs, EM | 38% |
  | CDR3–FR4 junction runs, X-ray | 38% |
  | Clean V domains, EM | 7% |
  | Clean V domains, X-ray | 2% |
  | High-resolution (<3 Å) templates | 0% |
  | FR4 bulge runs | 17% |
* **X-ray maps disagree with the deposited register.** I ranked how well the segment fits the map
  among the other residues of the same V domain, against the same IMGT window in clean V domains.

  | Pattern | Segment percentile | Control percentile |
  |---|---|---|
  | FR4 bulge | 6th | 37th |
  | CDR3–FR4 junction | 18th | 46th |
  | CDR2 boundary | 12th | 35th |

  In cryo-EM, Q-scores at 3–4 Å do not discriminate (48th vs 52nd), so the structural test above
  carries the EM evidence.
* **The FR4 bulge is not a SAbR convention.** It occurs in 1 of 1,039 high-resolution templates but
  in 99 of 15,897 chains at 3–5 Å.

## What predicts CDR3–FR4 junction errors

* **Method.** Cryo-EM: adjusted OR for X-ray is 0.10.
* **Era.** 2020–21: 2.3% of EM V domains, falling to 0.5–0.6% from 2022.
* **Map quality.** Poor *local* map quality: OR 0.70 per SD better. Unlike flags overall, this
  pattern does track map quality.
* **Antibody features:**
  * unusual kappa CDR3 length (11–13 residues: 7.5% vs ≤1%)
  * nanobodies (1.3% vs 0.8%)
  * mouse antibodies (1.0% vs 0.7% human)
  * V regions far from germline (≤80% identity: 1.2–1.4% vs 0.5–0.6%)
* **Not heavy CDR3 length.** It shows no trend.
* **AlphaFold starting models.** In EM, the rate is 0.34% with an AlphaFold starting model vs
  0.9–1.1% otherwise, but this is not significant after adjustment (OR 0.58, 0.08–4.3; 290 chains).

## Re-assigned origins

Each run is assigned to the first matching category. The "independent events" column counts one per
paper × antibody × position × shift, removing symmetry copies and the same model reused across a
paper's entries.

**34% of runs are the same error repeated in another entry of the same paper.**

| Origin | Chains, snapshot | Chains, newer | Events, snapshot | Events, newer |
|---|---|---|---|---|
| Chain end / next to an unbuilt loop | 26% | 40% | 29% | 39% |
| **CDR3–FR4 junction** | **22%** | **14%** | **28%** | **19%** |
| **FR4 bulge (IMGT 120)** | 6% | 11% | 4% | 8% |
| **CDR1/CDR2 boundary** (incl. atypical CDR2) | 5% | 10% | 9% | 11% |
| Copied from an earlier structure (unambiguous) | 9% | 7% | 7% | 6% |
| HIV FR3-insertion bnAbs / FR3 convention (ambiguous) | 27% | 14% | 18% | 6% |
| **Still unexplained** | **5%** | **5%** | **5%** | **11%** |

The FR4 bulge's rise in newer structures is mostly one paper's symmetry copies: 18 of 55 runs come
from 2 entries of one 2024 eLife paper.

## Synthesis

Register errors in antibody V domains arise where the modeller cannot confidently place every
residue:
* at chain ends and next to unbuilt loops;
* at the boundaries of the hypervariable loops. An off-by-one residue count in CDR3 (or an unusual
  CDR2) is absorbed by sliding the adjacent framework strand.

The FR4 strand is the most exposed. Its anchor, Trp/Phe118, gets pulled out of the core in about 40%
of these cases, a defect a simple geometric check would catch.

Once made, an error spreads: through templates, across symmetry copies, and across the entries of a
paper. It is almost never revised.

## Files

* `scripts/`:
  * `unexpl_build.py`: run-level table.
  * `unexpl_patterns.py`: pattern classes and map fit.
  * `unexpl_w118.py`: anchor burial.
  * `unexpl_correlates.py`: correlates.
  * `unexpl_cdrlen.py`: CDR-length test.
  * `unexpl_final.py`: re-assigned origins.
  * `anarci_germ.py`: germline assignment.
* `tables/`: `final.log`, `events.log`, `w118_burial.csv`, `flag_origin_v2.csv`,
  `runs_origin_v2.csv`.
