# Verification of SN2_trunc_FR4_modesDE (confounding lens)

**Claim checked:** CDR3–FR4 slips (D modes) and FR4 bulges (E) are enriched for SEQRES that ends inside FR4 (last
ANARCI IMGT position below 125).

**Verdict: weakened.** The raw counts reproduce, and the association survives adjustment for year, chain type,
resolution, SARS-CoV-2 and paper clustering. But it is not an independent or location-specific signal:

* one of the 4 D cases is a numbering artifact;
* the same truncation is just as common in N-terminal (mode A) errors;
* it disappears entirely once you account for the simpler known factor that the *model* ends inside FR4.

## Reproduction
Cryo-EM units, first-occurrence errors vs clean units of the same chain types:

| Errors | Errors truncated | Clean units truncated |
|---|---|---|
| D (D+, D−, D0) | 4/59 | 35/4036 |
| E | 2/18 | 35/4036 |
| Any error | 9/183 | 35/4036 |

These match the investigator's numbers. Firth-penalised M2 model: D OR 14.7, p = 3e-4; E OR 17.0, p = 0.007.

## What weakens it
1. **One D case is a definitional artifact: 9cj7 H (Ward 2024).**
   * The SEQRES is a full Fab heavy chain of 293 aa. Its CDR3 (`…EMDVWGQGTDDYSYAMDVWGQGTTVTVSS`) contains an internal
     WGQG motif.
   * ANARCI stops the V domain at that internal motif (IMGT 122). So both the "truncation" and the D0 run (no shift
     value, reaching that position) come from the same ANARCI parse.
   * Four clean controls have the same kind of parse stop.
   * Corrected D: 3/59 vs 32/4036, Fisher p = 0.013.
2. **Not specific to D/E.**
   * SEQRES ending inside FR4 is just as enriched in mode A (FR1 N-terminal) errors: 4/59 (6.8%), Firth OR 10.0.
   * In all non-D/E error units: 4/107, OR 5.6.
   * In controls, FR4-truncated SEQRES co-occurs with N-terminal truncation (44% vs 7.5%). This is the "SEQRES =
     modelled residues only" deposition practice.
3. **Fully absorbed by a simpler known factor: the model ends inside FR4.**
   * The last modelled residue is below IMGT 125 whether the SEQRES is truncated or the residues are simply unbuilt:

     | Mode | Errors | Clean | Firth OR |
     |---|---|---|---|
     | D | 39% (23/59) | 3.0% | 27.5 |
     | D+ | 44% | 3.0% | 33 |
     | E | 22% | 3.0% | 10 |

   * Adding this term to M2 removes the effect of SEQRES truncation:

     | Errors | SEQRES-truncation OR | 95% CI | p |
     |---|---|---|---|
     | D | 0.67 | 0.19–2.4 | 0.23 |
     | E | 2.7 | 0.5–15 | 0.27 |
     | Any | 0.83 | | |

   * Among units whose model ends inside FR4, a truncated SEQRES carries no extra risk: 3/23 D errors are truncated vs
     32/122 controls (OR 0.42).
   * "Model ends inside FR4" is itself raised in every error mode: A 24%, B 16%, C 12.5%, D0 67%, D+ 44%. It is
     strongest for D, so the location effect is in the *model end*, not in the SEQRES.
4. **Every truncated D/E case is a chain-end run.**
   * All 4 D and 2 E truncated cases have runs that reach the last modelled residue.
   * There are no truncated cases among runs that do not reach the chain end (0/14 D; low power).
   * One D+ case (8khr) has a +5 shift over a 3-residue FR4 tail (WGQ), which fits poor anchoring at the chain end.
5. **Fragile.**
   * Dropping any single D case gives p = 0.08; dropping any single E case gives p = 0.13.
   * One unit per paper with cases kept (2,000 draws): D median p = 0.054, with p < 0.05 in only 45% of draws. E holds:
     median p = 0.008.

## What does NOT explain it (it survives these)
* **Deposition year or period.** Exact stratified p is 0.009 for corrected D and 0.008 for E. Truncation among clean
  units is lowest in 2020–21 (0.5%), so the error peak does not drive it.
* **Year × chain type strata:** D p = 0.004.
* **Resolution, nanobody, non-human/mouse species, SARS-CoV-2:** the Firth M2 models still show the effect.
* **Paper clustering:** cluster-robust D OR 9.6, p = 3e-5. The truncated cases come from 4 papers and 4 labs; the
  truncated controls from 28 papers and 28 labs.
* **Alternative control** (all units without an error at this location): D OR 10.4, E OR 17.5.

## Positive nuance
* The two genuinely truncated D+ cases (7wtf J, 8khr H) show 0 unbuilt residues in the deepdive's ±5 window, because
  the missing residues are absent from the SEQRES, and their site map fit is ordinary.
* The deepdive would therefore count them as "no gap". Measuring the chain end from the model rather than the SEQRES
  closes that blind spot.

## Files
* **Scripts:**
  * `v01_features.py`: unit features, model ends, SEQRES tail, chain-end runs;
  * `v02_tests.py`: all tests, with BH correction in `tables/v02_tests.csv`;
  * `v03_extra.py`;
  * `v04_mode_specificity.py`.
* **Tables:**
  * `tables/unit_features.csv.gz`;
  * `tables/runs_first_with_cterm.csv.gz`;
  * `tables/v02_tests.csv`;
  * `tables/v02_tests.log`;
  * `tables/v04_mode_specificity.csv`;
  * `v03_extra.log`.
