# Verifier (confounding lens): FR3_INDEL_C

**Verdict: weakened.** Period, SARS-CoV-2, chain type, resolution and lab/paper clustering do not explain the association.
Most of it, though, comes from the error definition itself. At an FR3 indel, SAbR and ANARCI routinely put the gap in
different places, even in correct high-resolution crystal structures.

After re-classifying the 10 exposed units with evidence from independent structures, 4 units remain (7te9, 7y0o, 9dm7
and 9ml5). Only 2 of them, 9dm7 and 9ml5, are clear long register slips. The residual association is real but small in
n, and it is weak:
* OR about 6–18;
* p = 1e-4 to 0.03.

## Reproduction

The investigator's numbers are reproduced exactly: C units 10/24, clean units 59/4,036, adjusted OR 52 (19–141). See
`v01`, `v03`, R0.

## What does NOT explain it (classical confounders)

The OR stays about 40–60 under each of these:
* adding SARS-CoV-2 status;
* cluster-robust errors by lab;
* one unit per paper;
* Mantel-Haenszel over year bin × chain class;
* heavy chains only;
* non-SARS-CoV-2 units only.

All are in `tables/tests.csv`.

## What does explain most of it (the definition, `v02`, `v04`–`v07`)

1. **The detector flags FR3 indels in correct structures.**

   | Structures | FR3-atypical chains flagged | FR3-typical chains flagged |
   |---|---|---|
   | X-ray ≤ 2.5 Å (`v06`) | **40% (19/47)** | 0/50 |
   | X-ray 3–5 Å | 21% | 0.14% |
   | EM 3–5 Å | 40% | 0.7% |

   "Flagged" means ≥3 FR3 residues where SAbR ≠ ANARCI. Another 37–40% of FR3-atypical chains have 1–2 residues off,
   against about 0% of typical chains.

   EM units versus high-resolution crystals with the same indel pattern: 26 flagged observed against 19 expected; MH OR
   4.6 (0.8–28), p = 0.047.

2. **Every exposed run sits on the indel.** All 10 exposed C runs touch the ANARCI gap or insertion position, and 6 are
   3–4-residue runs. No C run away from an indel is associated with FR3 length (R5: 0/14).

3. **The control arm is depleted by design.** Controls are units with no run of any kind (`err_any` False), but 28 of
   the 106 EM FR3-atypical units are "copied/artifact-only" and can never be controls. Using all non-C units as
   controls lowers the OR from 48 to 31 (R3).

4. **Independent structures of the same sequence** (`v04`; SAbR run on PDB structures):

   | Unit | What the independent structures show | Classification |
   |---|---|---|
   | 7m3l | Identical 90–95 −1 disagreement in 3gk8 (2.0 Å X-ray) and 7m3n (2.4 Å EM) | Artifact |
   | 8c8t | 85A–D insertion; 31/31 other-paper chains disagree, and sibling chain R is already classed as an artifact | Artifact |
   | 9yfu | Identical disagreement in 8tb7 (2.94 Å); none in 9jrt or 9jro (same VHH) | Probable artifact |
   | 8xlk | Same V and identical 93–95 run as the earlier 7te9 from another lab; the copy rule missed it because SEQRES carries a signal peptide | Copy |
   | 9bop, 9e89 | (removed by the investigator) | Artifact |
   | 7te9 | 7te4 (2.46 Å) agrees, though its chain H is 1 residue off | Model-specific |
   | 7y0o | 7y0n has 1 residue off | Model-specific |
   | 9dm7 | Long 72–92 run | Model-specific |
   | 9ml5 | 85–95 −1, opposite to the 81–83 +1 seen in every high-res (84,) rabbit structure | Model-specific |

## Residual association after correct removals (`tables/refit.csv`)

| Test | Error units | Control units | OR (95% CI) | p |
|---|---|---|---|---|
| S4: 4 artifacts + 9yfu + 8xlk removed | 4/18 | 59/4,036 | 18 (5–64) | 1e-5 |
| S5: S4, controls = all non-C units | 4/18 | 96/4,224 | 11 (3.4–38) | 1e-4 |
| S8: only 9dm7 and 9ml5 counted | 2/16 | 1.46% | 9.5 (1.9–48) | 0.006 |
| S8b: S8, controls = all non-C units | 2/16 | 2.27% | 5.9 (1.2–29) | 0.03 |

None of the residual cases falls in the 2020–21 error peak: 0/6 C units there are exposed, and every exposed C unit
dates from 2022 or later.

## Errors in the investigator's materials

* **Mislabelled "high-resolution" data.** `s08` labels `sabr/out/validation.jsonl` as a "<3 Å high-resolution" set.
  Every entry in it is 3.0–5.0 Å. So the `n_hires` and `hires_dis` columns in `gap_patterns.csv` are not
  high-resolution evidence. The `s09` checks did use real high-resolution structures and are fine.
* **The "44/44" detail is wrong.** The claim is that clean chains with a one-residue FR3 deletion have the gap at 81
  (44/44). In fact, clean EM heavy chains with a single FR3 deletion have it at 81 (44), 83 (5), 85 (2) and 94 (3).
  The 85 and 94 clean chains are the *same antibodies* as the error units 9yfu and 7y0o. Gap position does not decide
  the outcome.

## Nulls

* Next-to-unbuilt residues: 2/10 FR3-atypical C runs against 7/14 typical C runs (p = 0.21). The direction is the
  opposite of the "poor-build" explanation.
* Local Q-score percentile at the run: 0.29 against 0.44 (p = 0.20).
* FR3-atypical length does not mark A, D+, D−, E or "other" errors (OR about 0.7–4, q ≥ 0.3).
* The F mode association is borderline at q = 0.05, with 1 unit.
