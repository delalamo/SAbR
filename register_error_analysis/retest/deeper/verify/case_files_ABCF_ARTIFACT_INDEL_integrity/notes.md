# Verification: case_files_ABCF / ARTIFACT_INDEL (integrity)

The verifier returned this verdict without writing a notes file. The text below is its structured return, copied unchanged from the workflow log.

**Verdict: confirmed**

## Reason

I reproduced the claim cleanly, and the cause turns out to be stronger than the investigator stated. Some n's are inflated and some of the wording needs correcting.

(1) The unit counts reproduce exactly from runs_first_occurrence.csv after removing the amyloid entries (v03):
- C_FR3: 31 units / 44 runs, of which 6 units / 12 runs are artifacts (2x7l, 9bop, 9e89, 30bg, 30ad, 9pf9). That is 4 of the 7 X-ray C units.
- F_CDR1: 13 units, of which 2 units / 7 runs are artifacts (8gs*, 7eaj/7eak).

(2) The FR3 part is built into SAbR's code, so the "high-resolution X-ray" test is close to tautological. src/sabr/corrections.py (de_loop_positions / correct_de_loop) renumbers the residues between the IMGT 79 and 85 anchors by count alone:
- a one-residue deletion is always placed at 81;
- a two-residue deletion is always placed at 81+82.

ANARCI places the gap with its sequence model (HMM), so whenever it puts the gap at 84, 80+84 or 84+85, a 3-residue run at 81-83 appears whatever the resolution or model quality.

(3) The gap position is set by single residues, not by structure (v04). With the scan's ANARCI setting (human and mouse HMMs):
- T→A in the rabbit control 9p4c (ISKTSSTTV→ISKASSTTV) moves ANARCI's gap from 81 to 84;
- A→T in 30ad, and A→S in 9bop, moves it back to 81;
- with all-species HMMs, every one of these sequences gets the gap at 84.

So the "gap at 81 never disagrees" control holds by construction and depends on the ANARCI species setting. Rerunning the 9p4c, 4ztp and 6cjk controls with all species turns them into 81-83 runs.

(4) The strongest independent test: SAbR on ImmuneBuilder and IgFold models of the same sequences (v02). These models cannot contain a deposition error.
- 2x7l, 9bop, 30ad, 9pf9, 8gs* and 7eaj: both predictors give the identical run (81-83 or 27-35).
- 30bg and 9e89 (gap 84+85): IgFold gives the run, ImmuneBuilder does not. For two-residue deletions the result depends on where the anchors land in the input structure. For these two the support is still good:
  - every crystal structure of this pattern I tested, 5/5 (4z9k, 7mlh, 8zca, 30bh, 5trp), also gives the run;
  - 30bh is a near-identical relative of 30bg from the same lab.
- Positive control: 9ml5 also carries the gap at 84. Its predicted models give 81-83, but the deposited model gives 85-95 with shift -1. That run is a genuine error, and s10 correctly did not flag it.

(5) I extended the high-resolution sets beyond the investigator's picks (v01 searches the whole PDB; v02 reruns SAbR):
- Gap at 84: 7 of 7 fully built X-ray chains disagree in the claimed window. 7kbk is the only exception, and it has an unbuilt residue inside the DE loop (IMGT ~79-85), so SAbR skips the rule there.
- FR1 gap at 26: 2/2 independent antibodies (5aam 2.49 Å, plus 1ghf 2.7 Å).
- In the scan, 0 of 67 fully built chains with the gap at 81 disagree, across 22 entries.

Corrections the claim needs:
- "4/4" for the gap at 84 counts 4o4y and 4ma3, which are the same antibody, so it is 3 distinct antibodies below 2.6 Å.
- The "1/1" for 80+84 is 5dhv, the same rabbit antibody as 2x7l from the same lab. It is backed up by the predicted models of 2x7l.
- The germline labels in the detail line are wrong for 3 chains. 2x7l, 9bop and 30ad are rabbit VH, misassigned to human IGHV3-66/3-53 at 55-61% identity. 20 of the 22 entries in the gap-81 control are also rabbit-like. The actual shared feature is the shorter rabbit FR3, not human IGHV3-53/3-66.
- 9yfu (gap at 85, run 81-84) is probably a seventh artifact unit, so the C count is slightly low:
  - both predicted models reproduce its run exactly;
  - 9jrt and 9jro are not counterexamples because their DE loop is unbuilt.
- s08 bug: it labels everything in validation.jsonl as "hires", but that file contains 3-5 Å EM chains (7te9, 7eaj). Its n_hires / deterministic flags are therefore wrong (for example, the gap at 96 is marked deterministic). The final s09/s10 results do not depend on this.
- The "44/44 clean" control count does not reproduce exactly (I get 67 chains / 22 entries), but the direction is the same.

There are no p-values to check: the test is deterministic.

## Recomputed values

Units (v03, amyloid removed):
- C_FR3: 31 units / 44 runs; artifacts 6 units / 12 runs; X-ray C units 7, of which 4 are artifacts.
- F_CDR1: 13 units / 20 runs; artifacts 2 units / 7 runs.

High-resolution X-ray, SAbR rerun with scan-matched ANARCI (v02):
- Gap at 84: 7/7 fully built chains (4 distinct antibodies: 9f1i, 9f18, 6dbd, 4o4y, 4ma3, 4o51, 4rzc) show the 81-83 +1 run. 7kbk is excluded: unbuilt residue in the DE loop, so the rule is skipped.
- 80+84: 2/2 chains (5dhv, same antibody as 2x7l).
- 84+85: 5/5 distinct antibodies (4z9k, 7mlh, 8zca, 30bh, 5trp), run 81-83 with +2.
- FR1 gap at 26: 2/2 distinct antibodies (5aam, 1ghf), run 27-35.
- Gap at 81: 0/3.
- Gap at 96 (7te4): no run.

Predicted models of the flagged antibodies (ImmuneBuilder / IgFold):
- Identical run in both predictors for 2x7l, 9bop, 30ad, 9pf9, 8gs*, 7eaj, and also 9yfu.
- 30bg and 9e89: IgFold yes, ImmuneBuilder no.
- 9ml5: predicted 81-83 versus deposited 85-95, consistent with a genuine error.

Scan determinism (fully built windows):

| Pattern | Chains | Entries | Chains disagreeing |
|---|---|---|---|
| FR1:26 | 7 | 6 | 7 |
| 80+84 | 10 | 4 | 10 |
| 84 | 6 | 4 | 4 (the other 2 are 9ml5's genuine error) |
| 84+85 | 2 | 2 | 2 |
| 81 | 67 | 22 | 0 |

ANARCI point mutants (v04): a single T↔A or A↔S change moves the gap between 81 and 84; all-species HMMs give 84 for every sequence.

Files (all in /home/user/SAbR/register_error_analysis/retest/deeper/verify/case_files_ABCF_ARTIFACT_INDEL_integrity/):
- v01_pattern_search.py
- v02_sabr_rerun.py
- v03_scan_recount.py
- v04_anarci_point_mutants.py
- tables/pdbwide_patterns.csv
- tables/rerun_residues.csv
- tables/scan_pattern_chains.csv
- tables/v03_scan_recount.txt
- tables/anarci_point_mutants.txt
