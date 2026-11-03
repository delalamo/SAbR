# Adversarial check: model_fingerprint family on the paper-confirmed set

Scripts in `scripts/`, outputs in `tables/`. All code is the verifier's own. The investigator's binsuite/common were
not used. Inputs: Table S1 xlsx, SAbR jsonl, deeper/runs_all.csv, chains.csv.gz, chain_flags.csv.gz,
pdb_meta chain_cis_sheet, validation tallies, and the investigator's framework-RMSD values (fw_computed.csv plus DE2).

## 1. Case set (v01_caseset.py -> tables/v01_caseset.csv)

The independent re-derivation reproduces `paper_errors_annotated.csv` exactly:

- 162 Table S1 errors on scanned chains, 131 of them with at least 50% of residues in the V region.
- Statuses: 69 / 16 / 4 / 5 / 1 / 36.
- 0 mismatches in status, overlap or run.
- Every Table S1 error overlaps at most one run.
- No chain has more than one ok SAbR record, so file order does not matter.
- Strict set: 67 chains, 38 entries, 25 papers, 49 EM. The control set has no err_any chain and nothing released
  after the window.

**One discrepancy in the write-up.** The dominant paper (Zhou 2021, 10.1038/s41422-021-00487-9) has **19** of the
67 strict chains, not 17. These 19 chains are 39% of the EM chains.

## 2. Recomputed numbers (v02_recompute.py, v03_paperlevel_het.py, v04_selection_paperlevel.py)

**T1 Pro8 (reproduced).**

- EM: 0/18 (6 papers, 12 from Zhou) vs 615/895. Haldane OR 0.012, p=1.2e-9.
- Year-matched controls (2018-21): 66% cis.
- 24% of EM control papers are all-trans, so the chance that all 6 case papers are all-trans is 0.24^6 = 2e-4.
- Caveat: 10 of the 18 EM chains are FR1 slips, where Pro8 lies inside the slipped segment. Of the 8 non-FR1 chains,
  5 are from Zhou, and Zhou has no cis peptide anywhere. Without FR1 slips and without Zhou, the evidence is 3 chains
  in 3 papers.

**T9 (reproduced).**

- Pooled: 12/32 vs 94%.
- Without Zhou: 12/13. Cκ is 4/4 cis and CH1 8/8, both at or above the controls. The pooled point estimate is
  below 1 only because of one λ chain (0/1).

**T3 Ramachandran beyond site (reproduced).**

- EM chain level +8.1 (48 chains, 19 of them Zhou; the Zhou chains average 20.5%).
- Paper level +6.1 (bootstrap +1.9 to +10.3).
- Rest of the deeper set: +2.2 at paper level.
- **Paper vs rest at paper level: Welch p=0.11, MW p=0.15; without Zhou p=0.18.** The reported heterogeneity
  (z p=0.03, 3-group p=0.017) comes from chain weighting of Zhou and from cluster-robust SEs with few clusters.

**T4 gap.** At paper level the paper set is larger than the rest (+6.4 vs +1.6, Welch p=0.024; without Zhou
p=0.04). This heterogeneity claim survives.

**T8a sheet.** At paper level −0.10 vs rest −0.07, p=0.38. The "stronger than rest" claim does not survive. The
within-entry result of 9/10 lower comes from 5 papers, and the paper-level sign test gives p=0.06, not 0.02.

**T5 framework RMSD.**

- EM chain level +0.16 Å. The 19 Zhou chains average 0.976 Å, *below* the controls' 1.045 Å. Without Zhou the
  chain-level difference is +0.30.
- Paper level +0.32 (bootstrap +0.12 to +0.53) vs rest +0.47: het p=0.26 (without Zhou p=0.36).
- D/E slips: +0.04 with Zhou, **+0.19 without Zhou** (8 chains, 5 papers).

So the attenuation, and the "near-null for CDR3/FR4 slips" claim, come from one paper.

**Conkit-selection contrasts at paper level.** EM paper-confirmed papers vs conkit-missed papers in the window:

| Measure | Reported (chain level) | Paper level |
|---|---|---|
| Framework RMSD | p=0.03 | 1.34 vs 1.59 Å, Welch p=0.17, MW p=0.36 |
| Ramachandran | p=0.006 | 12.6 vs 8.0, Welch p=0.08, MW p=0.16 |

The selection story ("conkit inflates T3/T4/T8a and deflates T5") is therefore mostly a Zhou-weighting story at
chain level.

**T6 (reproduced).**

- EM: 7/14 vs 17/554, OR 30.7. Paper level 4/9 vs 10/273, OR 20.5. Without Zhou: 4/11, OR 18.
- VH errors that conkit missed have VL errors in only 4/29 (14%) of entries, against 7/14 (50%) when conkit flagged
  the VH error.
- The OR of 31 is therefore inflated by conkit's preference for wholly misbuilt entries. The same-era comparator
  (OR 8.9) is the fairer reference. The direction holds.

**T7b.** EM per-entry mean +0.23 Å. Per paper +0.19 Å, t p=0.34 and Wilcoxon p=0.41 (11 papers). The EM p=0.044 is
entry-level only, and the trend is better called "null still null".

## 3. Verdict audit

| Trend | Verdict | Assessment |
|---|---|---|
| T1 | holds | Agree. Fragile outside FR1 slips and outside Zhou. |
| T1b | holds | Agree. |
| T1c | null still null | Agree. |
| T2 | holds | Agree. The non-L3, non-Zhou evidence is about 3 chains. |
| T3 | holds | Agree. "Larger than rest" is not paper-robust. |
| T4 | holds (EM) | Agree. Heterogeneity is paper-robust. |
| T5 | attenuated | **Disagree.** At paper level the effect holds, with a CI that excludes 0, and it is not significantly smaller than the rest. The chain-level attenuation comes from the 19 near-normal Zhou chains, not demonstrably from conkit selection. |
| T6 | holds | Agree. The magnitude is inflated by selection. |
| T7a | null still null | Agree. |
| T7b | same direction, underpowered | Better labelled **null still null**: the per-paper difference is not significant. |
| T8a | holds | Agree. "Stronger than rest" is not paper-robust. |
| T8b | holds | Agree, as a subset test. Not recomputed. |
| T9 | absent or reversed | Agree on the point estimates (Cκ 4/4, CH1 8/8 without Zhou), but this is uninformative (minimum detectable OR about 0.06), not evidence against the trend. |

**Headline overclaims:**

- "Geometry, gap and sheet are larger than in the rest" holds only for the gap.
- The causal attribution of T5's attenuation, and of the larger geometry effects, to conkit selection rests on
  chain-level contrasts dominated by Zhou 2021.
