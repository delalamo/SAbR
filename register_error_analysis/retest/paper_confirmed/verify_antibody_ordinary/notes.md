# Adversarial check: antibody_ordinary re-test (paper-confirmed set)

Scripts are in `scripts/` and outputs (tables and logs) in `tables/`. All code is mine. The only investigator outputs I
read were per-Fv and per-site measurements (`struct_fv_frame.csv`, `struct_site_err.csv`) and the `rest_win` labels.

## 1. Case set re-derived independently (v01_case_set.py)

I matched Table S1 against SAbR records (validation, then snapshot_rest, then new; later wins), `runs_all.csv`
(i0/i1 index the IMGT-numbered residues; 0 index mismatches) and the synthesis ledger.

- **Errors:** 162 Table S1 errors fall on scanned chains, and 131 of them have at least 50% of residues in the V region.
- **Status counts:** 69 / 16 / 4 / 5 / 1 / 36. This is identical to `paper_errors_annotated.csv`: 0 status discrepancies,
  and every error maps to the same run.
- **Record precedence:** no duplicate records differ, so the opposite precedence used in build_cases.py for the V
  fraction has no effect.
- **Best-overlap rule:** taking the best-overlapping run never hides a first-occurrence run.
- **Strict set:** 67 chains, 38 entries, 34 units, 25 papers (49 EM chains, 18 X-ray). The last release is 2022-02-02.
  All of this agrees with the investigator.
- **Discrepancy 1 (controls; minor).** `paper_flagged_entry` is documented as "entry appears anywhere in Table S1".
  It is implemented as "entry has one of the 131 V-region errors on a scanned chain". As a result, 176 control units
  (90 EM) come from entries that Table S1 does flag elsewhere (2,378 control units vs 2,202 under the documented
  definition). Key numbers barely move: has_hq, rare length, alpaca and non-human ORs change in the second decimal.
- **Discrepancy 2 (comparison set; matters).** The 37 `rest_win` units are described as "errors conkit missed".
  - 9 of them sit in Table S1-flagged entries.
  - In 4 of those, conkit flagged the error chain itself.
  - This includes both units of the bovine paper 10.1371/journal.ppat.1009507 (7D3K/L/M/R). Conkit flagged 7d3m:H
    96-101, 7d3r:H 95-101 and 7d3k:L 50-58.
  - Our pipeline dropped these errors itself: the 7d3m run is at IMGT 111-112 inside the ultralong CDR3 and does not
    overlap, and 7d3r is a ledger DE_chain_end artifact.
  - Only 28 units are truly unflagged by conkit. Among them, 2 are bovine, 2 rabbit and 12 of 28 (43%) have a rare
    CDR length.

## 2. Own recomputations (v02 to v05)

**X-ray ≤3 Å representation (has_hq)**

- Overall: 3/34 vs 1098/2378, crude OR 0.11 (0.03-0.37).
- EM: 3/24 vs 30.1%, OR 0.33 (0.10-1.12), Fisher p=0.071. My year-, resolution- and paper-adjusted GEE gives 0.34
  (0.11-1.03), p=0.056, against the investigator's p=0.04.
- X-ray: 0/10 vs 56.7%, p=2.5e-4.
- The "46%" in the headline mixes methods: standardised to the cases' method mix, the control rate is 38%.
- 218 control units have an X-ray chain at exactly 3.0 Å, so their own structure makes has_hq true. Excluding them
  gives X-ray 0/10 vs 49%, and the pooled adjusted OR is 0.18 (0.06-0.59).
- Verdict: holds when pooled and in X-ray. In EM it is borderline.

**Rare CDR length**

- Crude pooled: 5/34 vs 26.1%, OR 0.49 (0.19-1.27), p=0.17. The investigator's M1 gives 0.35, p=0.049.
- EM: 2/24, 0.22, p=0.022.
- X-ray: 3/10, 1.36.
- The "depleted" claim is EM-only and borderline among about 75 tests.

**Species**

- Alpaca: crude 3.85 (1.65-8.99), p=0.005; EM 3.26 (1.18-9.0).
- Non-human or mouse: crude 2.57 (1.11-5.97), p=0.034.
- All 7 non-human cases are VHH (alpaca), in 6 papers.

**VH-VL orientation outliers (v03)**

- The strict case Fvs include every NCS or repeated Fab copy in an entry (6 entries have 3 copies, 7 have 2). The
  controls have one Fv per entry.
- EM, 37 instances: 10/37 vs 23/150 (reproduced). These collapse to **15 distinct Fvs in 12 papers, with 4/15 (27%)
  outliers vs 15%**, OR 2.0.
- The dominant paper contributes 3 distinct Fvs, and **1** of them is an outlier (7 copies).
- **Without that paper: 3/12 (25%) vs 15%. The reported "15% vs 15%" is an artifact of copy-weighting.**
- Mahalanobis, one value per distinct Fv: +0.42 (+0.54 without the dominant paper), against the investigator's +0.29.
  Against the deeper EM controls (2.57) it would be about +0.9. The rest's figure is +1.44.

**Distorted cores**

- Core RMSD: +0.25 Å with one value per distinct Fv (p=0.003); +0.28 without the dominant paper.
- Undistorted cores: 1/10 distinct Fvs are outliers vs 15/135 controls. So the investigator's "7/32, OR 1.8" is
  copy-inflated, and the trend is consistent with "explained by core distortion". Only 4 distinct outlier Fvs exist.

**HC2 and LC1 nominal shifts**

- One value per distinct Fv: HC2 EM -1.27°, p=0.18; LC1 X-ray -1.34°, p=0.09.
- The reported p=0.005 and p=7e-4 come from pseudo-replicated copies, with only 12 or 7 clusters for the
  cluster-robust SE.

**ImmuneBuilder site uncertainty**

- EM: +0.160, paper-cluster bootstrap (+0.06, +0.34). Per paper: 14 of 17 papers are positive, Wilcoxon p=0.001.
- Unique sites: +0.13.
- Without the dominant paper: +0.175, p=0.015.
- Holds.

**Earlier structure of the same antibody:** 5/34 vs 30.6%, crude OR 0.39, p=0.058.

## 3. Verdict judgements

| Trend | Investigator | Mine | Why |
|---|---|---|---|
| Mean ABangle unchanged | null still null | agree | The HC2 nominal shift disappears when copies are collapsed (p=0.18) |
| More orientation outliers | attenuated | **same direction, underpowered** | 27% vs 15% per distinct Fv, and 25% without the dominant paper. The "one paper explains it" claim is wrong. Only the Mahalanobis magnitude is smaller than the rest's, and part of that comes from using different controls |
| Outliers explained by distorted cores | holds | agree (weak) | 3 of 4 distinct outlier Fvs have distorted cores; among undistorted cores, 1/10 vs 11% |
| Predicted orientation normal | null still null | agree | |
| Rare CDR length | null still null | agree | But the headline's "depleted, OR 0.35" is not robust: crude pooled p=0.17, X-ray OR >1, one vote per paper not significant |
| Specific canonical lengths | same direction, underpowered | agree | Not recomputed |
| Predicted H3 base (VH and VHH) | null / underpowered | agree | |
| IB site uncertainty | holds | agree | Robust per paper |
| Germline / novelty / identity / humanisation / rare residues / L123 | null still null | agree | The "same era vs conkit-missed" contrasts use a contaminated comparator (9 of 37 units are conkit-flagged) |
| Earlier same antibody | same direction, underpowered | agree | |
| Non-human | same direction, underpowered | agree | |
| Nanobody / alpaca | same direction, underpowered | agree, borderline | Crude and M1 CIs exclude 1; only one vote per paper does not |
| Rabbit | absent or reversed | **cannot test** | Expected about 1.7 cases; the minimum detectable OR (8) is above the full-set OR (4) |
| Bovine | absent or reversed | **cannot test** | Expected about 1.4 cases; minimum detectable OR 16. The note "missed by conkit" is wrong: conkit flagged the bovine lab's entries, and **our** confirmation and artifact ledger removed them |
| X-ray ≤3 Å representation | holds | agree | Pooled and X-ray hold; EM is borderline (p 0.04-0.07). The headline "8.8% vs 46%" is method-mixed (38% after standardising) |
