# Verification of `cis_constant_and_L3` (pdb_meta): data integrity and statistics

**Verdict: weakened.**
* The numbers reproduce exactly, and the cis annotation matches the coordinates.
* The headline, however, is chain-level pseudoreplication from one paper whose entries are all-trans everywhere.
* CH1 is null without that paper.
* The constant-domain deficit is explained by entry-wide all-trans models.
* The Pro115 "error elsewhere" set is mislabelled. It is mostly FR1 N-terminal errors 7–10 Å from Pro115.

**Scope.** EM only, the 15 amyloid entries excluded, error = `err_first`, control = `~err_any`. The model is the
investigator's `fit_feature`: logistic regression adjusted for year bin and resolution, with SEs clustered by paper.

## 1. Rebuild from raw caches (`01_rebuild.py`, `02_reproduce.py`)

* **Independent boundaries.** The V/C boundary comes from `vseq` located in SEQRES, not from the SAbR `seq_index`.
  Cis and unobserved residues come directly from `cis_sheet.jsonl`.
* **Agreement with the investigator's table:** 100% on the constant-domain-modelled flag (2,387 chains) and on
  constant-domain cis status.
* **Reproduced:**

| Site | Error | Control | OR | p |
|---|---|---|---|---|
| Cκ | 6/29 | 738/913 | 0.039 | 0.0013 |
| CH1 | 13/31 | 72% | 0.25 | 0.06 |
| Pooled constant | 19/60 | 74% | 0.11 | 0.002 |
| Pro115 | 3/24 | 62% | 0.079 | 0.026 |

* The canonical-site version gives the same result: Cκ FYP 6/29 vs 734/908 (OR 0.039, p = 0.0013).

## 2. The annotation is correct (`03_omega.py`, `05_pro115_checks.py`)

I recomputed ω from the mmCIF coordinates and compared the cis-peptide count with the RCSB CIS-PEPTIDE feature.

| Chains | Agreement |
|---|---|
| All 60 error chains with a modelled constant domain | 60/60 |
| 50 entry partners | 50/50 |
| 80 random controls | 77/80 (3 differ only through the ω threshold) |
| Pro115 in the 28 error chains | 28/28 |

* **No register propagation into C.** Constant-domain cis peptides in error chains sit at the canonical offsets:
  * Cκ: +33 from the V end, as in 713 controls;
  * CH1: +33/+35.

## 3. Pseudoreplication and the dominant paper (`04_robust.py`)

**Clusters behind the Cκ result.** The 29 error chains come from 13 antibody-paper units, 18 entries and 10 papers.
17 of the 29 are from 10.1038/s41422-021-00487-9 (Zhou Q., 2020).

**The dominant paper.** All 9 of its entries have `cis_peptide_count` = 0. That includes the spike and the clean Fabs.

**Robustness tests:**

| Test | Cκ | CH1 | Pooled constant |
|---|---|---|---|
| Chain level, as published | OR 0.039, p = 0.0013 | OR 0.25, p = 0.06 | OR 0.11, p = 0.002 |
| Dominant paper dropped (worst leave-one-paper-out) | 6/12 (50%) vs 81%, OR 0.14, p = 0.025 | 68% vs 72.5%, p = 0.67 | 61% vs 74%, p = 0.06 |
| Paper level (one vote per paper) | 4/10 vs 78%, p = 0.011 | 11/18 vs 72%, p = 0.42 | p = 0.019 |
| Paper level, dominant paper dropped | p = 0.03 | | p = 0.037 |
| Cluster bootstrap of the OR | 0.041, p ≈ 0.0025 | | |

The cluster bootstrap agrees with the cluster-robust p, so the p-value itself is not a computational artefact.

## 4. The deficit is an entry-wide all-trans property

**Restricted to chains whose entry has at least one cis peptide in other chains:**

| Site | Error | Control | p |
|---|---|---|---|
| Cκ | 6/8 | 92% | 0.19 |
| CH1 | 80% | 85.5% | 0.65 |
| Pooled constant | 78% | 87.5% | 0.20 |

**Adjusted for "entry all-trans elsewhere":**

| Site | OR | p |
|---|---|---|
| Cκ | 0.19 | 0.13 |
| Pooled constant | 0.40 | 0.13 |
| Novel-V pooled constant | 0.49 | 0.29 |

**The entry trait itself is only modestly enriched.** An entry with zero cis peptides anywhere: 51.7% of error
entries vs 39.0% of clean entries (OR 1.8, p = 0.005).

**Clean partner chains in error entries:** constant-domain cis 63.8% vs 74.1% (OR 0.54, p = 0.31).

**Novel-V sub-claim:**
* With the dominant paper: 28.1% vs 72.7% (OR 0.136, p = 0.001), as reported.
* Without it: 57% vs 73.5% (p = 0.04).

## 5. Pro115 (`05_pro115_checks.py`)

**Filter bug.** The regex `'D|E_FR4'` also matches F_CDR1 and B_CDR2FR3, because "CDR" contains a D.
* So the "no CDR3/FR4 error" set of 24 is really 21 FR1 (mode A) chains, 2 FR3 chains and 1 "other" chain.
* The simple fix `Dplus|Dminus|E_FR4`, used by the confounding verifier, is also wrong. It misses `D0` runs at IMGT
  109–117 and a CDR1 run that spans IMGT 31–127, both of which cover Pro115.

**Position-based filter** (error chains with no run reaching IMGT ≥104):

| Set | Error | Control | OR | p |
|---|---|---|---|---|
| All | 4/28 | 62% | 0.096 | 0.012 |
| Paper level | 2/12 papers | | | 0.002 |
| Dominant paper dropped | 4/16 | | | 0.056 |

**Mostly not "elsewhere" in 3D.**
* 22 of the 28 chains are FR1 N-terminal errors.
* In 25 correctly built reference κ chains, those runs (starting at IMGT 1–3) are 7.3–10 Å (CA–CA) from IMGT 115.
* Non-A errors alone: 1/6 cis (p = 0.10).

**Entry adjustment:** adjusted for entry all-trans elsewhere, OR 0.15, p = 0.067 (investigator's set).

## Multiple testing

BH across this verifier's 68 tests (`all_tests_bh.csv`):
* the published-style Cκ p = 0.0013 gives q = 0.018;
* every test with the dominant paper dropped has q ≥ 0.057.

## Files

| Kind | Files |
|---|---|
| Scripts | `01_rebuild.py`, `02_reproduce.py`, `03_omega.py`, `04_robust.py`, `05_pro115_checks.py`, `06_sites_novel_bh.py` |
| Tables | `frame.csv.gz`, `frame_em.csv.gz`, `03_omega.csv`, `04_robust.csv`, `04_lopo_*.csv`, `05_pro115_tests.csv`, `05_pro115_omega.csv`, `05_run_to_pro115_distance.csv`, `06_sites_novel.csv`, `all_tests_bh.csv`, `*.log` |
