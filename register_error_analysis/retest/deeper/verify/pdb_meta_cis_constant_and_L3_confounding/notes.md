# Verification of `cis_constant_and_L3` (pdb_meta) — confounding lens

**Verdict: WEAKENED.**
* The **constant-domain** part of the claim is mostly the effect of one paper, which built its whole entries with no cis
  peptides at all. CH1 is null once that paper is removed.
* **Pro115** survives clustering, but it is the same "all-trans V-domain" chains as the Pro8 finding, not an
  independent site.

## Method

* **Scope:** EM chains only, amyloid fibrils excluded. Error = `err_first`; control = `~err_any`, as in the
  investigator's analysis.
* **Model:** logistic regression adjusted for deposition-year bin and resolution, with SEs clustered by paper. This is
  the investigator's `statlib.fit_feature`.
* **BH:** q is computed across all 110 verifier tests (`all_tests_bh.csv`).

## Results

### 1. Reproduction is exact

| Site | Error | Control | OR | p |
|---|---|---|---|---|
| Cκ | 6/29 | 737/912 | 0.039 | 0.0013 |
| CH1 | 13/31 | 848/1173 | 0.24 | 0.054 |
| All constant domains | 19/60 | 1723/2327 | 0.11 | 0.0019 |
| Pro115 | 3/24 | 1269/2045 | 0.079 | 0.026 |

### 2. One paper dominates

The paper is 10.1038/s41422-021-00487-9 (Zhou Q., SARS-CoV-2 S with P2B-1A1 and related Fabs, 12 entries
7czp–7d0d).

* **Share of the error chains:** 17/29 Cκ, 12/31 CH1, 29/60 pooled and 12/24 Pro115 error chains come from it.
* **Zero cis everywhere:** every entry of this paper has **zero cis peptides anywhere** (`cis_peptide_count` = 0).
  This includes the clean antibody chains (7czp, 7czy-H, 7czz-H, 7d0c, 7d03) and the spike.
* **Within this paper,** error and clean chains are identical: 0% cis.

Removing this paper (`02`, `06`):

| Site | Error | Control | OR | p | q |
|---|---|---|---|---|---|
| **CH1** | 13/19 = 68% | 73% | 0.75 | 0.63 | |
| | 11/17 papers | 294/399 papers | | 0.41 | |
| **Pooled constant** | 19/31 = 61% | 74.5% | 0.38 | 0.054 | 0.09 |
| **Cκ** | 6/12 = 50% (9 papers) | 81% | 0.14 | 0.025 | 0.06 |
| Cκ, paper level | 4/9 papers | 270/341 papers | | 0.026 | |
| **Novel-V pooled constant** | 57% | 74% | 0.35 | 0.036 | 0.07 |

The novel-V pooled constant result was 28% vs 73% with the paper included.

### 3. Within-entry and within-paper tests are null (low power)

| Comparison | Error vs control | OR | p | Informative strata |
|---|---|---|---|---|
| Pooled constant, within entry | 52% vs 68% | 0.62 | 0.55 | 7 entries |
| Pooled constant, within paper | | 0.39 | 0.25 | 7 papers |
| CH1, within entry | | | 0.92 | |

### 4. Entry-wide all-trans models explain the constant-domain deficit

Restricting to entries with at least one cis peptide anywhere:

| Site | Error | Control | p |
|---|---|---|---|
| Cκ | 6/8 = 75% | 91% | 0.23 |
| Pooled constant | 79% | 87% | 0.23 |

At entry level, "no cis peptide anywhere in the entry" is only modestly enriched:

| Set | Error entries | Clean entries | OR | p |
|---|---|---|---|---|
| All | 51.7% | 39.6% | 1.8 | 0.006 |
| Without the Zhou paper | | | 1.65 | 0.015 |

### 5. Period strata

| Stratum (Zhou paper removed) | Error | Control | p |
|---|---|---|---|
| 2020–21, Cκ | 4/6 | 83% | 0.38 |
| 2020–21, pooled | 7/10 | 81% | 0.40 |
| ≥2022, pooled | 11/19 | 70% | 0.19 |

There are too few error chains before 2020 to test.

### 6. Selection on whether the site is modelled

Error κ chains have the Cκ Tyr-Pro site modelled more often: **83% vs 45%** (driven by the Zhou paper).

If unmodelled sites are counted as not cis, Cκ cis is:

| Set | Error | Control | p |
|---|---|---|---|
| All | 17% | 36% | 0.12 |
| Without the Zhou paper | 33% | 36% | 0.57 |

### 7. Pro115 survives clustering

Clustering tests:

| Test | Error | Control | Effect | p | q |
|---|---|---|---|---|---|
| Zhou paper removed | 4/19 = 21% | 62% | OR 0.14 | 0.03 | |
| Paper level | 2/13 papers | 411/634 papers | | 0.0004 | 0.013 |
| Entries with any cis | 33% | 75% | | 0.024 | |
| L3 = 9 only | | | OR 0.093 | 0.013 | |
| L3 = 9, Zhou paper removed | | | | 0.07 | |

It is not explained by:
* **unbuilt residues:** all Pro115 sites are modelled;
* **L3 length or species:** OR 0.088 with both adjusted;
* **sub-threshold CDR-L3 disagreement:** 3/31 error chains have it; excluding them gives OR 0.096, p = 0.012
  (0.056 without the Zhou paper);
* **local resolution:** n.s., but only 8 error sites have EMDB-VA coverage.

### 8. Pro115 is not independent of Pro8

* **No cis in V:** 24/31 error chains have no cis peptide anywhere in the V domain.
* **Pro8-trans stratum:** Pro115 cis is **10.7% vs 31.1%** (OR 0.40, p = 0.46). Clean chains with a trans Pro8 also
  mostly lack Pro115 cis.
* **Adjusted for Pro8 cis:** OR 0.28, p = 0.24.
* **Not significant in ≥2022:** 30% vs 60%, p = 0.18.
* **Caveat:** in mode-A chains the Pro8 status may itself be a consequence of the FR1 error, so this adjustment may
  over-adjust.

### 9. Bug in the investigator's mode filter (not material)

* The regex `'D|E_FR4'` matches any mode containing "CDR" (F_CDR1, B_CDR2FR3).
* So the "no CDR3/FR4 error" set also excluded CDR1 and CDR2/FR3 error chains: 21 of the 24 remaining chains are
  mode A.
* The corrected set gives 4/31 vs 62%, OR 0.084, p = 0.007.

### 10. Adjustments that do not change the chain-level ORs

Adding SARS-CoV-2, chain Q-score or starting-model type leaves the ORs unchanged:

| Site | OR with all three added | p |
|---|---|---|
| Cκ | 0.053 | 0.001 |
| CH1 | 0.30 | 0.02 |
| Pooled | 0.17 | 0.0003 |

These adjustments do not address clustering.

## Files

| Kind | Files |
|---|---|
| Scripts and logs | `01_reproduce.py` … `06_summary_bh.py`, each with a `.log` |
| Tables | `02_clustering_results.csv`, `03_canonical_sites_tests.csv`, `04_period_novel_localres_tests.csv`, `04_local_resolution_at_sites.csv`, `05_L3_local_disagreement.csv`, `06_summary_tests.csv`, `all_tests_bh.csv` |
| Frames | `frame.csv.gz`, `03_canonical_sites.csv.gz` |
