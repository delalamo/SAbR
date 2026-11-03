# Verification: antigen_competition / "capsid_target" (data-integrity and statistics check)

**Verdict: weakened.**
* The headline numbers reproduce exactly.
* About 40% of the capsid error entries are numbering artefacts, and the claimed breadth across labs is overstated.
* Once the artefacts are removed, an excess remains (OR about 2.9), but two groups carry it: Hafenstein and Zhu P.

## 1. Reproduction (`v02_verify.py`)

* **Case and control sets**, rebuilt from `entries.csv`:
  * error: `err_first`;
  * control: no run of any kind;
  * excluded: the 15 amyloid entries.
  * The rebuilt sets match the investigator's: 3,994 entries, 212 errors; cryo-EM 2,752 and 178.
* **Entry level:** capsid 20.8% (32/154) vs 5.6% (146/2598). OR 4.72 (2.53–8.82), p = 1.1e-6. Identical to the claim.
* **Paper level:** 29.0% (18/62) vs 8.7% (109/1249). Adjusted OR 4.39, p = 1.9e-6. Fisher p = 8e-6.
* **X-ray:** 0/37 (the claim says 0/38).
* **Join and exclusion checks:**
  * no join errors were found;
  * the exclusion of copied-only or artefact-only entries does not favour capsids (1/155 capsid entries vs 63/2676 others).

## 2. Breadth: "12 labs" is really about 7 groups

Citation and audit author lists (`entry_aux.csv.gz`, `v03_consortia.py`) collapse the 13 last-author strings:

| Group | Last-author strings | Capsid error entries | Error papers |
|---|---|---|---|
| Hafenstein | "Hafenstein, S.", "Hafenstein, S.L.", Charnesky (a Hafenstein-lab paper) | 7 | 6 |
| Wang Xiangxi lab, IBP | Wang X., Feng R., Fu W. (Wang X. is a co-author on all three) | 7 | 3 |
| Lou Z. / Li K. | "Kun, L." = Li K.; Lou Z. is a co-author | 7 | 3 |
| Zhu P. | Zhu P. | 6 | 1 |
| Xiamen | Li S.W., Cheng T. | 3 | 3 |
| Hogle | Hogle | 1 | 1 |
| McKenna | McKenna | 1 | 1 |

* The investigator's leave-one-lab-out split Hafenstein into two "labs".
* **Dropping curated groups (original errors):**
  * the two groups with most errors (Hafenstein and Wang X.): OR 3.30;
  * the top three: OR 2.08, p = 0.13.
* **Group-level comparison:** 7 of the 33 groups that published capsid antibody structures have any error, vs 89 of 631 other labs. Fisher p = 0.30; adjusted OR 1.54, p = 0.37.

## 3. Numbering artefacts are concentrated in capsids

13 of the 32 capsid error entries rest *only* on runs that are not register slips. Among the 146 non-capsid error entries, only 3 do.

* **Lou Z. lab, FMDV (7 entries):**
  * these are **bovine** antibodies (*Bos taurus*), numbered with human or mouse HMMs (v_id about 0.65);
  * the runs are D0 runs inside ultralong or truncated CDR3s with no FR4 (7d3l: CDR3 of 50; 7d3m: 61);
  * other runs have shifts of 11–15 positions (7d3k H, 8grr);
  * SAbR numbers run past 128 in some of them.
* **Wang X. lab, echovirus 3 (6 entries; Fabs 6D10 ×4 and 5G3 ×2):**
  * both heavy chains carry a one-residue deletion at the FR1/CDR1 junction ("CQAFYTF…");
  * ANARCI places the gap at IMGT 26, while SAbR places it inside CDR1;
  * the frameworks agree on both sides, delta is NaN, and the run is 5 residues.
  * This is gap-placement ambiguity, not a register shift.
  * These entries are 6 of the "9/14 F_CDR1" entries the claim cites as the capsid signature.
* **Other D0 runs in capsids:**
  * 7c80 L has a sequence that lacks Phe118;
  * 7c80 H has a SAbR misalignment, with numbers up to 142;
  * 8w5d and 8w5e are truncated FR4 chain ends.
* **The project's artefact filter missed these.** It flagged 63 non-capsid entries but only 1 capsid entry.

## 4. With the artefact-prone runs removed

Entries whose errors rest only on D0, within-CDR1 gap placement or bovine runs are dropped (`R4`–`R7`, `v04_extra.py`).

| Test | Capsid | Other | OR (95% CI) | p |
|---|---|---|---|---|
| Entry level | 13.7% (19/139) | 5.5% | **2.88** (1.46–5.70) | 0.0024 |
| Entry level, SE clustered by consortium | | | 2.88 | 0.014 |
| Paper level | 23.2% (13/56) | 8.6% | **3.29** | 7e-4 |
| Units (antibody in paper) | 9.7% | 3.6% | 3.09 | 0.004 |
| Lighter filter (not every bovine run flagged) | | | 2.99 | 0.0016 |
| Drop Hafenstein | 9.4% | 5.5% | 1.95 | 0.15 |
| **Drop Hafenstein and Zhu P.** | **5.4% (6/112)** | **5.5%** | **1.06** | **0.89** |
| Same, units | | | 1.05 | 0.89 |
| Same, lighter filter | | | 1.21 | 0.67 |

**Class correction strengthens the effect.**
* 13 norovirus VP1 VLP and rotavirus VP4 entries had been misclassified as "viral surface glycoprotein".
* Correcting this adds Prasad's norovirus preprint (4/9 errors, all Dplus with delta 2).
* Robust OR 3.21 (p = 2e-4).
* Dropping Hafenstein and Zhu P. then leaves OR 1.59 (p = 0.26).
* Also dropping Prasad leaves OR 1.02.

**Species.**
* The robust capsid effect sits in source-organism mouse antibodies: OR 4.96, 11/69 vs 16/446, p = 0.013.
* In non-mouse antibodies it is OR 1.96 (p = 0.12).
* The claimed "56% mouse" counts the bovine antibodies as mouse, because of ANARCI's species call. By source organism, it is 49% vs 17%.

## 5. Other parts of the claim

**Within-capsid lq_z (−0.11 vs +0.21, p = 0.007).**
* This reproduces, but it is not specific to capsids:
  * non-capsid entries show the same gap (−0.26 vs +0.10);
  * the interaction term has p = 0.67;
  * among robust capsid errors the gap vanishes (+0.21 vs +0.20, p = 0.41).

**Period (robust errors).** OR 3.6 for ≤2019, 3.7 for 2020–21 and 2.3 for 2022+. None of these is individually strong.

## Conclusion

* The investigator's statistic is computed correctly.
* Its interpretation, a broad target-class effect "across 12 labs" with a CDR1/D0 signature, does not survive:
  * the signature is mostly a numbering artefact from two groups;
  * the residual excess comes from Hafenstein (7 entries, 6 papers, 2016–2024) and one Zhu P. paper (6 of 15 entries).
* **Best reading:** a lab-concentrated excess among mouse anti-capsid Fabs, not a general capsid-target effect.

## Files

* **Scripts:**
  * `v01_extract.py`: per-entry author and organism table;
  * `v02_verify.py`: reproduction, artefact flags, paper level, species, period, lq_z;
  * `v03_consortia.py`: curated consortia, leave-one-consortium-out, units;
  * `v04_extra.py`: lighter filter, class correction, chain-level unit flag;
  * `v05_bh_all.py`: one BH correction over all 71 tests.
* **Tables:**
  * `all_tests_bh.csv`, `verify_tests.csv`, `consortium_tests.csv`, `extra_tests.csv`;
  * `capsid_error_consortia.csv`, `leave_one_consortium_out.csv`, `leave_one_paper_out_robust.csv`;
  * `em_entries_verify.csv.gz`, `entry_aux.csv.gz`;
  * the logs.
* **Superseded rows.** In `v02`, the automatic union-find of consortia over-merged on common surname+initial collisions (for example Wang X. and Li S.). Superseded:
  * R3, R8 and R11–R12 are replaced by the curated `v03` results;
  * the unit-level "robust" row (`units, robust errors`, entry-level flag; in `consortium_tests.csv` / `all_tests_bh.csv`) is replaced by the chain-level version in `v04`.
