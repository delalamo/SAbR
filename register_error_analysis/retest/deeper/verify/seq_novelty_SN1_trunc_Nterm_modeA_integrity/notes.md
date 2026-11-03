# Verification of SN1_trunc_Nterm_modeA (lens: data integrity and statistics)

**Verdict: weakened.** The association reproduces exactly and is not a numbering or detector artifact. But it is
almost entirely a sub-case of the already-known "N-terminus not modelled" link. It adds little independent
information.

## Reproduced
| Check | Result |
|---|---|
| Raw | 19/59 mode-A EM units vs 313/4036 clean EM units (32.2% vs 7.8%), Fisher p 7.7e-8 |
| M1 / M2 / M3 | OR 5.81 / **5.32 (2.89–9.82), p 8.5e-8** / 7.51 (n = 40 vs 2,317). Same as the investigator's |
| All first errors | 16.4% vs 7.8% |
| Germline-alignment truncation flag (`v06`) | Independent of ANARCI's HMM, and agrees with the ANARCI flag in 99.6% of units. M2 OR 5.47 (2.97–10.07) |
| "Starts at IMGT 2–8" definition | Control 6.4%. OR 7.0, so the investigator's 7.8% control is conservative |
| Leave-one-lab-out / leave-one-paper-out | OR 4.7–6.2, max p 1.5e-6 |
| Periods | ≤2021: 25% vs 7.9% (p 0.02). ≥2022: 36% vs 7.7% |
| Non-SARS-CoV-2 | 30% vs 7.8% |
| Alternative control (all EM units without mode A) | OR 5.33 |
| Cases are real shifts (`v05`) | ImmuneBuilder prediction superposed on the framework core: in **13/14** analysable truncated cases (28/33 non-truncated), the N-terminal run residues sit nearer the predicted Cα at SAbR's shifted position than at the sequence position |

## What weakens it
* **It overlaps a known finding.** A truncated SEQRES forces an unmodelled N-terminus. So the feature is a subset of
  the deepdive's mode-A marker "first built residue after IMGT 1".
  * Unmodelled N-terminus (any cause): M2 OR 12.8 (6.4–25.9).
  * Adding truncation on top: OR **1.72 (0.93–3.20), p 0.08**.
  * Within unmodelled N-termini, matched on the first modelled IMGT position: OR **1.47 (0.74–2.94), p 0.27**. CMH OR
    1.13, p 0.66.
  * With the claim's "IMGT 2–8" definition (SEQRES starting at IMGT 1–8, first modelled at IMGT 2–8; `v04b`): M2 plus
    first-modelled position gives OR **2.44 (1.22–4.90), p 0.012**. CMH OR 1.68, p 0.09.
  * So the increment beyond an unmodelled N-terminus is OR about 1.5–2.4, with p 0.01–0.27 depending on the
    definition. The claimed OR is 5.3.
  * Rates: N-terminus built 0.35% (10/2,887). Residues in SEQRES but unmodelled 3.5% (30/865). Residues absent from
    SEQRES 5.7% (19/332).
* **Same mechanism with or without the residues in SEQRES.** In both groups the shifted model threads the first
  residue to IMGT ≤1 about equally often: 37% (truncated) vs 43% (present but unmodelled), p 0.77.
* **Chain types.** No effect in λ chains (27% vs 21%, p 0.71). The signal comes from heavy chains (39% vs 7.2%) and κ
  chains (20% vs 4.1%, 3 cases).
* **Minor counting errors.**
  * The truncated cases come from 19 papers, not 18 (18 labs is correct).
  * The control "truncated" rate includes sequences starting at IMGT 9–27, which does not match the "2–8" wording.

## No difference
* Truncation conditional on the unmodelled N-terminus.
* Threading-to-start behaviour (truncated vs present but unmodelled).
* λ chains.

## Files
* `v01_reproduce.py`: counts.
* `v02b_case_detail.py`: per-case numbering. It writes `v02_modeA_case_detail.csv`.
* `v04b_conditional_28.py`: the "IMGT 2–8" conditional test.
* **Files from an earlier, committed run of this verification** (`v01_build.py`, `v02_cases.py`, `v03_tests.*`,
  `v04_conditional.*`, `v05_m3.*`, `v06_templates.*`, `modeA_case_nterm_labels.csv`) are left as they were.
  * They agree with this run.
  * Lab×V deduplicated M2 OR 6.7.
  * M2 plus first-built category: OR 1.33 (p 0.42) for any truncation, and 2.32 (p 0.012) for truncation at IMGT 2–8.
  * M3 plus first-built category: OR 2.03 (p 0.09).
* `v03_first_modelled.py`: first modelled IMGT per chain.
* `v04_models.py`: M1–M3, conditional models, LOO, strata.
* `v05_struct_check.py`: prediction superposition.
* `v06_anarci_free_trunc.py`: germline-alignment flag.
* `v07_mechanism_and_bh.py`: mechanism comparison and BH.
* Outputs: `v0*_*.csv`, and `v07_all_tests_bh.csv` with BH q across this verification's tests.
