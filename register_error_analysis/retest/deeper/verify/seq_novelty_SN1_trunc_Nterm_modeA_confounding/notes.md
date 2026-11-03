# Verification of SN1 (mode A: SEQRES starts after IMGT 1), confounding lens

**Verdict: weakened.** The association reproduces exactly and survives period, resolution, chain type, SARS-CoV-2,
species/nanobody and paper/lab clustering. But it adds nothing beyond the already-known "N-terminus not built" factor.
SEQRES truncation is nested inside "the built model starts after IMGT 1", and once the model's first built IMGT
position is adjusted for, the record-level effect disappears. It is not an independent link.

## Reproduction (`v02_chainfeat.py`, `v03_strata.py`)
* 19/59 (32.2%) cryo-EM mode-A error units vs 7.8% of 4,036 clean EM units. This is identical to the investigator's
  figure, and the error chain itself gives the same 19 as the earliest-chain representative.
* M2 OR 5.32 (2.89-9.82).
* Cluster-robust by paper: OR 5.32 (2.68-10.6), p 2e-6. By last author: (2.50-11.3), p 1.5e-5.
* Broader controls (every EM unit without a mode-A first-occurrence error): OR 5.32 (`v06`).
* Strata, crude Fisher:
  * deposition 2022-23: 26% vs 7.0%; 2024+: 45% vs 8.1%. ≤2019 and 2020-21 point the same way (OR 5.7 and 3.2) but
    are not significant (n = 6 and 14).
  * H: 39% vs 7.2%; κ: 20% vs 4.1%; **λ: 27% vs 21%, no difference** (λ controls are often truncated).
  * Both resolution halves, SARS-CoV-2 and non-SARS-CoV-2, human/mouse non-nanobody, and first structure of the
    antibody: all OR 4.8-8.6.

## The simpler known factor explains it
* **Model start.** The first *built* residue sits after IMGT 1 in 83% of error units vs 28% of controls (OR 12.9).
* **Record matches the model.** A SEQRES cannot start before the model. In 95% of truncated cases and 90% of truncated
  controls, the SEQRES starts exactly at the first modelled residue. The truncated record *is* the unbuilt N-terminus,
  entered as modelled residues only.
* **Adjusting for model start:**

  | Model start adjustment | OR for SEQRES truncation | 95% CI | p |
  |---|---|---|---|
  | 1 / 2 / 3 / 4+ | 1.17 | 0.60-2.28 | 0.65 |
  | Paper-clustered | 1.17 | 0.55-2.48 | |
  | Finer bins | 1.78 | 0.90-3.52 | 0.10 |
  | Mantel-Haenszel across start strata | 1.39 | 0.76-2.57 | 0.27 |

* **Same model start, different record.** Among units whose model starts after IMGT 1, a truncated SEQRES vs a full
  SEQRES with an unbuilt N-terminus gives OR 1.01 (0.46-2.20).
* **Linear adjustment is misleading.** Adjusting for model start as a linear IMGT number leaves OR 4.0. That
  specification is wrong: risk is non-monotone (start 2 = 1.6%, 3 = 8%, 4 = 17%, 7-9 = 26%, ≥10 = 0%, since a mode-A
  run must start at IMGT ≤10).
* **Possible small residual.** At model start 2-3 there is a residual OR of 2.3 (1.06-5.1, p 0.036, BH q 0.06). At
  start ≥4 the effect reverses (OR 0.26). Treat this as unconfirmed.

## The error definition does not explain it (both detector arms checked)
* **ANARCI arm (`v05`).** Adding the germline N-terminal residues back to the 19 truncated SEQRES changes 0 ANARCI
  numbers in all 19, so the "starts at IMGT 2-8" call is germline-consistent. Separately, removing 1-2 residues from
  150 complete control SEQRES changed ≥3 numbers in 0 chains; removing 3 changed them in 8-12%.
* **SAbR arm (`v04`).** 105 clean EM chains (45 H, 30 κ, 30 λ) had their first 1, 2 or 3 built residues deleted. SAbR
  produced 0/315 disagreements, so SAbR does not invent N-terminal shifts when the start is missing.
* **Geometry (`v08`).** Each chain was superposed on the framework core of the X-ray references 1N8Z, 12E8 and 1AQK.
  Run residues sat closer to the reference CA at their SAbR position than at their ANARCI position in 16 of 19
  truncated cases (1 favoured ANARCI, 2 were ambiguous), and in 30 of 40 non-truncated cases. So these are real
  out-of-register strands. Several of the non-truncated "ANARCI fits" cases are λ chains with core RMSD ~17 Å, i.e.
  non-V-like folds (9faa, 9fab, 9liw).

## Interpretation
SEQRES truncation is a marker of the unbuilt N-terminus (often depositors entering only the modelled residues). The
mode-A errors in these chains are genuine one- or two-residue slips of the first built strand, not detector artifacts.
The SEQRES side adds no independent information, so the finding should be merged into the existing "N-terminus not
built" result, not reported as a separate strong link. One of the 19, 22ge, is an FR1 insertion (QQLEQSGGGAEGG) that
ANARCI gaps at position 1, not a true truncation.

## Files
* Scripts: `v01_cases.py`, `v02_chainfeat.py`, `v03_strata.py`, `v04_insilico_trunc.py`, `v05_anarci_trunc.py`,
  `v06_controls.py`, `v07_finer.py`, `v08_geometry.py`, `v09_summary.py`.
* Tables:
  * `chain_nterm.csv.gz`, `units_modeA_analysis.csv.gz`;
  * `v03_models.csv`, `v06_models.csv`, `v09_all_tests_bh.csv` (BH over 38 tests);
  * `v04_insilico_sabr.csv`, `v05_anarci_trunc_controls.csv`, `v05_case_addback.csv`, `v08_geometry_cases.csv`;
  * logs `v03_strata.log`, `v06_controls.log`, `v07_finer.log`.
