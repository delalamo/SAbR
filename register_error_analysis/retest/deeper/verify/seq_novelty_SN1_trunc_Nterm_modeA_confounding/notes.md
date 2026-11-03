# Verification of SN1 (mode A: SEQRES starts after IMGT 1), confounding lens

**Verdict: weakened.**
* **It survives the confounders.** The association reproduces exactly and survives:
  * period, resolution, chain type, SARS-CoV-2 and species/nanobody;
  * paper and lab clustering, including within-lab and within-paper comparisons.
* **It is not separable from the known "N-terminus not built" factor.**
  * The SEQRES truncation is nested inside "the built model starts after IMGT 1".
  * Adjusting for the first built residue leaves OR 1.2 (0.6-2.3).
  * Adjusting for the structural start leaves OR 3.8, so the independent effect lies somewhere in 1.2-3.8.
* **λ chains** show no effect.
* **It is a marker, not a strong independent link.** See "Second pass" below.

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

## Second pass (`v10_independent.py`, `v10_independent.log`, `v10_tests.csv`, `v10_case_entity_poly.csv`)
* **The deposited record really is truncated.** RCSB `entity_poly` for all 19 cases starts exactly at the V-domain
  start reported by ANARCI. None has a tag or leader, and none has a modified N-terminal residue (e.g. pyroglutamate),
  so this is not a parsing artifact. An independent truncation flag gives the same result: SEQRES index of the first
  built residue equal to the V start, from SAbR `seq_index`. M2 OR 5.59 (3.00-10.4).
* **Lab and paper do not explain it.**
  * Within-lab conditional logit (23 labs with a case): OR 5.33 (2.27-12.5).
  * Within-paper conditional logit (21 papers): OR 4.76 (1.80-12.6).
  * One row per paper: OR 5.10 (2.79-9.32).
  * There *is* a lab-habit component: clean units from labs that have a mode-A case are truncated 12.2% vs 6.6% (OR
    1.95). But the unit-level association holds within labs.
* **X-ray** points the same way but is underpowered: 1/7 vs 3.1%, OR 5.3, p 0.20.
* **The known factor as previously coded does not absorb it.** The deepdive flag (unbuilt residues counted on the
  SEQRES numbering, or first built residue > 3) leaves OR 5.77. A record that starts at IMGT 2-3 is invisible to that
  flag, because nothing in it is "unbuilt". So SN1 does fill a hole in the earlier coding.
* **Adjusting for which residue the model starts with (S) removes it.** OR 1.17 (0.60-2.28), or 1.17 (0.51-2.66)
  lab-clustered. Within S = 2-9, a truncated record vs an unbuilt-but-recorded N-terminus gives OR 1.83 (0.95-3.53),
  p 0.07.
* **But S is partly downstream of the error, so treat that 1.17 as a lower bound.** In 13 of the 30 full-SEQRES
  cases with S > 1, the shifted residue occupies structural slot 1 (or 0). There, "residue 1 unbuilt" may be a
  consequence of the shift rather than its cause. Two checks:
  * Adjusting for the structural start (SAbR) instead gives OR 3.75 (1.76-7.96).
  * Dropping those 13 cases gives OR 2.10 (0.98-4.48).

  So the record-level effect independent of "N-terminal residues absent from the model" lies somewhere between about
  1.2 and 3.8, and these data cannot pin it down.

## Revised interpretation
* **Weakened, not refuted.** The association is real, and none of the following explains it:
  * deposition year, method, resolution, chain type, SARS-CoV-2 or species;
  * lab or paper clustering;
  * a detector artifact.
* **Not independent of the known factor.** It is a sub-case of "the model lacks the first V residues", and its
  independent contribution beyond that is not established (OR 1.2-3.8 depending on how model start is measured).
* **Chain type and the "strong" rating.**
  * It is confined to H and κ chains. For λ the rate is 27% vs 21%, no difference.
  * The "strong, separate link" rating should be downgraded to "moderate marker of the unbuilt N-terminus".

## Files
* Scripts: `v01_cases.py`, `v02_chainfeat.py`, `v03_strata.py`, `v04_insilico_trunc.py`, `v05_anarci_trunc.py`,
  `v06_controls.py`, `v07_finer.py`, `v08_geometry.py`, `v09_summary.py`, `v10_independent.py`.
* Tables:
  * `chain_nterm.csv.gz`, `units_modeA_analysis.csv.gz`;
  * `v03_models.csv`, `v06_models.csv`, `v09_all_tests_bh.csv` (BH over 38 tests);
  * `v04_insilico_sabr.csv`, `v05_anarci_trunc_controls.csv`, `v05_case_addback.csv`, `v08_geometry_cases.csv`;
  * logs `v03_strata.log`, `v06_controls.log`, `v07_finer.log`, `v10_independent.log`;
  * `v10_tests.csv` (BH within v10), `v10_case_entity_poly.csv`.
