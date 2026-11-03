# Are the erroneous antibodies structurally unusual? (lens: struct_unusual)

**Short answer:** mostly no. The antibodies themselves are ordinary, but their models were built less carefully.
The VH-VL orientation implied by the sequence, the CDR-H3 length, the predicted H3 base, the predicted domain cores
and the canonical cis-Pro in CDR-L3 all look like the controls'. What differs is the deposited antibody model:

* worse backbone geometry across the whole V domain, not only at the slip;
* the same, to a lesser degree, in the error-free partner chain;
* larger departures from ImmuneBuilder predictions and from canonical CDR conformations;
* the canonical κ L3 cis-Pro95 left out;
* domain cores that are distorted.

The rest of the same model (antigen, constant domains) is as good as in the controls.

There is one modest intrinsic signal. ImmuneBuilder's ensemble is less certain at the error sites. Among nanobodies,
error cases more often have a predicted extended (non-kinked) H3 base.

## Sample and methods

### Sample (`s01`)
* **Cases:** 183 cryo-EM first-occurrence error units (VH 78, κ 43, λ 24, VHH 38), one representative chain each,
  from 159 entries.
* **Controls:** 361 units from 185 fully clean EM entries, with no disagreement run in any chain. Matched on chain class
  (VH / VHH / κ / λ), deposition-year bin and resolution band, 1–3 per case (mostly 2).
* **Balance:** median resolution 3.50 vs 3.47 Å; median year 2023 in both; SARS-CoV-2 20% vs 24%.
* **Total:** 344 structures.

### Metrics
**Numbering.** All coordinate metrics use SAbR (structure-based) IMGT positions, so a register slip does not move a
residue's apparent location.

**Statistics.** Each test adjusts for chain class, year bin and resolution, with cluster-robust SEs by entry.
Benjamini–Hochberg q is computed over 95 tests (`tables/tests.csv`). Sensitivity checks are in
`tables/sensitivity*.csv`:
* adding SARS-CoV-2 as a covariate;
* restricting to 2020–22;
* matched-set Wilcoxon;
* adding local map fit as a covariate.

**(a) VH–VL orientation.** ABangle in the published Dunbar 2013 frame, using the consensus cores and principal
components from the ABangle repository. Validation: 7 of 9 published Fvs reproduce *exactly*, and the other 2 within
0.9° (`tables/abangle_validation.csv`). Each Fv is compared with the 1,265 X-ray Fvs in `All_Angles.dat`.

**(b) CDR-H3 base and canonical features:**
* CDR-H3 base pseudo-torsion over Cα 115-116-117-118 (Kabat 100x–103), classed as kinked when −30° ≤ τ ≤ 100°.
* CDR lengths.
* The canonical κ CDR-L3 cis-Pro95.

**(c) Geometry.** From wwPDB validation XML: Ramachandran, rotamer, clashes, bond and angle outliers. Computed from
coordinates: cis peptides, Cα–Cα distances and side-chain truncation. Each is measured at the site (run ±2) and outside
it (excluding run ±3). The same metrics are taken for the error-free partner and for the non-antibody rest of the entry.

**(d) Map fit.** Q-score (`local_all.jsonl`).

**(e) Comparison with ImmuneBuilder.** ImmuneBuilder was run with the same protocol as the case predictions (model 0,
unrefined) on 120 control Fvs and 45 control VHHs, plus 60 case Fvs and the case VHHs to obtain ensemble error.
Measures:
* framework RMSD, excluding the site;
* CDR RMSD after superposition on the framework;
* ABEGO backbone-bin agreement for the non-H3 CDRs;
* predicted H3 base;
* predicted orientation.

## Results (errors vs controls)

| Question | Errors | Controls | Effect (adjusted) | q |
|---|---|---|---|---|
| **(c) Beyond the site:** Ramachandran 'allowed', per 100 V residues outside the site | mean 9.5 (median 6.7) | 5.7 (4.5) | +3.8 (2.3–5.2) | 2e-6 |
| … frameworks only | 8.3 | 4.4 | +4.0 | p 4e-8 |
| … by distance from the site (run / 1–3 / 4–10 / 11–20 / >20 residues), % non-favoured | 15 / 20 / 9.6 / 10.4 / 9.8 | 6.4 / 6.5 / 5.3 / 6.7 / 6.2 | | |
| Clash residues per 100, outside the site | 24.8 | 17.7 | +6.2 (2.8–9.7) | 0.0015 |
| Rotamer outliers per 100, outside the site | 1.33 | 0.64 | +0.67 | 0.02 |
| Ramachandran *outliers*, cis non-Pro and Cα–Cα outliers outside the site | | | n.s. | |
| At the site: Ramachandran outlier / Cα–Cα outlier / rotamer outlier | 13.7% / 4.9% / 15.3% | 6.1% / 1.1% / 7.5% | OR 2.6 / 4.8 / 2.3 | 0.01–0.03 |
| **Error-free partner chain:** clash residues per 100 | 23.2 | 17.5 | +5.2 | 0.03 |
| Partner: Ramachandran 'allowed' per 100 | 10.5 | 5.9 | +1.2 | p 0.065 (weak) |
| **Within the same entry:** V-domain minus non-V Ramachandran 'allowed' (pp) | +3.65 | +0.21 | +3.5 (2.5–4.4) | 9e-11 |
| Within the same entry: V minus non-V clash (pp) | +5.2 | +1.6 | +3.6 | 0.001 |
| Non-V residues (antigen etc.): Ramachandran, rotamer, clash, Q | | | all n.s. | |
| **(d) Map fit:** site Q minus rest-of-domain Q | −0.018 | −0.002 | −0.017 | 0.005 |
| V-domain Q, entry Q, entry atom inclusion, V-minus-rest Q gap | | | n.s. | |
| **(a) Orientation:** mean HL/HC1/HC2/LC1/LC2/dc | | | all n.s. | |
| Mahalanobis distance from the X-ray reference | 3.36 | 2.57 | +1.24 | 1e-4 |
| Outside the 99% reference ellipsoid | 36% | 16% | OR 2.7 | 0.004 |
| Core-set Cα RMSD to the ABangle consensus | 0.84 Å | 0.58 Å | +0.35 | <1e-6 |
| Mahalanobis distance, Fvs with undistorted cores only | 2.75 (n 54) | 2.54 (n 158) | +0.10 | 0.67 (n.s.) |
| *Predicted* (sequence-intrinsic) orientation, Mahalanobis | 1.60 | 1.41 | +0.13 | 0.24 (n.s.) |
| *Predicted* core RMSD | 0.36 | 0.34 | +0.02 | 0.11 (n.s.) |
| Deposited minus predicted, \|ΔLC2\| / \|Δdc\| | 1.80° / 0.65 Å | 1.06° / 0.39 Å | +1.3° / +0.18 | <1e-4 / 0.01 |
| **(b) H3 base kinked (deposited):** all | 59% | 89% | OR 0.18 | <1e-5 |
| … CDR3/FR4 (D-mode) errors | 48% | 89% | OR 0.11 | <1e-5 |
| … VH with errors outside CDR3/FR4 | 89% | 93% | OR 0.60 | 0.52 (n.s.) |
| H3 base kinked, *predicted*, VH | 97% | 98% | | 0.54 (n.s.) |
| … VH D-mode errors: predicted 92% kinked vs deposited 49% (n 37); controls 98% vs 95% (n 120) | | | | |
| H3 base kinked, *predicted*, VHH | 50% (n 34) | 70% (n 43) | OR 0.31 | 0.078 (weak) |
| CDR-H3 length | 15 | 16 | | 0.95 (n.s.) |
| Any non-H3 CDR length with frequency <2% (overlaps the known atypical-length finding) | 9.8% | 4.2% | OR 2.3 | 0.069 |
| **κ 9-residue L3 with Pro95: canonical cis peptide modelled** | 23% (6/26) | 76% (48/63) | OR 0.08 (0.02–0.29) | 7e-4 |
| … cases whose error is not in CDR3/FR4 | 11% (1/9) | 76% | OR 0.02 | 0.028 |
| ImmuneBuilder predicts cis for all 67 predicted chains in both groups | | | | |
| **(e) Framework Cα RMSD, deposited vs predicted** (site excluded) | 1.02 Å | 0.72 Å | +0.50 | <1e-6 |
| … error-free chains of case Fvs | 0.98 Å | 0.72 Å | +0.30 | 1e-4 |
| Non-H3 CDRs in the same backbone bins as predicted | 0.76 | 0.89 | −0.13 | <1e-6 |
| … error-free chains of case Fvs | 0.77 | 0.89 | −0.09 | 1e-5 |
| All non-H3 CDRs in the predicted bin pattern | 14% | 34% | OR 0.28 | <1e-4 |
| CDR1 / CDR2 / CDR3 RMSD after framework superposition | 1.5 / 1.3 / 2.7 | 0.9 / 0.9 / 1.6 | +0.8 / +0.75 / +1.0 | <1e-4 |
| ImmuneBuilder ensemble error at the site (controls: same IMGT positions) | 0.46 | 0.32 | +0.16 | 0.006 |
| Ensemble error, whole chain | 0.38 | 0.33 | +0.06 | 0.01 |
| Ensemble error, CDR3 | 0.73 | 0.63 | +0.11 | 0.07 (+0.16, p 0.03 adjusting for H3 length) |

## Robustness

The beyond-site geometry excess survives all four sensitivity checks:
* adding SARS-CoV-2;
* restricting to 2020–22;
* adding local Q;
* matched-set Wilcoxon (all p ≤ 0.001).

It also holds in every chain class and in both periods (`beyond_site_strata.csv`). The one exception is clashes in VHH
(+0.9, n.s.). Recorded starting-model type and fitting protocol do not explain it: the effect is +3.73 after adjusting
for both.

The site-level Ramachandran and Cα–Cα effects and the ensemble error at the site lose significance in 2020–22 alone.
That is fewer cases, with effects in the same direction.

## Interpretation

### Mostly ordinary antibodies, modelled poorly
* **Orientation.** The sequences imply ordinary VH–VL orientations and cores. The excess orientation outliers in the
  deposited models come entirely from distorted domain cores.
* **H3 base.** The non-kinked bases in D-mode VH errors are a product of the misbuilt CDR3/FR4 junction: ImmuneBuilder
  predicts kinked bases for 92% of them.
* **Canonical cis-Pro.** The κ L3 Pro95 is predicted cis for every sequence, but modelled trans in 77% of error chains,
  including chains whose error sits far from L3.
* **Whole-domain geometry.** Backbone geometry is non-ideal across the whole domain, frameworks included (8.3% vs 4.4%
  non-favoured). It is partly shared by the error-free partner chain, but not by the non-antibody rest of the same
  model, and not explained by poorer local density.

This points to the antibody part of these entries being built or refined with less restraint. Plausible routes are
docking a homology or template model with little refinement, or building without canonical-structure knowledge. It does
not point to the antibodies being structurally exotic.

### A modest intrinsic component
* ImmuneBuilder's ensemble is less certain where errors occur (0.46 vs 0.32 Å at matched positions).
* Nanobody error cases more often have a predicted extended H3 base (50% vs 70% kinked; weak).
* Rare CDR lengths are more frequent (weak; already known).

These fit the earlier "atypical antibody" findings, but they are small next to the modelling-quality signal.

## Caveats
* Controls come from entries with no disagreement run of any kind, which may select carefully built entries. The
  within-entry contrast (antibody vs rest of the same model) and the per-domain metrics are less exposed to this.
* "Beyond the site" excludes the run ±3 residues. Some geometry strain may still spread further from a slip: the
  non-favoured rate is 9.6% at 4–10 residues, but 9.8% even beyond 20 residues.
* The ImmuneBuilder comparison treats the prediction as canonical. Its own errors (CDR-H3) add noise equally to both
  groups.
* Ensemble errors for cases come from a random 60 of the 108 case Fvs, plus the case VHHs.
* The κ cis-Pro analysis has small n: 26 case chains, 9 away from the site.
* Software and protocol metadata are sparse (85% of cases have no starting model recorded).
* All cryo-EM, 3–5 Å. The X-ray ABangle reference differs in era and method: the EM controls themselves have a slightly
  larger dc (16.4 vs 16.2 Å).
