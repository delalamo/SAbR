# Verification (confounding lens): struct_unusual / beyond_site_backbone

**Verdict: weakened.** The core association holds. Parts of the claim, and its framing, do not.

## What holds

The Ramachandran 'allowed' and clash excess outside the error site survives every confounder adjustment I tried.

**Reproduction.** I reproduced the original numbers exactly: +3.75 'allowed' per 100 residues and +6.25 clashes
(`v01`). The effect is unchanged when I:
* cluster standard errors by paper or by last author (196 clusters);
* aggregate to one row per paper, giving +2.85 (p 2e-5), or one row per lab, giving +3.54 (p 4e-6);
* model deposition date as a continuous quadratic and resolution as bands;
* add SARS-CoV-2 status;
* add refinement-software flags;
* add starting model and fitting protocol;
* add auxiliary-reagent role and journal tier;
* add internal gaps, unbuilt residues, a late N-terminus and side-chain truncation;
* add local Q-score;
* add the geometry of the non-V part of the same entry;
* put all of the above in one model, giving +3.60 (p 1e-5);
* restrict to 2020–22, to non-SARS-CoV-2 entries, or to Phenix-refined entries.

Leaving out any one last author leaves the effect between 3.8 and 4.6.

**Error-definition and gap checks (`v02`).** I excluded every ANARCI/SAbR-disagreeing residue ±3, every other run
±3, and residues within 3 of a chain break or a V-region end. The effect stays at +4.3 (p 3e-6). Framework residues
only give +4.2. Excluding everything within 12 Å of the site in 3-D gives +3.8 (p 3e-4).

**Verified D-mode slips.** Among cases with verified CDR3/FR4 slips (origin 5) the effect is +3.3 (p 0.001), so it is
not limited to possible detector artifacts.

**Independent replication (`v03`).** I recomputed everything with my own code, straight from the validation XML and the
SAbR records:
* *New cryo-EM control set:* 329 chains from 307 entries. None of the investigator's entries are reused, and the entry
  is not required to be clean. 'Allowed' is +4.5 (p 1e-8; controls 5.6%, as low as the original controls) and
  clashes +5.9 (p 6e-4).
* *X-ray (not analysed by the investigator):* 35 case units against 140 controls. 'Allowed' is +3.7 (p 4e-5) and
  clashes +8.8 (p 0.006).

## What weakens it

1. **Rotamer outliers are not robust.** They are not significant with:
   * the independent EM controls (p 0.087);
   * X-ray (p 0.19);
   * paper-level aggregation (p 0.09) or lab-level aggregation (p 0.22);
   * non-SARS-CoV-2 entries only (p 0.10);
   * the all-covariates model (p 0.07).
2. **The excess is not specific to the error chain (`v04`).** I compared chains within the same entry, which fixes the
   map, lab, software, refinement run, year and resolution. Against error-free V chains of the same entry, the error
   chain's beyond-site 'allowed' rate is only +0.8 (95% CI −0.5 to 2.2, p 0.23; 106 entries). Clashes are +0.7
   (p 0.34). This holds for the error-free partner (+0.9) and for a different antibody in the same entry (+0.6).
   Error-free V chains of case entries are themselves close to the controls: +0.6 (p 0.30).
3. **The excess is concentrated in a subset (`v05`).**

   | Group | 'Allowed', % | Excess over controls |
   |---|---|---|
   | Entries where every V chain has an error (59 cases) | 13.7 | +8.2 pp |
   | … of which single Fabs with both H and L slipped (33 cases) | 16.3 | |
   | Entries that also hold error-free V chains (111 cases) | 8.2 | +2.4 pp (p 0.002) |
   | Controls | 5.8 | |

   Dropping the 40 most extreme cases (23%) removes the effect: +0.6 (p 0.17). That said, 34% of cases are above the
   controls' 90th percentile, against 9% of controls.

   In those all-error entries the non-antibody part has normal geometry (5.3% vs 5.4% 'allowed'). So this is
   antibody-specific poor modelling at the level of the entry, which produces both the slips and the poor geometry. It
   is not a property of the slipped chain.
4. **The excess is not uniform across the domain.** It is about twice as large within 12 Å of the site in 3-D (+5.0 pp)
   as at 12–20 Å or beyond (+2.6 to +3.6 pp; ≥20 Å +2.9, p 0.02). It is still present far away, but it is not flat.

## Nulls
* Partner versus error chain, and different antibody versus error chain, within the same entry: 'allowed', clashes,
  rotamer outliers and Q-score.
* Error-free V chains of case entries versus controls: 'allowed' and clashes. The partner's clashes are the exception
  (+4.8, p 0.02).
* Non-V geometry in all-error and mixed entries versus controls.
* Rotamers in the replication sets.

## Files
`v01_reanalysis.py`, `v02_strict_outside.py`, `v03_independent_controls.py`, `v04_within_entry.py`, `v05_decompose.py`,
`v06_summary.py`. Tables are in `tables/`; `v06_key_tests.csv` holds the key tests with BH q over 81 tests.
