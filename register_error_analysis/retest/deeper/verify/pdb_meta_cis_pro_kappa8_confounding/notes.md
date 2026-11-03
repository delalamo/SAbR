# Verification (confounding lens): pdb_meta finding `cis_pro_kappa8`

**Verdict: weakened.** The association is real and survives every confounder tested. But the headline magnitude is
inflated by chain-level clustering, and the claim "even when the error is far from FR1" is not supported in 3D.

## What reproduces

* The headline reproduces exactly: EM κ chains with Pro8, error 5/70 (7.1%) cis vs clean 1,891/2,975 (63.6%);
  OR 0.036 (0.012–0.11), p = 2e-9.
* **The annotation is correct.** I recomputed ω(7→8) from coordinates (`05_omega.py`, `omega_check.csv`) for all 70
  error chains and 160 random clean chains. Every chain annotated cis is cis, and every chain annotated non-cis is
  trans (except 3 distorted). This holds in both arms.
* The SAbR `seq_index` equals the 0-based SEQRES index for all 15,865 chains (`00_check_index.py`), so the mapping is
  sound.

## Confounders: the association survives all of them (`models_main.csv`, `models_more.csv`, `all_tests_bh.csv`)

**Adjustments to the chain-level OR** (stays 0.03–0.04):
* lab clustering;
* SARS-CoV-2 and species;
* resolution bins, starting-model type and fit protocol;
* local Q at Pro8;
* V-domain and chain Q;
* other V-domain cis peptides;
* software flags;
* IGKV family;
* all of the above together: 0.029.

**Strata**, each in the same direction:
* by period: ≤2019 0/4; 2020–21 1/37 vs 66%; ≥2022 4/29 vs 62% (OR 0.09);
* by resolution: <3.6 Å 0/42 (unit Fisher p = 3e-7); ≥3.6 Å 5/28 vs 67%;
* by SARS-CoV-2: non-SARS-CoV-2 OR 0.06; SARS-CoV-2 0/28;
* novel V regions only: OR 0.024;
* after dropping the 2 largest error papers: OR 0.066;
* GEE: OR 0.073;
* X-ray replication, tiny: 4/7 vs 86%, OR 0.17, p = 0.04.

**Within paper** (chain level, conditional logit): OR 0.078 (0.008–0.71), p = 0.024. Error chains are 8% cis vs 52%
for clean κ chains of the same papers.

## Where the claim is overstated

1. **Clustering inflates the effect.**
   * 27 of the 70 error chains come from two papers: Zhou Q. (17 chains) and Hafenstein (10). All of them are trans,
     and so are those papers' clean κ chains.
   * The counts are 31 paper clusters and 26 labs, not 26 papers and 22 labs.
   * One unit per antibody per paper: 5/38 (13%) vs 63%, OR 0.074.
   * Unit level, excluding FR1-mode errors: **5/24 (21%) vs 63%, OR 0.14 (0.05–0.37), p = 9e-5**.
   * Paper level: OR 0.26 (p = 0.01).
   * Within paper at unit level: OR 0.14 (0.016–1.23), **p = 0.076**, with only 6 informative papers.
2. **"Even when the error is far from FR1" does not hold in 3D** (`07_proximity.py`).
   * The non-FR1 error chains are dominated by CDR3/FR4 errors on the G strand (Dplus 19 chains / 4 units; E 7; D− 4).
   * The G strand pairs with the A′ strand directly after the cis-Pro8 kink.
   * In correctly built reference κ chains, E and D− runs sit 4.4–5.9 Å (CA–CA) from IMGT 6–12, and Dplus runs about
     9.4 Å away.
   * 34 of 38 error units have a run within 10 Å of the kink.
   * Only 4 units have errors more than 10 Å away, and 1 of those 4 is cis.
   * Errors only in CDR1, CDR2 or FR3 (no FR1, CDR3 or FR4 component): 2/4 cis vs 63% (OR 0.6, p = 0.6). There is no
     evidence for genuinely distant errors.
3. **FR1-mode errors are part of the definition.** 28 of the 70 chains (14 of 38 units) have a −1 shift at FR1, which
   moves Pro8 off the kink. Their lack of cis-Pro8 is close to definitional.
4. **The supporting evidence for a "whole-model" route is mostly one paper.**
   * Cκ cis deficit: 17 of the 29 error chains are Zhou Q. At unit level it is 5/13 (38%) vs 79%.
   * Entry marker (clean κ all-trans → heavy-chain error in the same entry) reproduces: 51.7% vs 36.9% all-trans, OR
     2.1, p = 0.016.
5. **Paper practice does not predict errors.** The cis rate of a paper's own clean κ chains does not predict the
   paper's errors (46.7% vs 60.5%, OR 0.45, p = 0.34).

## Nulls

* Local Q at Pro8, chain Q and V-domain Q do not explain the association.
* Starting model, fit protocol, software, IGKV family, SARS-CoV-2 status and lab do not explain it.
* Within entry: only 5 entries have both error and clean κ chains; 2/5 vs 6/8 cis, p = 0.45.
* Errors only in CDR1, CDR2 or FR3: no association (n = 4).
* Paper-level clean-chain cis rate: no association (p = 0.34).
