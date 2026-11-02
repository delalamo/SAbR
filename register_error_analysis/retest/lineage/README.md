# Do the non-inherited errors spread through *related* antibodies (same V or J gene, close relatives)?

**Question.** About three-quarters of the 2024 study's antibody errors arose in the first structure of that antibody
(`../../template_propagation`, first-instance section). Could some of them still be copied, from a *different but
related* antibody? Candidates:
* the same J gene (relevant for CDR3–FR4 junction errors, where Trp/Phe118 ends up facing the solvent);
* the same V gene (CDR1/CDR2 boundary errors);
* a close relative (≥90% V-region identity: variants, humanised or matured versions).

**Data.**
* The 15,865 single-domain V chains of the AlphaFold-independent 3–5 Å scan (to 2026), grouped into 6,518 units, one
  per antibody per paper.
* Their classified disagreement runs (`../unexplained`), in these classes:
  * CDR3–FR4 junction: 66 erroneous units, including 18 with Trp/Phe118 pulled out of the core;
  * FR4 bulge: 18–19 units;
  * CDR1/CDR2 boundary: 30 units;
  * chain end / gap: 98 units (comparison class).
* Antibody identity and similarity use V-region sequences only.

## 1. Are these errors concentrated in particular germline genes?

| Class | Gene | Heterogeneity across genes |
|---|---|---|
| CDR3–FR4 junction | J | p = 0.12. Rates 1.0–2.4% in every common J gene; IGHJ4 has the most only because it is the most common |
| FR4 bulge | J | p = 0.48 |
| CDR1/CDR2 boundary | V | p = 0.095 (IGHV3-53 4%, the public SARS-CoV-2 germline) |
| Chain end / gap | V | p = 5×10⁻⁵ (IGLV1-44 7.4%, IGHV4-34 6.4%) |

## 2. Do errors follow earlier errors in related antibodies? (`scripts/lineage_perm.py`)

**The test.**
* **Statistic:** the number of erroneous units that had an erroneous predecessor of the same class in the previous 3
  years. A predecessor is a different antibody, released before the unit was deposited.
* **Null:** labels shuffled within germline gene × method × period strata (2,000 permutations). This keeps any
  germline's or era's own error-proneness.

| Class | Tier | Observed | Expected | p |
|---|---|---|---|---|
| CDR3–FR4 junction | same J gene, other lab | 37 | 39.7 | 0.90 |
| | same V+J gene, other lab | 13 | 8.1 | 0.047 |
| | close relative (≥90%) | 1 | 1.1 | 0.59 |
| Trp/Phe118 pulled out | same J gene | 4 | 4.2 | 0.75 |
| | same V+J gene | 1 | 0.2 | 0.16 |
| FR4 bulge | same J gene | 5 | 4.1 | 0.39 |
| | close relative | 1 | 0.1 | 0.11 |
| CDR1/CDR2 boundary | same V gene | 5 | 4.7 | 0.59 |
| | same V+J gene | 0 | 2.0 | 1.0 |
| Chain end / gap | same V+J gene, other lab | 24 | 18.5 | 0.038 |
| | close relative, same lab | 1 | 0.06 | 0.06 |

The same-lab tiers are all null. Full table: `tables/lineage_permutation.csv`.

## 3. Was the stated starting model a related antibody carrying the same error? (`scripts/lineage_templates.py`)

**Coverage.** Only 103 of 491 erroneous runs (21%) come from entries that record an experimental starting model. The
template antibody closest in sequence was found in each, and the SAbR/ANARCI detector was run on it.

| Template relation | Erroneous chains: template carries an error | Clean control chains: template carries an error |
|---|---|---|
| Identical antibody | 6/7 | 7/105 (7%) |
| **Close relative (≥90%)** | **8/8** | **1/38 (3%)** |
| Same V+J gene | 3/11 | 4/12 |
| Same J gene only | 0/16 | 3/21 |
| Unrelated antibody | 0/21 | 1/58 |

**Errors explained by a related, non-identical template with the same error: 11 runs in 5 events.**

| Flagged entries | Template | Relation | Error class |
|---|---|---|---|
| 7JHG/7JHH | 6CMO | 98% V identity, other lab | gap |
| 7QJI/7QJJ | 7QIA | 94%, same lab, 2 days apart | gap |
| 8F2X H+L | 7CZU | 91–95%, other lab | junction and gap |
| 4XNZ | 4JB9 | same V+J, same lab | FR3 83–85, a likely convention artifact |

That is 2.4% of the 450 classified runs. A further 6 junction runs come from an *identical* antibody's template
(3IDY←3IDX, 6QX3/6QXE←6QPG), i.e. same-antibody inheritance.

## Conclusion

1. **Errors pass between close relatives, not between germline lineages.**
   * A ≥90%-identical template with the error transmits it every time (8/8). A clean close relative almost never
     produces it (1/38).
   * Antibodies sharing only a V gene or a J gene do not transmit errors. Errors are not concentrated by J gene,
     and do not follow earlier errors in same-J antibodies more than chance.
   * Same-V+J templates are no more often erroneous for erroneous chains (3/11) than for clean ones (4/12).
2. **The CDR3–FR4 / Trp118 errors recur independently.**
   * There is no J-gene concentration and no lineage clustering (4 observed vs 4.2 expected).
   * No related template carries them.
   * They are a recurring *failure mode* of model building, not an inherited one: a CDR3 built one residue short or
     long, the FR4 strand sliding to compensate, and the Trp/Phe118 anchor pulled out. They track cryo-EM, poor local
     map quality, 2020–21 and atypical CDR3 lengths (`../unexplained`).
3. **What this adds to the earlier count.**
   * Close-relative templates add roughly 2–3% of errors to the ~19% inherited from the same antibody.
   * The small excess within same V+J combinations (about 5 units each for junction and gap errors, p ≈ 0.04–0.05)
     is not supported by the starting-model data. It more likely reflects antibodies of the same class being built
     at the same time (e.g. SARS-CoV-2 public antibodies) than copying.
   * This is a lower bound: 79% of erroneous runs record no experimental starting model.

## Files

* `scripts/lineage_units.py`: units, germlines, V regions and classes (shared).
* `scripts/lineage.py`: germline concentration, plus a stratified comparison. That comparison is biased because an
  erroneous unit cannot be its own predecessor; it is superseded by the permutation test.
* `scripts/lineage_perm.py`: permutation test.
* `scripts/lineage_templates.py`: stated starting models.
* `tables/`: `lineage_permutation.csv`, `lineage_templates.csv`, `lineage_germline_top.csv`, logs.
