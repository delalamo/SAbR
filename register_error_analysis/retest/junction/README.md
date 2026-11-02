# CDR3/FR4 register errors: how many, and is there an algorithmic pattern?

## How many

**2024 study (Table S1).** 35 of its 165 antibody errors (21%) touch CDR3/FR4 (IMGT ≥105), in 24 of 87 entries. The
others are 98 in FR1–FR3/CDR1–2, 29 in constant domains and 3 unmapped.

**AlphaFold-independent scan** (15,865 single-domain V chains, 3–5 Å, to 2026):

| | Runs | Entries | Independent events* | EM |
|---|---|---|---|---|
| CDR3–FR4 junction (W/F118 displaced) | 124 | 74 | 66 | 89% |
| FR4 bulge after the W/F-G motif (IMGT 120) | 55 | 21 | 19 | 95% |
| Any run touching IMGT 105–128 (all origins) | 281 | 124 | 112 | 92% |

\*One per antibody per paper. (`scripts/counts.py`)

## The pattern (`scripts/mechanism.py`, `tables/mechanism.log`)

**1. Slips sit on the C-terminal side of the loops.** These counts exclude HIV/FR3-convention artifacts and
chain-end/gap runs.

| Loop | Before the loop (N-flank): events | After the loop (C-flank): events | Shift on C-flank |
|---|---|---|---|
| CDR1 | 4 | 9 | +7 / −5 runs |
| CDR2 (no Cys anchor) | 13 | 25 | +16 / −29 |
| **CDR3** | **9** | **80** | **+149 / −22 (87% +1)** |

* **What +1 after CDR3 means.** The loop was built with one residue too few. Its last residue sits in the first FR4
  slot, the strand slides by one, and Trp/Phe118 is pulled out of the core.
* **Why this suggests N→C assignment.** If the sequence is assigned from the N-terminal side (counting forward from
  FR1, or from the Cys23/Cys104 disulfide), a residue miscount in a poorly resolved loop surfaces in the strand that
  follows it.
* **Not just anchoring.** CDR1 and CDR3 have a Cys anchor on their N-flank, which could also explain their asymmetry.
  But CDR2 has no Cys anchor and still shows 2:1.

**2. Errors happen when the starting model's CDR3 length differs from the target's.** This covers EM chains whose
recorded PDB starting model contains an antibody domain of the same type.

| Template antibody | EM chains | CDR3–FR4 junction errors |
|---|---|---|
| Different antibody, same CDR3 length | 410 | **0** |
| Different antibody, CDR3 length differs by 1 | 128 | 0 |
| **Different antibody, CDR3 length differs by ≥2** | 361 | **9 (2.5%)** (vs same length: Fisher p = 0.001) |
| Identical antibody | 507 | 4 (0.8%, inherited) |

* **The 9 chains are only 3 independent cases:**
  * 6ORP←4FQQ: target CDR3 11 shorter than the template's;
  * 7K8X/7K8Y←6W41: 10 longer;
  * 7V2A←6Z3P: 4 longer.
* **The shift sign follows the length difference.** A target longer than its template gives +1 (loop too short) in
  7/7 runs; a shorter target gives −1 in 3/5.

**3. Starting-model type and fitting protocol** (EM chains, CDR3–FR4 junction + FR4 bulge):

| | Chains | Error rate |
|---|---|---|
| AlphaFold starting model | 290 | 0.34% (adjusted OR 0.34, 0.05–2.6) |
| PDB template | 2,023 | 1.09% (OR 0.56, 0.28–1.09) |
| None recorded | 8,186 | 1.70% |
| Ab initio fitting | 441 | 0.23% |
| Flexible fitting | 885 | 1.13% |
| Rigid-body fitting only | 1,226 | 1.79% |

* **Software records:** chains recording Coot, ISOLDE or Rosetta have lower rates (0.5–0.7% vs about 1.6%).
* **ModelAngelo is never recorded,** so automated builders can't be tested.

## Interpretation

The data fit one workflow:
1. A Fab template is docked into the map, often rigid-body only.
2. The new sequence is threaded onto it from the N-terminus.
3. CDR3 is not fully rebuilt, or is rebuilt too short in weak 3–4 Å density.

Whatever residue count is missing or extra in CDR3 then ends up in FR4. FR4's glycine-rich W-G-X-G strand gives few
side-chain cues at 3–4 Å, so the slip goes unnoticed. Starting models with the right loop length (AlphaFold, or
building from scratch) show the lowest rates.

**Caveats.**
* Only 21% of erroneous chains record a starting model.
* The template-length result rests on 3 independent cases.
* The flank asymmetry could partly be structural: FR4 is the edge strand at the V–C elbow, and Cys104 anchors the
  N-terminal side of CDR3.

## Is this a known tool problem? Methods sections and bug reports

**Methods sections** (`scripts/methods_mining.py`, `tables/methods_keywords.csv`, `tables/methods_snippets_cases.txt`).
* **Sample:** open-access full texts from Europe PMC for 38 of the 59 papers behind CDR3/FR4-error entries. Controls
  are 199 papers of error-free EM antibody entries from the same years.
* **No building tool is over-represented in the error papers:**

  | Tool | Error papers | Controls |
  |---|---|---|
  | Coot | 68% | 74% |
  | Phenix | 74% | 76% |
  | ISOLDE | 8% | 7% |
  | Rosetta | 16% | 17% |
  | SWISS-MODEL | 13% | 14% |
  | A template of another Fab | 42% | 33% |

* **The only difference is AlphaFold/ColabFold:** 8% vs 30% (p = 0.004).
  * This is mostly timing: before mid-2021, 0% vs 3%.
  * After mid-2021 the error papers still mention it less: 14% vs 36%.
* **The error papers describe the same workflow again and again.** A *different* antibody, or a homology model of
  one, is docked rigid-body and mutated to the target sequence, and the CDRs are rebuilt by hand. Examples:
  * 7V2A: the heavy chain of 6Z3P and the light chain of 6PZE, fitted in Chimera;
  * 7MKL: Fabs found by BLAST, 1KIQ for VH and 5XJM for VL;
  * 7STF: 7KGU "in silico mutated" in Coot;
  * 7K22/23: an unrelated Fab, 3GK8, "mutated to match", with the CDRs "manually rebuilt";
  * 7D3K–R: crystal structure 6E9U rigid-body fitted;
  * homology models from SWISS-MODEL, BioLuminate, Discovery Studio or abYsis.

  Controls use the same workflows at similar rates.

**Literature and bug reports.**
* **No antibody-specific report.** Nothing describes FR4/CDR3-junction register errors, and there is no tool bug,
  changelog fix or GitHub issue that would produce this pattern. Phenix CHANGES, sculptor/Chainsaw documentation,
  ModelAngelo/ISOLDE/Coot issues and the Sánchez Rodríguez 2024 paper were checked; that paper does not discuss
  antibodies or loops.
* **The general mechanism is documented:**
  * **Phenix `assign_sequence` documentation:** "loops are often built short by one or a few residues."
  * **Chojnowski 2023 (PMC10306063):** "A solvent-exposed, poorly resolved loop … was traced too short in the
    original model and resulted in a two-residue sequence-register shift in a C-terminal part of the chain."
  * **Croll 2018 (ISOLDE):** a CHAINSAW homology-model register error "presumably arising from a misalignment
    between the sequence and the homology template", with the residues after it out of register.
  * **Sculptor and Chainsaw:** both only rename or delete template residues according to the alignment. Neither
    builds insertions.
* **Not found anywhere:** a measured N- vs C-side bias, or a link to template CDR3-length mismatch. CCP4BB and the
  Coot mailing list could not be searched.

**Conclusion.** This is not a bug in one program. It is the documented general failure, a poorly resolved loop built
too short, with the error showing up downstream. It is concentrated in antibodies by the common workflow of starting
from a different antibody whose CDR3 length differs. FR4 takes the slip because it is the strand after the loop
that is rebuilt, and it is glycine-rich with few side-chain landmarks.
