# Case files for CDR3/FR4 slips: modes D+ (CDR3 one short), D− (CDR3 one long), E (FR4 bulge)

**Lens.** I built 27 case files, 9 per mode. They are mostly cryo-EM, from 26 labs, deposited 2016–2026. Each case is
paired with 1–2 correctly modelled controls: a correct antibody from the same paper, and/or the same antigen from
another lab within 12 months. Specifics that recurred across the case files were then tested on every D/E unit,
against matched clean cryo-EM units and against clean units from the same papers.

**Files.**
* `tables/case_files_DE.txt`: narrative case files, with methods excerpts and controls.
* `tables/case_table_DE.csv`: one row per case.
* `tables/cases_DE.csv`: cases plus controls, all fields.
* `tables/case_commonalities.csv`: per-mode summary.

## 0. First, a validity check: about a quarter of D/E "errors" are not reproduced independently

**Method** (`s02_struct.py`).
* Each model's framework Cα atoms (FR1–FR3, excluding every run) were superposed on a clean high-resolution reference:
  5DT1 H/L for heavy and λ, 1N8Z A for κ.
* Each residue at sequence-based IMGT 116–124 was then assigned to the nearest reference Cα slot.
* A run is confirmed when its residues sit one or more slots away, in the direction of the SAbR–ANARCI shift.
* This test uses neither SAbR nor maps.
* False-positive rate in 271 clean controls: 1.8% (any FR4 residue off-slot).

**Results by mode** (`tables/independent_check_DE_units.csv`):

| Mode | Units | Confirmed | Not confirmed | Not testable |
|---|---|---|---|---|
| D+ | 40 | 29 | **11** | 0 |
| D− | 22 | 11 | 7 | 4 |
| E | 18 | 16 | 2 | 0 |
| **All** | **80** | **56** | **20** | **4** |

* **D+ models ending at or before IMGT 121 are almost never reproduced:** 10 of 12 are not confirmed, vs 1 of 28 longer
  models (Fisher p = 8e-7). Examples: 7K22/7K23, 7T3D, 8TQI, 8PQ2, 9DN2-H, 8GRR, 9FTE and 30US.
* **What SAbR does at a chain end.** In these models W/F118 is the last or second-last built residue. SAbR numbers it
  119 and pulls the preceding residues forward by one. The superposition puts them in their canonical slots, and Trp118
  stays buried (Cα count within 10 Å: 15–17).
* **Most other unconfirmed runs sit next to unbuilt FR4 residues:** 26PF-L, 7VNE, 8JAP-L, 9DN2-L and 9NNZ.
* **6XJA is not a register error.** It is a polyclonal IgA whose heavy chain carries a poly-Ala placeholder CDR3,
  "AAAGAAAAAA" (V identity 48%), so its register is undefined.
* **Mode A is real.** The same test applied to FR1 N-terminal runs confirms 48 of 57 (84%), vs a 1% null rate in clean
  chains (`s09`, `tables/modeA_nterm_check.csv`).
* **Recommendation:** drop or flag D/E runs in chains whose model ends at IMGT ≤121, or that abut unbuilt FR4 residues,
  unless confirmed. The earlier "Fv-only / FR4 truncated" association for D+ is partly this artifact.

All tests below use confirmed (or untestable) runs only.

## 1. Per-mode commonalities in the case files (n = 9 cases each; descriptive)

**D+** (5KEN, 6QX3, 7K8X, 7CZV, 7V2A, 7WE8, 8GNH-H, 8XSD, 12TY)
* **Long CDR3 with a Tyr or Val right before W118.**
  * CDR3 length: median 17 vs 14 in the controls.
  * Seven of nine are heavy chains ending in -DY, -DV or -MDV (IGHJ4/J6).
  * Shifting the strand puts that Tyr or Val into the Trp118 pocket.
* **CDR3 always contacts the antigen** (9/9).
* **Three cases record a PDB template of a different CDR3 length or their own flawed earlier model:**
  * 7K8X←6W41: CDR3 12 vs 22, rigid-body only;
  * 7V2A←6Z3P: 14 vs 18, rigid-body only;
  * 6QX3←6QPG: the lab's own 3.34 Å X-ray structure carries the same +1 slip, so the EM models inherited it.
* **The 8GNH methods** describe "a Fab model … generated as templates from our previous work, docked into the local
  refined map".
* **12TY: the slip equals a sequence discrepancy within the same paper.**
  * Nanobody M6 was deposited with two CDR3 sequences: NANFQIHRSG**AD**YVRNY (17) in 12TX/TY/TZ/UA/UB/UD, and
    NANFQIHRSGYVRNY (15) in 12TW/UC/UE.
  * The +2 slip occurs in four 17-residue entries.
  * 12TX and 12TZ, with the same 17-residue sequence, are correct, as are the 15-residue entries.
  * This most likely means a 15-residue model was re-sequenced without rebuilding.
* **Six of nine target the SARS-CoV-2 spike**, as do their within-paper controls (71%).

**D−** (6JMR, 6LHT, 6ORP, 6XJA, 7D3K-L, 7MEJ, 8GNH-L, 8JAG, 9MA0)
* **Heterogeneous** antibodies: mouse κ, nanobody, NHP and human.
* **Two of nine are mouse IGKV8-30 κ** (6JMR, 6LHT); 6LHT also has a +5 CDR-L1 slip.
* **Template mismatch:** 6ORP←4FQQ (CDR3 26 vs 15, rigid-body only).
* **Less antigen contact:** CDR3 contacts antigen in 7/9 and the V domain in 7/9. 6JMR and 6XJA touch no antigen.
* **Poor or unusual entries:** 6XJA is a placeholder sequence; 7D3K has clashscore 82. 9MA0 was built *ab initio*,
  so not every D− comes from a template.

**E** (7CZR, 7KC1, 7MKL, 7O85, 7T0W, 8RMK, 9DM7, 9PV5-L, 9ZXD)
* **Every case is confirmed independently.** Residues 118–119 sit in register, and 120 onward is +1 to the C-terminus.
  There are no chain breaks (`s06`), so a β-bulge is modelled into strand G.
* **Short or ordinary CDR3s** (median 10). **Five of six open-access methods sections mention local or focused
  refinement** (n.s. vs controls; §3).
* **Copies, and sometimes both chains, bulge.** In 9PV5 both H and L bulge. In 7KC1, 7O85, 7T0W and 8RMK every
  symmetry copy carries the bulge, so each copy group was built once and replicated.
* **Mouse IGKV8-30 κ again:** 9PV5 (and 7WP6 among the non-chosen units).
* **7CZR (Cell Res 2021): a sequence-provenance anomaly.**
  * Two different clones, c542 (described as IGHD3-10/IGHJ6) and c642 (IGHD1-26/IGHJ6), are deposited with an
    *identical* J4-type CDR3/FR4: ARETLAFDY-WGQGTLVTVSS.
  * All three entries (7CZR, 7CZS, 7D00) have the bulge.
  * The same paper's other heavy chains (7CZU/V) have D+ slips. Its κ chains, deposited as 100%-germline "IGL c4203
    … Uncharacterized protein", have N-terminal slips.
* **9ZXD:** the same nanobody in 9ZXC from the same paper is correct.

**Common to all three modes in the case files**
* **The partner chain of the same Fab is also wrong** (quantified in §2).
* **The framework is distorted away from the slip** (fw_rmsd median 1.2–1.9 Å vs 1.0–1.15 Å in the controls).
* **Not common:**
  * a specific program;
  * a specific deposition or processing site (PDBj 56% of D cases vs 41–44% of their controls; E 11% vs 12%);
  * a crowded antigen. Cases and their controls have the same SARS-CoV-2 share. For D+ cases on the spike, 2–10% of
    the other labs' same-antigen entries in the same ±6 months have first-occurrence errors, at or below the 2020–21
    cryo-EM background.

## 2. Cross-case tests (all EM D/E units; `s07`, `s08`, `s12`; `tables/struct_tests.csv`)

The error set is the confirmed EM units: 49 (D+ 21, D− 13, E 15). Controls are 271 matched clean EM units, and 108
clean units from the same papers. Models adjust for year bin, resolution and chain type, with SEs clustered by paper;
the within-paper comparison uses a conditional logit.

| Feature | Errors | Matched clean | OR (q) | Within paper |
|---|---|---|---|---|
| Framework RMSD to reference, excluding runs (Å) | **1.56** | 1.02 | **3.2/SD (q<0.001)** | 1.56 vs 1.09, **OR 10/SD (q = 0.004)** |
| W/F118 exposed (<14 Cα within 10 Å) | 22% | 6% | 5.7 (q = 0.005) | 22% vs 15%, n.s. (consequence of the slip) |
| Model ends before IMGT 125 | 20% | 4% | 7.9 (q = 0.001) | 20% vs 15%, OR 3.1 (p = 0.15): **a paper-level habit** |
| SEQRES ends <8 residues after W118 | 8% | 0.7% | 17 (q = 0.03) | n.s. |
| CDR3 contacts antigen | 84% | 87% | 0.73 (n.s.) | 84% vs 82%, n.s. |
| D− only: V domain contacts antigen | 69% | 93% | 0.13 (p = 0.011, q = 0.07) | n.s. (6 papers) |
| D+ only: CDR3 contacts antigen | 21/21 | 87% | Fisher p = 0.09 | n.s. |

* **VH–VL pairing** (EM Fab entries with both chain types, n = 2,123; `tables/hl_cooccurrence.csv`):
  * When VH has a confirmed D/E slip, the VL of the same entry has a first-occurrence error in **34% vs 1.3%**
    (adjusted OR 36, p < 1e-4).
  * Both chains with D/E: 6.3% vs 0.8% (adjusted OR 6.3, p = 0.06).
  * With the artifacts included this reads OR 17. Part of that was artifacts occurring in both chains (9DN2, 8JAP).
* **Mouse IGKV8-30 κ:** 4/41 EM mouse κ units with a confirmed D−/E slip, vs 4/333 for other mouse κ genes (OR 8.9,
  Fisher p = 0.006). The four are from 4 labs and 4 papers (6JMR, 6LHT, 7WP6, 9PV5). Not adjusted; post hoc.
* **The same antibody deposited with different CDR3 lengths within one paper:** 2/16 D/E multi-entry groups vs 9/1,190
  (OR 19, p = 0.008). That rests on only 2 events, and most control "variants" are different antibodies sharing a
  germline.
* **Same antibody, same paper, right and wrong entries** (5 informative groups): the error entries had lower map-model
  fit in 3 of 4, and no difference in resolution. Inconclusive.

## 3. NO difference

| Area | What was tested |
|---|---|
| Methods text (Europe PMC open access; 27 D/E vs 74 year-matched clean-control papers) | Local/focused refinement 59% vs 49%; template Fab/homology 56% vs 55%; AlphaFold/predictors 15% vs 23%; rigid body 19% vs 22%; mutated to sequence 11% vs 15%; ISOLDE 4% vs 14%; Rosetta 19% vs 24%; ModelAngelo/automated 0% vs 1%; Fv-only statements. All q = 1, and the same within E, D+ and D− |
| Local structure | Partner contacts; side-chain truncation at the site; relative B-factors at the site; SEQRES residues present but unbuilt after W118; Cα–Cα chain breaks (none) |
| Antigen binding | CDR3–antigen contact (overall and within paper) |
| Case vs control metadata (descriptive) | EMDB local-refinement keyword; processing site; same-antigen crowding; same antibody solved elsewhere |

## 4. What links the D/E structures

1. **It is the model, not the antibody.**
   * The whole V framework of an erroneous chain is distorted, even within the same paper.
   * Its Fab partner chain is wrong 34% of the time.
   * Within the same paper, the error chains do not differ from correct ones in antigen contact, truncation or
     B-factors.
   * This is a per-model (per-Fab) building event that leaves several marks, not a property of the sequence.
2. **Locally, the recurring routes are:**
   * a starting model with a different CDR3 length (7K8X, 7V2A, 6ORP), or a lab's own flawed earlier model
     (6QX3←6QPG);
   * a model re-sequenced to a variant without rebuilding (12TY);
   * deposited sequences that do not match the clone (7CZR family, 6XJA);
   * a Tyr/Val before W118 that makes the +1 register look plausible.
3. **Gene-specific risk:** mouse IGKV8-30 κ (long L1-17) for D−/E.
4. **A detector artifact:** about 25% of D/E units, mostly D+ in models ending at W118, should not be counted as
   register errors.

## Caveats

* **Small, hand-picked case set** (9 per mode), and the commonality tables are descriptive.
* **Superposition test:**
  * It uses one reference per chain type.
  * It covers only IMGT 116–124, so 4 D− runs could not be tested.
  * Nanobodies are compared with a VH reference. The FR4 strand is conserved, but the framework RMSD of VHHs is
    inflated; this is controlled by matching on chain type.
* **Overlap with other lenses.** The framework-distortion signal overlaps the β-sheet and geometry findings of
  `pdb_meta` and `struct_unusual`. The FR4/SEQRES truncation overlaps `seq_novelty`.
* **Full-text coverage.** Only 59% of papers have open-access full text, and EMDB keywords under-record local
  refinement.
* **Sequence-provenance anomalies** are case-level observations. The true sequences of these clones were not
  available to verify.
* **IGKV8-30 and the D− antigen-contact results** are post hoc, with small n.
