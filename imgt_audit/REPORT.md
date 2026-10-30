# Structural audit of IMGT unique numbering

**Question.** Does IMGT numbering put structurally equivalent residues at the same position number?

**Method.**
1. **Domain types.** IMGT/DomainGapAlign (DGA) offers seven domain types. I audited the six that have a fold-based numbering: V, C, G (MHC groove), S (SRCR), F (FN3) and A (Apple/PAN). R ("RPI") returns short unstructured regions only, so I left it out.
2. **Numbering.** 102 PDB entries were chosen to cover every IMGT domain label: IG, TR αβ and γδ, MHC-I/II, MHC-like, IgSF V-like and C-like, SRCR, FN3 and Apple. Each protein entity's full SEQRES was submitted to DGA for the relevant types.
3. **Threading.** Per-residue IMGT labels were read from DGA's alignment ruler, which is exact for insertions such as 111.1, 84.x/85.x and 72A. They were mapped onto observed residues by `label_seq_id`. This gave 313 domains: V 100, C 146, G 44, F 12, A 6, S 5.
4. **Structural comparison.** Every domain pair of the same type (15,900 pairs) was aligned with TM-align, which is sequence-independent. Residue pairs within 3 Å Cα after superposition count as structurally equivalent. Each pair was then checked for whether both residues carry the same IMGT label.
5. **Error criterion.** A numbering error is a label shift that is consistent across many pairs between two classes, while each class agrees with itself.
6. **Independent check.** Each headline claim was re-tested by Kabsch superposition on conserved core labels only. This compares Cα distances for the IMGT pairing against the proposed pairing and does not use TM-align's residue alignment.

Code: `imgt_audit/tools/`. Tables: `imgt_audit/results/`.

---

## What holds up

- **Ig β-sandwich core.** The B, C, E and F strands agree ≥ 0.97 within every IG/TR class and between VH, Vκ, Vλ, Vα and Vβ. The anchors Cys23, Trp41, 89, Cys104 and 118 are consistent.
- **V ↔ C equivalence.** The cross-type V vs C claim holds for B, C, E, F and the start of the G strand.
- **V ↔ FN3 equivalence.** Fibronectin-type FN3 matches V labels at 0.93–1.00 in B, C, E and F.
- **G-domain β-platform.** Positions 3–36 agree at 0.97–1.00 across MHC-I α1/α2, MHC-II α1/β1 and MHC-like proteins.
- **Apple domains.** Self-consistent (0.95–0.98).
- **Non-finding: C vs FN3.** TM-align gives a 2-residue offset between C and FN3. Because V≈C and V≈FN3 both hold, C≈FN3 should hold too. The offset is therefore a β-step slip in the superposition, not a numbering error.

## Errors found

Confidence: **High** means consistent at ≥ 85% across pairs and confirmed by the core-superposition check. **Med** means supported by one method, or a loop region.

### A. Scheme-level errors (IMGT gap-placement rules; would affect any IMGT implementation, including ANARCI-style)

| # | Where | Observation | Evidence (TM pairs; core check) | Conf. | Proposed fix |
|---|---|---|---|---|---|
| A1 | **V-LAMBDA, A strand (8–10)** | λ residues labelled 8/9 are structurally κ/TR 9/10. The λ deletion is at 8, not at 10. | 114/114 pairs; median Cα 3.7 Å → 1.0 Å, 100% | High | Place the Vλ A-strand gap at 8: relabel λ 8→9 and 9→10. |
| A2 | **Light chains / TRBV, D strand (80–85)** | Vκ/Vλ lack 81–82, but their 83/84 sit on VH 81/82. TRBV lacks 82, but its 83 sits on VH 82. TRAV 81/82 sit on κ 83/84. The deletion belongs next to 85 (the DE turn, where Kabat puts 82a–c), not after 80. | λ 96%, κ 57–79%, TRBV 93%; core 5.3→1.4 Å (λ), 6.9→3.4 Å (κ), 4.7→1.9 Å (TRBV), 97% | High | Number short D strands left-justified from 80 (…80, 81, 82, gap 83–84, 85). |
| A3 | **TRBV C″ strand (65–74)** | TRBV 66–72 and 74 sit on IG 65–71 and 72. The extra TRBV residue belongs to CDR2, and TRBV lacks 73–74. TRBV vs VH C″ agreement is 0.02. | 77–90% vs κ; core 4.5→2.7 Å, 85% | High | Shift TRBV C″ by −1 (66–72→65–71, 74→72) and absorb the extra residue into CDR2. |
| A4 | **CDR1 N-terminal base (27–30)** | Light-chain 27/28/29 sit on VH 28/29/30, and TRBV 27/28 on VH 28/29. VH has one extra residue at the B-strand/CDR1 junction ("GFTF"). κ 30/36 also cross-map with VH 35, so the gap in the middle of CDR1 is not where the loops actually differ. | κ→VH 49–87%; core 4.4→2.7 Å, 92% (TRBV 71%) | Med–High | Anchor light-chain/TR CDR1 one position later (gap at 27), or define CDR1 structurally. This changes CDR1-IMGT boundaries, so treat it as a scheme decision. |
| A5 | **G domains, MHC-II α1 helix N-terminus (50–62)** | G-ALPHA 51/59/60/62 sit on MHC-I α2 50/54/55/59, an offset of 1–5. | 45–59 of 60; core 8.4→2.8 Å, 100% | High | Re-anchor the MHC-II α1 helix start; numbers 52–62 do not mark equivalent helix turns across MHC-I and MHC-II. |
| A6 | **G domains, MHC-II β1 helix C-terminus (84–90)** | G-BETA 84 ≈ α1/α2 85, and 87 ≈ α2 90. | 41–49 of 60; core 4.7→2.5 Å, 92% | High | Re-register the β1 helix C-terminal segment. |
| A7 | **G-ALPHA2 helix kink insertion** | α2 72A sits where α1 has 72, and α2 72 sits on α1 71. The insertion should be 71A. | 75/100 (TM); core check not decisive (helices move relative to the platform) | Med | Move the kink insertion before 72. |
| A8 | **V-LIKE domains** | Within-class agreement is low even in strands: C′ 0.27, C″ 0.26, D 0.32, G 0.40. | strand matrix | Med | V-like numbering by sequence profile is unreliable; use structure-based assignment. |
| A9 | **FN3 (F type), cytokine-receptor FN3** | GHR, IFNGR1, IL-2Rβ/γc, gp130 and G-CSFR agree with each other only at 0.12–0.16 in C, D and G (B strand 0.94). Their offsets from FN1/TNC are mostly ±2, the signature of a sequence alignment landing one β-step off. | strand matrix | High | F-domain numbering is not structural for this subfamily; it needs structure-based references. |

### B. Reference-directory and implementation errors (IMGT data / DGA software)

| # | Where | Observation | Evidence | Proposed fix |
|---|---|---|---|---|
| B1 | **Mouse IGHG1\*03 CH1 reference** | Every CH1 assigned to this allele (12e8, 1yjd) is misnumbered: 0/90 structural matches. The offset is +4 at the N-terminus and grows to about +9 by the G strand. A mouse IgG1 CH1 differing by about 4 residues, assigned to IGHG1\*01 (1mlc), gets 88/90. | TM-align vs human CH1 | Fix the IGHG1\*03 gapped reference. Any query whose top hit is this allele inherits the error. |
| B2 | **TRDC reference** | One extra DE-turn insertion (85.4) shifts the whole E strand +1 (TRDC 86–91 = all other C domains' 85–90), plus G 118 = 119 and FG 109–118. Seen in human and mouse. E-strand agreement with every other class is 0.00. | 100% of pairs; core 3.6→0.8 Å, 99% | Remove one 85.x insertion from TRDC and re-gap the D/E strands. |
| B3 | **TRGC reference** | F strand 97/98 = others' 95/96, and the CD-strand insertions are offset by 1. | core 6.5→2.1 Å, 100% (n=42) | Re-gap the TRGC EF/F region. |
| B4 | **TRAC** | Self-consistent, but agrees poorly with every other C class. TRAC is structurally atypical (TM about 0.55), so cross-class shifts are only low confidence. | strand matrix | Needs TRAC-specific structural definitions. |
| B5 | **DGA display vs FASTA export** | In 9 domains the "sequence with gaps" export differs from the ruler-aligned display: separators become `.`, and a blank appears at V-LIKE C-strand position 46. Parsing the FASTA export directly misnumbers residues downstream. | `results/thread.log` | Use the display aligned to the ruler; IMGT should fix the export. |
| B6 | **DGA cannot number some domains** | CD4 D1 (1cdh), PD-L1 V (3bik) and a mouse TRAV4/DV10 (1ypz): the query row overflows the ruler segments, so DGA assigns no valid IMGT number to those residues. | `DGA-INCONSISTENT` in log | Handle long insertions in strands C′/E explicitly. |

## Caveats

- **One reference per query.** DGA numbering follows the closest reference allele, so B-type errors hit only queries that match the bad reference.
- **Loops.** For CDR3, the FG loop and long CDR1/CDR2, low agreement mostly reflects genuine conformational diversity. I did not count those as numbering errors, except at the CDR1 base (A4).
- **Small classes.** TRDC/TRGC (3 each), SRCR (5) and Apple (6) rest on few structures. The SRCR numbering looks noisy (0.58–0.79 agreement; ±2 shifts in the A strand), but n is too small to diagnose.
- **Cutoff.** The 3 Å cutoff is strict for helices. G-domain helix findings were therefore required to pass the core-superposition test too.

## Relevance to SAbR

A1–A4 are properties of the IMGT scheme itself, not of DGA. ANARCI-compatible IMGT output, including SAbR's, inherits them. Because SAbR aligns structure embeddings to reference positions, structurally non-equivalent residues sharing a label (λ 8–10, light-chain 81–84, the CDR1 base) are places where the scheme forces the alignment against the structure.

Two options:
1. **Structure-consistent internal reference.** SAbR could align against a reference relabelled per A1–A4, then map back to standard IMGT on output. This keeps the scheme while improving the alignment.
2. **Optional scheme variant.** Offer a structure-consistent IMGT variant with the A1–A3 fixes.

**Next step:** measure how often SAbR's current output differs from structure at these positions on the test set.
