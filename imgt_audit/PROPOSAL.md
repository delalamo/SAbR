# A structurally consistent revision of IMGT numbering: large-scale audit and proposed rules

This follows `REPORT.md`, a 313-domain pilot. Here the audit is scaled up six-fold, the errors are re-tested on the larger set, and a rule system is proposed and tested. Under that system a position number means the same place in the structure in every domain of a fold family.

## 1. Data

| Source | Entries | Selection |
|---|---|---|
| SAbDab2 (antibodies) | 380 | X-ray ≤ 2.5 Å; ≤ 3 per (heavy subgroup × light subgroup × light type) stratum; unique VH. Includes nanobodies and non-human species. |
| STCRDab (TCRs, with their MHC) | 121 | X-ray ≤ 3.0 Å; one per (TRAV, TRBV) subgroup pair, plus every γδ TCR |
| RCSB Pfam clusters (95% identity, ≤ 3.0 Å) | 508 | V-set, C1-set, I-set, C2-set, Ig_2/Ig_3/ig, MHC-I, MHC-II α/β, FN3, SRCR, PAN |

Every protein entity was numbered by IMGT/DomainGapAlign (DGA) and threaded onto observed residues, as in the pilot. After keeping domains that are ≥ 80% observed and ≤ 3.0 Å, and removing redundant sequences, **1,954 domains** remain:

| Type | Domains | Classes |
|---|---|---|
| V | 1,099 | VH 403, Vκ 307, Vλ 85, Vα 117, Vβ 122, Vγ 6, Vδ 6, V-like 53 |
| C | 631 | CH1 175, CK 69, CL 44, CH2/3/4 21, TRAC 63, TRBC 42, TRGC/TRDC 5, β2m 16, MHC-I α3 54, MHC-II α2/β2 39, IgSF C2/I-set 103 |
| G | 173 | MHC-I α1 42, α2 45, MHC-II α 20, β 32, MHC-like α1/α2 34 |
| FN3 | 27 | |
| SRCR | 19 | |
| Apple | 5 | |

FN3, SRCR and Apple stay small because IMGT's F/S/A reference directories only cover immune-related proteins. Most Pfam FN3 domains get no DGA hit.

**Correction to the pilot.** IMGT's SRCR ruler prints adjacent numbers without a separator ("3839"). My parser read that as one number, so every SRCR position from 38 to 45 was mislabelled in the pilot. This is fixed here, and the pilot's "SRCR numbering is noisy" remark should be disregarded.

## 2. How "structurally consistent" was measured

- **Pairs and equivalence.** Each domain was aligned with TM-align, which ignores sequence, against 20 random domains of the same family. Residue pairs within 3 Å after superposition are taken as structurally equivalent.
- **Recall.** The fraction of structurally equivalent residue pairs that carry the same number.
- **Precision.** The fraction of same-numbered residue pairs that are within 3 Å, i.e. "same number ⇒ same place". This is what "consistent" means here.
- **Independent check of each change** (`validation.json`). For every proposed relabelling p→q in class X, domains of X were TM-aligned against domains of other classes of the same type in which q is unchanged. The check asks whether X's residue p lands closer to the partner's q (the proposal) or to the partner's p (IMGT). No consensus model is involved.

## 3. Proposed rules

The scheme keeps IMGT's integer frames (Ig 1–128, G 1–92, SRCR, Apple) so that numbers stay familiar. What changes is what a number *means*: a structural site rather than a column in a sequence alignment.

**R1 – One frame per fold.** V, C1, C2/I-set and FN3 domains share one Ig frame. MHC-groove domains have their own frame, as do SRCR and Apple.

**R2 – Anchors are structural sites.** A frame position is an anchor if, over the class consensus, its Cα has rms spread ≤ 1.6 Å after core superposition and it also meets one of these:
- (a) it is β or α by backbone H-bond pattern in ≥ 50% of members; or
- (b) it is a rigid loop position occupied in ≥ 65% of members **and** lies outside a hypervariable loop.

The hypervariable loops are V-domain BC 27–38, C′C″ 56–65 and FG 105–117. They never get loop anchors, so CDR numbering stays independent of conformation.

**R3 – Assignment by structure.** A residue takes an anchor number when it lies within 3 Å of that site in its *own class* model (VH, Vκ, CK…). The match is order-preserving and scored on Cα position plus Cβ direction, which separates neighbouring β positions.

**R4 – Strands are identified by their sheet partner, not by sequence position.** This settles three IMGT inconsistencies (§4.2): the C-domain "CD strand" and the FN3 "D strand" both pair with strand C, so both are C′ and take the V-frame C′ numbers.

**R5 – Cross-type identity.** Two anchors from different domain types (or classes) share a number only if they lie within 3 Å after superposition on the B/C/E/F core. That core is the only region verified equivalent in V, C and FN3. Loop anchors are never identified across types.

**R6 – Type- or class-specific sites take letters.** An anchor with no counterpart elsewhere is named by a letter suffix after the shared integer nearest to it in space, subject to chain order. Example: the C-domain A strand continues on the ABED sheet where V domains have no residues, giving 8A, 8B, …. Letters always denote structural sites and are never used for loop insertions.

**R7 – Loops are numbered by length.** Residues between two matched anchors fill the loop's free integers alternately from both ends, so gaps fall at the loop centre. Any surplus residues are insertions numbered **n.1, n.2, …** on the last N-side number, i.e. at the apex. All decimals attach on the N side, which removes IMGT's mix of N-attached (111.x, 45.x, 84.x) and C-attached (112.x, 85.x, 1.x) decimals.

**R8 – A loop never borrows a skipped strand's numbers.** If a domain lacks a strand (C1 domains lack C″), its residues there use only the free numbers adjacent to their own flanking anchors, and any surplus becomes insertions.

**R9 – Termini continue the integers.** N-terminal extensions are numbered 0, −1, −2, …, replacing IMGT's descending 1.1–1.8. C-terminal extensions count up past the last anchor.

**R10 – One sort order.** 8 < 8.1 < 8A < 8A.1 < 8B < 9. Every domain's labels must sort in chain order. This holds for all 1,954 domains.

## 4. Proposed changes to IMGT

Only changes with independent pairwise support (§2) or cross-type geometric evidence are listed. "Support" is the fraction of pairs in which the proposed partner is closer than IMGT's. Distances are medians.

### 4.1 Within-type relabelling

| Domain | IMGT → proposed | n | IMGT pair | proposed pair | support | Rule |
|---|---|---|---|---|---|---|
| Vλ A strand | 8→9, 9→10 (gap moves 10→8) | 60/58 | 4.7 / 3.3 Å | 1.2 / 0.9 Å | 98% | R3 |
| Vλ D strand | 83→81 (84→82) | 60 | 6.2 Å | 1.5 Å | 92% | R3 |
| TRBV D strand | 83→81 | 60 | 4.9 Å | 3.1 Å | 93% | R3 |
| TRBV C″ | 67→66, 68→67, 69→68 | 60 | 3.8–5.4 Å | 1.7–2.2 Å | 88–97% | R3 |
| VH C″–D | 70→71 | 60 | 2.8 Å | 2.4 Å | 77% | R3 |
| MHC-I α2 helix | 38→37, 39→40, 42→43, 57→56, 58→59, 62→61, 72A→72 | 59–60 | 2.7–6.0 Å | 2.1–3.3 Å | 65–100% | R3 |
| MHC-like α2 | 63→62, 72A→72 | 52–54 | 3.9–4.9 Å | 2.5–3.2 Å | 73–90% | R3 |
| MHC-II β1 | 42→40, 58→59, 63→62, 84→85 | 57–60 | 3.8–4.7 Å | 2.1–2.8 Å | 72–100% | R3 |
| MHC-II α1 | 63→62 | 60 | 2.9 Å | 2.3 Å | 95% | R3 |
| MHC-like α1 | 42→40 | 60 | 4.4 Å | 2.8 Å | 77% | R3 |
| Mouse IGHG1\*03 CH1 | whole domain (e.g. 15.1→17, 15.2→18, 45.4→80, 97→101) | 8–13 | 7.8–10.6 Å | 1.2–3.5 Å | 75–100% | fix the reference |

Earlier findings that the larger set re-confirms or extends:
- VH 82–84 sit where Vκ has 84 / 84A / 84B. This is the same light-chain/heavy-chain D-strand offset as pilot finding A2.
- The MHC-II α1 helix N-terminus (IMGT 59–61) has no MHC-I equivalent and becomes letter sites 54A–54E.
- TRDC's E strand is +1 relative to all other C domains (pilot finding B2). There are only 2 TRDC structures here, so it was not re-tested.

### 4.2 Cross-type harmonisation in the Ig frame

These come from class-consensus geometry after B/C/E/F core superposition. The core fits are 0.88 Å for C onto V (465 C domains) and 1.73 Å for FN3 onto V.

| IMGT label | Structural equivalent in the V frame | Δ (Å) | Proposal |
|---|---|---|---|
| C 1–8 | V 1–8 | 0.9–2.6 | keep |
| C 9–16 (A strand, ABED sheet) | none: V's A′ crosses to the other sheet | ≥ 4.4 | letter sites 8A–8H (R6) |
| C "CD strand" 45, 45.1–45.4 | **V C′ 50–54** | 0.7–2.3 | renumber as C′ 50–54 (R4) |
| C 81/82 | V 83/84 | 1.8–2.1 | C-domain gap is at 81–82 |
| C 98/99 | V 99/100 | 1.7–2.1 | +1 |
| C G 120–122 | V 121–123 | 0.7–2.0 | +1; V has a G-strand bulge at 120 |
| FN3 "D strand" 81–84 | **V C′ 53–57** | 0.4–1.7 | renumber as C′ (R4) |
| FN3 85 | V 83 | 1.2 | −2 |
| FN3 G 122–129 | V 118–126 | 0.4–2.3 | **−3/−4: IMGT's FN3 G strand is misregistered** |

### 4.3 SRCR and Apple

| Measure | SRCR (n = 19) | Apple (n = 5) |
|---|---|---|
| Recall, IMGT → new | 0.875 → 0.969 | 0.955 → 0.982 |
| Precision, IMGT → new | 0.783 → 0.968 | 0.903 → 0.957 |

The SRCR changes are concentrated in the 46–57 loop (50→56, 51–54→58–61, 55–57→63–64.1) and the 79–85 turn. With a single class there is no reference class, so these could not be pairwise-validated. They rest on consensus geometry from 19 structures.

## 5. Prototype implementation and evaluation

`tools/run_scheme2.py` implements R1–R10 end to end:
1. Build a class consensus by iterative TM-align, an order-preserving DP and Procrustes.
2. Filter anchors with R2.
3. Merge classes into a family frame (TM-align plus anchor DP, R5).
4. Merge the V, C and FN3 frames into the Ig frame on the B/C/E/F core.
5. Name class-specific sites with letters, using a global topological order (R6).
6. Number every domain against its own class model (R3, R7–R9).

On the 1,954 domains, measured with the independent pairwise TM-align metric:

| Family | Recall IMGT → new | Precision IMGT → new | Precision at anchors |
|---|---|---|---|
| Ig (V+C+FN3) | 0.696 → 0.718 | **0.597 → 0.740** | 0.800 |
| G | 0.906 → 0.877 | **0.753 → 0.902** | 0.932 |
| SRCR | 0.875 → 0.969 | **0.783 → 0.968** | 0.976 |
| Apple | 0.955 → 0.982 | 0.903 → 0.957 | 0.996 |

Selected class pairs:
- CH1×CH1: 0.70 → 0.975 precision, recall 0.69 → 0.99 (the IGHG1\*03 error is corrected automatically).
- CK×CH1: 0.68 → 0.97.
- Vκ×VH: 0.81 → 0.86.
- VH×VH: 0.91 → 0.94.

Anchor precision between V classes is 0.95–0.998.

**What did not work. Do not adopt these parts of the automatic output.**
- **TRAC.** Its atypical fold (TM ≈ 0.55 to other C domains) makes the class-model merge misregister it by 2. Almost all TRAC relabellings are contradicted pairwise (IMGT pair closer in 61–75% of pairs). TRAC needs hand-curated definitions.
- **CDR3 relabelling for Vα, Vβ, Vλ and V-like** (e.g. 108→107) is contradicted pairwise. IMGT's symmetric CDR3 stems are better, so keep IMGT CDR3 numbering. Vβ 109/110→108/109 is only weakly supported (67–70%).
- **Cross-class recall dips by 0.01–0.04 between V classes, and by 0.03 in G.** The cause is class-specific letter sites that a stricter equivalence rule would have unified.
- **The Ig frame needs 166 letter sub-sites.** Most come from C2/I-set, TRAC and TRBC loops. A production scheme should only create letter sites for classes with ≥ 30 structures and fold the rest into loop numbering.
- Of 97 testable per-class changes, 38 are supported (≥ 60%), 39 are contradicted (< 40%) and the rest are inconclusive. The contradicted ones are almost all TRAC, CDR3 stems and one Vα CDR2 artifact. §4.1 lists only the supported ones.

## 6. Recommendation

1. **Adopt rules R1–R10 as the definition of the scheme.** The prototype shows they produce a consistent, totally ordered numbering for every domain tested, and raise same-number⇒same-place precision by 15–19 points in Ig, G and SRCR.
2. **Apply the vetted changes in §4.1–4.2.** Leave the remaining IMGT positions unchanged, including CDR3 and TRAC, until they can be curated.
3. **Fix IMGT's reference directory** for mouse IGHG1\*03, TRDC (E strand) and the FN3 G strand. DGA copies reference gaps onto every query, so these errors propagate to every structure whose best hit is that reference.

## Files

| Path | Contents |
|---|---|
| `results_large/validation.json` | per-change pairwise test |
| `results_large/report_tables.json` | every IMGT→new change per class |
| `results_large/renumbered.json.gz` | IMGT and proposed numbering for all 1,954 domains |
| `results_large/class_models/` | anchor consensus models |
| `results_large/frames.json` | frames and letter maps |
| `results_large/selected_entries.json` | the entry list |

Pipeline order: `select_big.py` → `thread_big.py` → `run_scheme2.py` → `report_scheme.py` → `validate_changes.py`.
