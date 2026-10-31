# Pan-Ig numbering: one honest IMGT-compatible scheme for every Ig-fold domain

This is task A of the follow-up request. The goal was a numbering that:

1. gives every residue of every Ig-fold domain a number;
2. only claims "same number ⇒ same structural place" where that is true across domain types;
3. numbers everything else with a deterministic anchor-derived rule, the way IMGT already numbers CDRs;
4. stays as close to IMGT as possible.

Code is in `panig/tools/`. Reference frames are in `panig/panig_reference.npz`, and the numbers are in `panig/results/`.

## 1. The idea

IMGT numbers every domain type on the V-domain frame 1–128. For C, C2/I-set and FN3 domains it does this by sequence-profile alignment to a type-specific ruler. That causes the trouble documented in `REPORT.md` and `PROPOSAL.md`: the same number means different places in different domain types, especially around C'/C''/D, at strand ends and in C-domain loops.

The pan-Ig scheme keeps the IMGT V frame and the IMGT CDR fill rule. It changes two things.

* **Two kinds of number, explicitly.**
  * **Anchors** are positions where "residue numbered *n*" is a verified structural claim, checked across domain types (§3).
  * **Positional** numbers fill everything between anchors deterministically (§2). They make no cross-domain structural claim, exactly like IMGT CDR3 numbers 111/112 today.
* **Structure decides anchors; sequence does not.** A residue receives anchor number *n* only if, after core superposition onto the consensus Ig frame, it sits within 3.5 Å of site *n* with a compatible Cβ direction. An order-preserving dynamic program enforces this (§4). Domains of any type use the same 128-position frame. A class that lacks a strand (C1 has no C'/C''; FN3 lacks parts of A and D) simply leaves those anchor numbers unused, the way IMGT leaves gaps.

## 2. Rules

**Frame.** The IMGT V frame is 1–128, a topological superset of V, C1, C2, I-set and FN3. Strand positions are the IMGT V positions:

| Strand | Positions |
|---|---|
| A | 1–15 |
| B | 16–26 |
| C | 39–46 |
| C' | 47–55 |
| C'' | 66–74 |
| D | 75–84 |
| E | 85–96 |
| F | 97–104 |
| G | 118–128 |

**Anchors (82).** These are the residues where the number is a structural claim:

```
3-7, 9, 11-26, 38-48, 50-54, 56, 65-67, 69, 72, 74-81, 84-94, 96-97, 100-101, 103-106, 116-126
```

They include the conserved landmarks: Cys 23, Trp 41, hydrophobic 89, Cys 104 and J-Trp/Phe 118.

**Segments.** Each maximal run of non-anchor numbers is a positional segment with one fixed apex pair.

| Segment | Apex | Note |
|---|---|---|
| 1–2 | (2) | 1 reserved for N-terminal overflow 1.k |
| 8 | 8 | |
| 10 | 10 | |
| 27–37 | 32/33 | CDR1, same as IMGT |
| 49 | 49 | |
| 55 | 55 | |
| 57–64 | 60/61 | CDR2, same as IMGT |
| 68 | 68 | |
| 70–71 | 70/71 | |
| 73 | 73 | |
| 82–83 | 82/83 | |
| 95 | 95 | |
| 98–99 | 98/99 | |
| 102 | 102 | |
| 107–115 | 111/112 | CDR3, same as IMGT |
| 127–128 | 127/128 | |

**Fill rule.** This is the IMGT CDR rule, applied to every gap between consecutive matched anchors.

* Residues take the free (non-anchor) numbers alternately from both ends, front first; the front takes the odd residue. Unused numbers become gaps at the centre.
* If there are more residues than free numbers, the overflow goes at the apex pair nearest the span centre:
  * left insertions are `L.1, L.2, …` (after L);
  * right insertions are `R.k … R.1` (before R);
  * the first extra residue goes to the right, as in IMGT CDR3 (111.1 / 112.1).
* If a span has no free number at all, the residues become insertions after the left anchor (`A.1, A.2, …`).
* **Strict.** The fill never reuses an anchor number, so an anchor number is always a structural claim.
* **N-terminus.** Free numbers count down to 1, then `1.1, 1.2, …` (IMGT C-domain style).
* **C-terminus.** Free numbers count up to 128, then 129, 130, ….

Every label therefore has one context-free sort key: `(n, ±k)`, with the sign set by which side of its segment apex *n* lies on.

**IMGT compatibility.**

* For V domains, 93.9% of residue labels are identical to IMGT/DomainGapAlign.
* The CDR1/2/3 fill reproduces ANARCI's IMGT CDR numbering exactly for every length from 1 to 25. This was tested directly against ANARCI.
* The remaining V differences arise wherever the structural anchor match disagrees with IMGT's sequence-based placement. Positional numbers next to such a residue then shift as well.

## 3. How the anchors were chosen

1. **Consensus V frame.** Average Cα and Cβ-direction coordinates of IMGT sites 1–128 over 1,029 canonical V domains (VH, Vκ, Vλ, Vα, Vβ, Vγ, Vδ). The domains are superposed on the shared core (18–25, 39–44, 86–91, 99–104) and refined over three rounds of structural matching.
2. **Per-class site table.** Match every domain (V, V-like, C1 classes, C2/I-set, FN3) to the frame and record each site's per-class occupancy and spread.
   * The candidate set A0 is every site outside the CDR interiors with occupancy ≥ 0.5 in at least one of V/C1/C2-I and spread ≤ 1.9 Å, plus site 26.
   * Site 26 is added because without it the CDR1 apex shifts to 31/32 instead of IMGT's 32/33.
3. **Leave-one-out local test, the honesty filter.**
   * For each residue holding anchor *n*, superpose its domain onto each other domain using only the *other* shared anchors within ±12 positions (at least 6 required).
   * The claim counts as honest if the partner's residue numbered *n* is within 3 Å.
   * An anchor is dropped if within-type precision is below 0.90, or cross-type precision is below 0.80 with n ≥ 200.
   * 19 candidates failed: 1, 2, 8, 10, 27, 49, 55, 68, 70, 71, 73, 82, 83, 95, 98, 99, 107, 127, 128. They are strand ends and loop-adjacent spots whose register varies by class, and they became positional.
4. **Assignment model.** Each domain is superposed on the consensus frame of its own class, all expressed in the common V frame, and then matched at dmax 3.5 Å. If a class frame is unavailable, the pooled frame is used.

Why a local test? Global pairwise TM-align caps cross-type precision near 0.55 even for core anchors. Whole-domain superposition of a V and a C1 domain cannot bring both sheets within 3 Å at once. The local test asks the right question: is this residue in the same place relative to its structural neighbours?

## 4. Results

The dataset is 1,727 domains in 5 types. Precision means the fraction of same-numbered residue pairs within 3 Å after superposition. Recall means the fraction of structurally equivalent pairs that share a number.

| Measure | Pan-Ig | IMGT |
|---|---|---|
| Global pairwise P / R (TM-align, TM ≥ 0.5) | **0.686 / 0.754** | 0.595 / 0.694 |
| Within-type P / R | **0.869 / 0.934** | 0.817 / 0.918 |
| Cross-type P / R | **0.466 / 0.517** | 0.336 / 0.397 |
| Local test, anchor claims (within / cross) | **0.989 / 0.970** | n/a |
| Local test, all residues (within / cross) | **0.880 / 0.739** | 0.835 / 0.512 |
| V labels identical to IMGT | 93.9% | 100% |
| Insertion rate V / C1 / C2-I / FN3 | 1.9 / 25.6 / 11.3 / 23.4% | 0.7 / 18.1 / 12.2 / 5.1% |

**Type pairs** (P/R, pan-Ig vs IMGT):

| Type pair | Pan-Ig | IMGT |
|---|---|---|
| C1~C1 | 0.798 / 0.849 | 0.681 / 0.778 |
| C2/I~V | 0.783 / 0.869 | 0.563 / 0.611 |
| FN3~V | 0.711 / 0.755 | 0.414 / 0.557 |
| FN3~FN3 | 0.910 | 0.955 |
| C1~FN3 | ~0.07 | ~0.07 |

FN3~FN3 is slightly worse than IMGT: IMGT's FN3 ruler is internally consistent within FN3 but inconsistent with everything else. C1~FN3 is equally poor for both because those two folds superpose badly as wholes.

**Anchors claimed per domain** (mean of 82):

| Type | Anchors claimed | Anchors usually absent (claimed by < 50%) |
|---|---|---|
| V | 78.8 | 72 |
| V-like | 67.0 | 7, 45–48, 69, 84, 120 |
| C2/I | 57.5 | C'/C'' region 45–48, 56–77, 84–85 |
| C1 | 48.0 | A' 9–16, C'/C''/D 44–76, 84, 92–97, G-end 120–126 |
| FN3 | 46.2 | A, C', C''/D, E and G regions |

The absent anchors are absent structurally, not by failure. C1 domains have no C'' strand, so they never use 66–74. This is the honest version of IMGT's C-domain gaps.

**Reproducibility.** 157 sequences occur in several structures (2,377 copies). Residue labels agree across copies 98.9% of the time (the previous letter scheme: 94.3%). 69.5% of copies are numbered identically end to end (previously 23%).

**Leave-one-class-out.** To test generalisation, the anchor set was re-derived without the class, and the held-out class was numbered with the pooled frame only.

| Held-out class | n | Anchors (shared with full) | Labels equal to full scheme | Overall local precision |
|---|---|---|---|---|
| TRAC | 60 | 83 (82) | 0.827 | 0.989 / 0.970 |
| FN3 | 25 | 81 (81) | 0.809 | 0.989 / 0.970 |
| C2/I-set | 103 | 82 (81) | 0.901 | 0.989 / 0.969 |
| V-like | 53 | 82 (79) | 0.891 | 0.989 / 0.970 |
| γδ V | 12 | 81 (81) | 0.996 | 0.989 / 0.970 |

The anchor set is stable when a class is removed (79–82 of 82 shared across five held-out classes). The local precision of every claim stays at 0.989 / 0.970. The 10–19% label differences come from using the pooled frame instead of a class frame, mostly in loops; the anchors themselves still match.

## 5. What remains imperfect

* **Insertion rates in C1 and FN3** are higher than IMGT's. IMGT hides loop-length variation by assigning sequence-aligned integers that make false structural claims. The pan-Ig scheme refuses those claims, so the variation shows up as insertions at the apex. This is the intended trade-off, but it makes C1/FN3 numbers busier.
* **Cross-type global precision** stays below 0.5. The ceiling is set by whole-domain superposition, not by the numbering: the local test shows 0.97 for anchor claims.
* **The assignment depends on structure.** Pan-Ig numbering needs coordinates (predicted or experimental). For sequence-only use, the SAbR embedding route (`unified.py`) would need retraining on these labels. Task B looks at better assignment algorithms.

## 6. Files

* `panig/tools/pan.py`: consensus frame, structural matching (Kabsch + order-preserving DP), site statistics and class frames.
* `panig/tools/number2.py`: the `Scheme` class, covering segments, apices, the fill rule and sort keys.
* `panig/tools/number.py`: the original IMGT-fill implementation, validated against ANARCI.
* `panig/tools/score.py`, `local_test.py`, `final_eval2.py`: evaluation.
* `panig/tools/sweep*.py`, `anchor_quality.py`, `loco_a.py`: parameter sweeps, the anchor honesty filter and leave-one-class-out.
* `panig/panig_reference.npz`: the pooled and per-class consensus frames (site names, Cα coordinates, Cβ unit vectors) and the anchor list.
* `panig/results/final_anchors.json` and `final_eval2.json`.
