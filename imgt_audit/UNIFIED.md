# One embedding set for numbering every immunoglobulin-fold domain

This turns the structure-consistent scheme from `PROPOSAL.md` into a single SAbR reference. It is shipped as `src/sabr/assets/unified_reference.npz` and used via `scheme="unified"` (API) or `-n unified` (CLI).

The test here: can numbering inferred from embeddings alone recover the structure-derived numbering, and does the convention hold up?

## 1. How it works

**Reference.** One matrix: 292 rows × 64 dimensions, one row per structural anchor of the unified Ig frame, in global chain order.

**Training.** Each row is trained as a softmax classifier over all anchors plus a "loop / not an anchor" class, using SAbR residue embeddings of 1,727 domains labelled by the structural consensus. The stored vector is `w_anchor − w_loop`, so a dot product with a residue embedding is the log-odds that the residue occupies that anchor rather than a loop. Details:
- Classes are balanced, so VH does not dominate.
- L2 = 1e-6 and a scale factor of 3 were fixed on a pilot subset before the final held-out evaluation.
- 184 chains from 26 non-Ig Pfam families (36,662 residues) are added to the loop class, so that non-Ig environments score negative.

**Alignment.** SAbR's existing encoder and affine Smith–Waterman are used unchanged. A domain simply skips the anchors it lacks, for example a C domain skipping V's C″.

**Domains.** The whole chain is aligned. The alignment is split wherever consecutive anchors are more than 45 residues apart, the largest piece is kept, and the remaining segments are searched again. A segment counts as a domain if it has ≥ 30 anchors and either score ≥ 400 or ≥ 50 anchors.

**Loops.** These are numbered by a class-agnostic length rule:
- loop residues take the integers between the flanking anchors, filled from both ends with gaps at the centre;
- surplus residues become N-attached decimals (`111.3`);
- letters (`8A`) are never loop slots;
- termini count outward, so 0 and negative numbers can occur.

**Labels.** Every domain's labels sort in chain order (8 < 8.1 < 8A < 8A.1 < 9). Further domains in a chain are offset by 1000. In PDB/mmCIF a label becomes a residue number plus insertion code: `8A.1` → (8, `A1`), `111.3` → (111, `3`).

## 2. Recovery of the structural numbering

The held-out reference was trained on 1,389 domains. The test set is 338 domains from PDB entries never seen in training (an 80/20 split by entry).

| Class | n | Anchor precision | Anchor recall | Labels exact |
|---|---|---|---|---|
| **All** | 338 | **0.960** | **0.898** | **0.895** |
| Vκ | 57 | 0.989 | 0.951 | 0.974 |
| VH | 72 | 0.968 | 0.908 | 0.925 |
| Vλ | 15 | 0.981 | 0.927 | 0.927 |
| Vα | 22 | 0.983 | 0.934 | 0.919 |
| Vβ | 24 | 0.943 | 0.887 | 0.896 |
| V-like | 10 | 0.915 | 0.890 | 0.875 |
| CK | 16 | 0.999 | 0.942 | 0.932 |
| CL | 9 | 0.990 | 0.946 | 0.950 |
| β2m | 5 | 0.982 | 0.923 | 0.936 |
| TRBC | 6 | 0.996 | 0.921 | 0.924 |
| MHC-II α2/β2 | 7 | 0.963 | 0.909 | 0.898 |
| MHC-I α3 | 11 | 0.966 | 0.891 | 0.887 |
| TRAC | 10 | 0.985 | 0.862 | 0.855 |
| CH1 | 35 | 0.957 | 0.868 | 0.845 |
| FN3 (FN/TNC) | 5 | 0.892 | 0.697 | 0.664 |
| C2/I-set IgSF | 23 | 0.705 | 0.657 | 0.604 |

Exact-label agreement by region:

| Region | Agreement |
|---|---|
| C/C′ | 0.95 |
| E/F | 0.94 |
| B | 0.93 |
| G | 0.93 |
| A/A′ | 0.90 |
| CDR1 | 0.90 |
| C″/D | 0.84 |
| CDR2 | 0.83 |
| CDR3 | 0.78 |

Most residual error is in variable-length loops, where a one-anchor difference shifts the fill.

**Is the convention preserved when it is produced automatically?** I used the same pairwise TM-align test as the audit, over 4,868 held-out pairs: same number ⇒ same place within 3 Å.

| Numbering | Precision | Recall |
|---|---|---|
| IMGT | 0.587 | 0.686 |
| Structural target | 0.726 | 0.706 |
| Predicted from embeddings | **0.709** | **0.700** |

The embedding-predicted numbering keeps almost all of the structural-consistency gain over IMGT.

## 3. Whole chains and specificity

- **Domain detection** on held-out chains: 328 of 338 annotated domains found (97%), with 0.895 exact labels on those found. The 10 misses are all C2/I-set.
- **The 121 extra detections are real.** They are domains that exist in the chains but were absent from the annotated test subset. 108 carry both disulfide cysteines at 23 and 104; the rest are TRAC domains, which lack the canonical core.
- **Non-Ig folds.** Testing on 88 chains from 13 non-Ig families withheld from training (GFP, PDZ, SDR, SH3, TNF, WD40, β-lactamase, ferritin, globin, legume lectin, lysozyme, thioredoxin, ubiquitin), the detection rule admits 1.1% (one SH3 β-barrel) while keeping 97.9% of Ig domains.
- **Before and after negatives.** Without negatives, myoglobin, haemoglobin and λ repressor scored like Ig domains. With the shipped reference, none of 12 non-Ig control proteins yields a domain, including the repository's ubiquitin and lysozyme controls. SAbR's H/K/L alignment cannot reject these (a known xfail in its own test suite).

## 4. Robustness

**Missing residues** (anchor precision / recall):

| Condition | Precision | Recall |
|---|---|---|
| Intact | 0.960 | 0.898 |
| 5 residues removed from each terminus | 0.960 | 0.906 |
| 3 residues removed from the middle of every long loop | 0.961 | 0.898 |

**Coordinate noise.** Gaussian noise was added per backbone atom and the chain re-encoded.

| σ (Å) | Anchor precision | Anchor recall | Labels exact |
|---|---|---|---|
| 0 | 0.960 | 0.898 | 0.894 |
| 0.1 | 0.959 | 0.876 | 0.883 |
| 0.25 | 0.928 | 0.700 | 0.802 |
| 0.5 | 0.575 | 0.273 | 0.380 |
| 1.0 | 0.115 | 0.075 | 0.081 |

This is the main weakness. On the same 90 held-out V domains, SAbR's H/K/L IMGT pipeline reproduces its clean-structure numbering at 1.000 (σ = 0.25) and 0.997 (σ = 0.5), against 0.867 and 0.395 for the unified reference. The cause is that the anchor log-odds were trained only on clean embeddings. A noise-augmented reference is evaluated in §6.

**Unseen domain types.** For each group, the reference was retrained without that class, then used to number it.

| Held-out class | Labels exact, unseen | Labels exact, seen |
|---|---|---|
| γδ V | 0.836 | 0.857 |
| MHC-II α2/β2 | 0.807 | 0.898 |
| V-like | 0.767 | 0.875 |
| C2/I-set | 0.573 | 0.604 |
| FN3 | 0.262 | 0.669 |
| TRAC | 0.162 | 0.855 |

Classes built from shared integer sites transfer. Classes that need their own letter sites (TRAC, FN3) cannot be inferred without examples. This is a limit of any single reference.

**Reproducibility.** 103 domain sequences occur in two or more independent PDB entries (649 copies in total, mostly CK, CL, CH1, TRAC, TRBC).
- 94.3% of residue labels agree with the consensus labelling for that sequence.
- Only 23% of copies are numbered identically end to end.

The disagreements concentrate where the frame has near-duplicate letter sites contributed by different C classes: for example 12B vs 12, 16F vs 16G, 54 vs 67A, and the DE loop 84I/84J vs 84D.x. Those sites are structurally almost equivalent, so the choice between them flips with small structural differences.

## 5. Limitations and next steps

1. **Merge near-duplicate letter sites.** Letter sites from different classes that lie within about 4 Å of each other in the frame should be merged into one site, before relabelling and retraining. This directly targets the reproducibility gap above and also shrinks the reference.
2. **Noise robustness.** See §6.
3. **C2/I-set and FN3.** These remain weak because the structural target itself is noisy for these classes (only 103 and 27 domains, and heterogeneous). They need more structures and class-level consensus models.
4. **Out of scope:** MHC groove (G), SRCR and Apple domains are not Ig folds and are not covered by this reference.
5. **An existing SAbR limitation surfaced along the way:** `extract_chain` rejects chains containing any residue with an incomplete backbone. This affected 21 of 1,366 chains in this dataset.

## 6. Noise-augmented reference: evaluated, not shipped

I retrained the held-out reference with an extra copy of every training chain, re-encoded after per-atom Gaussian noise (σ = 0.25 or 0.5 Å, alternating by chain). It was then compared with the shipped, clean-trained reference on 130 held-out domains spanning all classes.

| Reference | Labels exact, clean | σ = 0.25 Å | σ = 0.5 Å | Same as clean-structure numbering (0.25 / 0.5) |
|---|---|---|---|---|
| Clean-trained (shipped) | 0.881 | 0.782 | 0.402 | 0.814 / 0.407 |
| Noise-augmented | 0.858 | 0.817 | 0.653 | 0.862 / 0.663 |
| SAbR IMGT (51 V domains) | n/a | n/a | n/a | 0.975 / 0.948 |

Across all 338 held-out domains, augmentation lowers clean accuracy from 0.895 to 0.877. It also lowers real-world reproducibility across independent structures of the same sequence: residue agreement drops from 94.3% to 93.2%, and copies numbered identically from 23% to 18%.

Synthetic per-atom noise distorts bond geometry far more than genuine structural variation does, so the clean-trained reference is shipped. Augmentation is the right choice only for heavily perturbed coordinates, and could be offered as a second asset in the way SAbR offers noise-level H/K/L references.

## Reproduction

`imgt_audit/unified/tools/` contains the scripts, run in this order:

| Step | Scripts |
|---|---|
| Embed chains and negatives | `prep.py`, `prep_neg.py`, `prep_aug.py` |
| Build the reference | `build_reference.py` |
| Main evaluation | `exp_final2.py`, `exp_chain.py`, `exp_neg1.py` |
| Robustness | `exp_robust.py`, `exp_loco.py`, `exp_consistency.py`, `exp_noise_baseline.py`, `exp_aug.py` |

Labels come from `imgt_audit/results_large/renumbered.json.gz`.
