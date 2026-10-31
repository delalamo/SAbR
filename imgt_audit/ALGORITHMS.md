# Task B: which algorithm should assign pan-Ig numbers?

This compares ways to place the pan-Ig numbering (`PANIG.md`) onto a new structure without any prior labels. The candidates are TM-align label transfer, SAbR-style embedding alignment, and several new structure-profile methods.

**Recommendation: M7, an adaptive consensus-frame matcher.** It is the most accurate on clean structures and as honest as the oracle that defined the labels. It is robust to missing residues, truncation, chain context and moderate noise. It is 6–7× faster than template TM-align, and it needs no neural network.

Code is in `panig/bench/`. A ready-to-use tool is `panig/number_structure.py`.

## Algorithms compared

All methods output anchor claims (residue → anchor number). The pan-Ig fill rule then produces the full numbering. Frames, templates and the embedding profile were built from the **train split only**. Results are on the held-out split: 338 domains (193 V, 106 C1, 23 C2/I, 10 V-like, 6 FN3).

| | Method | What it does |
|---|---|---|
| M1 | TM-align, best template | TM-align the query to 109 labelled train templates (up to 5 per class); copy anchors from the best-TM template for aligned pairs within 3.5 Å |
| M2 | TM-align, top-5 vote | As M1, but TM-weighted votes from the 5 best templates, resolved by an order-preserving DP |
| M3 | SAbR embedding | SAbR's MPNN embeddings with a softmax log-odds profile retrained on pan-Ig anchors; SAbR's Smith–Waterman alignment |
| M4 | Consensus-frame matching | New. TM-align the query to each class *consensus frame*; refine on core sites; order-preserving DP on Cα distance × Cβ-direction agreement; best structural score wins |
| M5a | Embedding seed + frame | New. Like M4, but the superposition is seeded from M3's anchors instead of TM-align |
| M5b | Fused DP | New. Like M5a, with the DP score multiplied by embedding compatibility |
| M6 | Frame matcher v2 | New. M4 with the Cβ-direction term at half weight |
| **M7** | **Adaptive frame matcher** | **New.** M4, but switches to M6's half Cβ weight when the backbone is irregular (std of consecutive Cα–Cα distances > 0.3 Å; clean structures ≤ 0.17 Å, 0.5 Å noise ≥ 0.45 Å) |

A *consensus frame* is the mean Cα position and Cβ direction of each pan-Ig site over many domains of one class, all in one common coordinate system. Matching against a consensus removes the template-specific idiosyncrasies that single-template TM-align inherits.

## 1. Accuracy on clean held-out structures

Anchor precision / recall against the oracle labels:

| Method | All | V | V-like | C1 | C2/I | FN3 |
|---|---|---|---|---|---|---|
| M1 TM-align best | 0.982 / 0.981 | 0.991 / 0.994 | 0.955 / 0.891 | 0.986 / 0.990 | 0.900 / 0.855 | 0.842 / 0.937 |
| M2 TM-align top-5 | 0.980 / 0.987 | 0.989 / 0.995 | 0.943 / 0.956 | 0.984 / 0.991 | 0.911 / 0.898 | 0.838 / 0.933 |
| M3 SAbR embedding | 0.985 / 0.899 | 0.994 / 0.947 | 0.956 / 0.822 | 0.985 / 0.853 | 0.910 / 0.623 | 0.763 / 0.609 |
| M4 frame match | 0.992 / 0.994 | 0.999 / 0.999 | 0.986 / 0.991 | 0.994 / 0.995 | 0.940 / 0.943 | 0.838 / 0.979 |
| M5a emb seed + frame | 0.993 / 0.994 | 0.999 / 0.999 | 0.986 / 0.991 | 0.994 / 0.995 | 0.939 / 0.942 | 0.914 / 0.979 |
| M5b fused DP | 0.993 / 0.993 | 0.999 / 0.998 | 0.985 / 0.984 | 0.994 / 0.992 | 0.943 / 0.937 | 0.921 / 0.979 |
| M6 frame v2 | 0.989 / 0.994 | 0.997 / 0.999 | 0.975 / 0.985 | 0.993 / 0.994 | 0.920 / 0.935 | 0.838 / 0.979 |
| **M7 adaptive** | **0.992 / 0.995** | **0.999 / 0.999** | **0.986 / 0.991** | 0.993 / 0.994 | **0.946 / 0.950** | 0.838 / 0.979 |

Full-label agreement with the oracle (all residues, after the fill):

| Method | Agreement |
|---|---|
| M1 | 0.962 (C2/I 0.788, FN3 0.761) |
| M2 | 0.967 (C2/I 0.845, FN3 0.746) |
| M3 | 0.854 (C2/I 0.547, FN3 0.424) |
| M4 | 0.987 (C2/I 0.914, FN3 0.781) |
| M7 | 0.988 (C2/I 0.926, FN3 0.781) |

**Caveat: the oracle shares a lineage with M4–M7.** The oracle was made by the same kind of frame matching, except that it knew the domain class and was initialised from prior core labels. So agreement with the oracle favours M4–M7. Two checks guard against this.

1. **An oracle-free honesty test (§2).**
2. **Disagreements.** Where the label-free methods disagree with the oracle, inspection shows the oracle is often the one that is wrong:
   * **3bik (IgSF C-like).** All label-free methods checked (M1–M5) agree with each other, and all are shifted from the oracle by one β-strand register (±2 residues). The label-free assignment is locally more honest: 0.965 vs 0.947.
   * **2rbl (FN3).** The oracle has only 4 anchors; TM-initialised matching claims 35, with local honesty 0.977.

## 2. Oracle-free honesty

This applies the local leave-one-out test of `PANIG.md` to each method's own claims, over 4,012 held-out pairs with TM ≥ 0.5, of which 2,220 are cross-type. For each claimed anchor shared by both domains, the pair is superposed on the *other* shared anchors within ±12 positions, and the claim must land within 3 Å.

| Method | Anchors / domain | Within-type | Cross-type |
|---|---|---|---|
| M1 TM-align best | 66.6 | 0.9859 | 0.9646 |
| M2 TM-align top-5 | 67.1 | 0.9838 | 0.9642 |
| M3 SAbR embedding | 60.9 | **0.9892** | 0.9688 |
| M4 frame match | 66.8 | 0.9882 | 0.9682 |
| M5b fused DP | 66.6 | 0.9886 | **0.9705** |
| M6 frame v2 | 67.0 | 0.9867 | 0.9661 |
| **M7 adaptive** | 66.8 | 0.9881 | 0.9681 |
| Oracle | 66.7 | 0.9885 | 0.9697 |

All methods are honest; the scheme's anchors are robust to the assignment method. The differences:

* **TM-align transfer** is the least honest: it copies template-specific register errors.
* **SAbR embeddings** are precise but conservative. They claim 9% fewer anchors and miss many C2/I, FN3 and V-like anchors (recall 0.61–0.82).
* **M7** equals the oracle without knowing the class or any prior labels.

## 3. Robustness

The test uses 169 held-out domains, each perturbed and renumbered. Each cell gives anchor precision / recall against the oracle, then agreement with the method's own clean output.

| Perturbation | M1 TM best | M2 TM top-5 | M3 SAbR | M4 frame | M5b fused | **M7 adaptive** |
|---|---|---|---|---|---|---|
| None (domain only) | .980/.980 · 1.00 | .977/.985 · 1.00 | .981/.914 · .969 | .988/.993 · 1.00 | .990/.990 · .998 | **.988/.993 · 1.00** |
| Gaussian noise 0.5 Å | .980/.978 · .996 | .978/.984 · .996 | .646/.445 · .624 | .980/.977 · .985 | .943/.915 · .938 | **.979/.986 · .989** |
| Gaussian noise 1.0 Å | **.978/.960 · .979** | .977/.971 · .984 | .225/.180 · .384 | .932/.851 · .897 | .627/.425 · .593 | .952/.925 · .944 |
| Termini −7/−7 | .978/.980 · .992 | .976/.985 · .996 | .954/.766 · .870 | .987/.992 · .999 | .987/.986 · .994 | **.988/.993 · .999** |
| Internal break (5 missing) | .979/.982 · .993 | .978/.986 · .998 | .972/.899 · .952 | .984/.989 · .995 | .989/.988 · .995 | .984/.989 · .995 |
| ±30 flanking chain residues | .975/.980 · .997 | .973/.986 · .998 | .978/.921 · .979 | .984/.993 · 1.00 | .986/.991 · .999 | **.984/.993 · 1.00** |

* **The MPNN embeddings are very noise-sensitive.** At 0.5 Å i.i.d. backbone noise, SAbR-style alignment keeps only 45% of anchors. Every hybrid that trusts the embedding (M5a/b) inherits this. On clean inputs the embedding barely helps: it raises FN3 precision on 6 domains.
* **The Cβ-direction term** is valuable on clean structures, but it degrades under heavy noise, because virtual Cβ is computed from noisy N/CA/C. M7 detects irregular backbones and halves that term. This raised recall at 1 Å from 0.85 to 0.93 with no loss on clean inputs.
* **Template TM-align stays the most tolerant of extreme (1 Å i.i.d.) noise.** Its d0-scaled scoring is more forgiving than a 3.5 Å match cutoff. Wider cutoffs for M7 recover some of this (σ 2, dmax 4: R 0.957) at a small clean-precision cost (`bench/tune8.py`), so they were not adopted. Real coordinate error, in crystal structures or models, keeps local backbone geometry intact. That i.i.d. per-atom noise is therefore a stress test, not a typical case.

## 4. Speed

Single core, median over 57 held-out domains:

| Method | ms / domain |
|---|---|
| M1 / M2 TM-align vs 109 templates | 1,693 / 1,767 |
| M3 SAbR (MPNN encode + alignment) | 166 + 24 = 190 |
| M5a/b (encode + alignment + frame matching) | about 257 |
| M4 / **M7** frame matching (19 frames; one TM-align each, about 18 ms) | 257 / **264** |

Template-transfer cost scales linearly with the number of templates; fewer templates would be faster but less accurate across classes. M7's cost is dominated by its 19 frame TM-alignments.

## 5. Things that did not work

* **A single TM-align to the pooled frame**, reusing it for all class frames because they share coordinates, saves about 90% of the time. But accuracy drops: clean P/R 0.93/0.95, against 0.98/0.99 for per-frame initialisation. The pooled V-centred frame is a poor initial model for C1, C2/I and FN3.
* **Seeding the superposition from embedding anchors** (M5a) is fast and accurate on clean data, but fails when the embedding fails: under noise, and on unusual domains such as the 2rbl FN3 domain. Picking the best of the embedding and TM initialisations by score did not beat TM initialisation alone.
* **Fusing embedding probabilities into the DP** (M5b, and λ sweeps in `tune6.py`/`tune7.py`) gives at most +0.002 honesty on clean data and hurts under noise.

## 6. Whole chains

`panig/number_structure.py` runs M7 greedily on a whole chain:

1. Match the best domain.
2. Mask its span.
3. Repeat while at least 30 anchors are claimed.
4. Split linkers at their midpoint and number each domain with the pan-Ig fill.

On 60 random dataset chains (75 domains: V, C1, C2/I, V-like and FN3), every domain was found, with anchor P/R 0.985 / 0.986 against the oracle. These chains are part of the data the shipped frames were built from, so this checks domain finding and plumbing, not generalisation.

## 7. Bottom line

* **Better than TM-align:** consensus-frame matching (M4, and M7 as tuned) beats template TM-align on clean accuracy (anchor recall 0.995 vs 0.981–0.987), honesty and speed (6–7×). It loses only under extreme synthetic noise.
* **Better than SAbR embeddings** for structure input: much higher recall (0.995 vs 0.899) and far more robust to noise. SAbR remains the right tool when only an embedding path is wanted, or for speed in batch use (about 190 ms).
* **What actually matters** is matching to class consensus frames instead of individual templates, with per-class initialisation and a structural-fit model selection. Embeddings help at most marginally once the structure is available.
