# Sequence and germline novelty of the erroneous antibodies (lens: seq_novelty)

**Question.** Were the antibodies with first-occurrence register errors further from anything previously solved (no
good template)? Did they carry first-time germlines, or rare residues exactly at the error location?

**Short answer.**
* **Not novel overall.** Once you account for two things, they were no further from previously solved antibodies, and
  their germlines were not first-time ones:
  * whether the same antibody had been solved before, which is partly circular with the "first-occurrence" definition;
  * species (nanobodies, bovine, rabbit).
* **The deposited sequence itself is often unusual at the error spot.** These are the location-specific links that
  hold up:
  * **SEQRES truncated at the strand that slips.** The sequence starts after IMGT 1 in mode A errors. It ends inside
    FR4 in mode D/E errors.
  * **A residue at a rarely used IMGT column inside the run window.** Examples: an extra N-terminal residue
    ("EVQLLEQSG…"), an atypical CDR2 length, a bovine ultralong CDR3.
  * **Mode E (FR4 bulge) antibodies use V genes rarely or never seen before in the PDB.**

## Data built
* **Reference set (all resolutions):**
  * ANARCI IMGT numbering of every PDB protein sequence that passed a regex prefilter: 52,285 unique sequences, with
    99.8% sensitivity on the study chains.
  * **11,837 unique antibody V domains** (H 6,662, κ 3,841, λ 1,334) from 11,973 entries.
  * Release dates come from `cite_all.jsonl`.
  * Germlines were assigned with a vectorised copy of ANARCI's identity rule: human/mouse for genes, all species for
    the species call.
* **Measured for each of the 15,865 chains,** against domains released *before* the entry's deposition:
  * the closest prior V domain of the same chain type: whole V, framework, and per IMGT region (NT 1–10,
    FR1…FR4, CDR1–3), counting any antibody, or excluding the same antibody (CDR3 identical and ≥95% V identity);
  * the identity of the closest prior CDR3 of the same length;
  * prior PDB entries sharing the V gene, the J gene, the V–J pair, the CDR length combinations, V + CDR1/2 lengths,
    and J + CDR3 length;
  * somatic mutations per region;
  * rare residues (<1% at that IMGT position) and residues at rarely occupied columns. These are counted in the mode
    windows and at canonical, Vernier and VH–VL interface positions, mapped from Kabat to IMGT empirically;
  * nanobody hallmarks, a humanisation signature, a PDB-profile "typicality" log-likelihood, and SEQRES termini.
* **Units:** one per antibody per paper, represented by the earliest-deposited chain of the unit.
* **Comparison:** cryo-EM first-occurrence error units (183) vs clean cryo-EM units (no disagreement run of any kind,
  4,036), per mode, with controls of the same chain types.
* **Models:**
  * **M1:** deposit-year bin + resolution + chain type.
  * **M2 (primary):** M1 + nanobody + non-human/mouse species + SARS-CoV-2.
  * **M3:** M2 restricted to the first PDB structure of the antibody, in human/mouse non-nanobody units.
* **Multiple testing:** global Benjamini–Hochberg over 382 tests (`tables/all_tests_bh.csv`).

## Main results (cryo-EM units)
| Feature | Errors | Clean | Adjusted OR (M2) | q | Holds in M3? |
|---|---|---|---|---|---|
| Earlier structure of the *same* antibody | 13.1% | 32.8% | 0.32 | <1e-4 | n/a. **Largely by construction:** counting copied errors as cases gives 25.9% vs 32.8%, OR 0.76, p = 0.08 |
| Closest prior V (any) | 0.854 | 0.902 | 0.69/SD | <1e-4 | **No.** OR 0.90 (q 0.70). Also gone after adjusting for an earlier same-antibody structure (0.88, p = 0.23) |
| Closest prior *different* antibody | 0.840 | 0.869 | 0.82/SD | 0.09 | No (0.91) |
| Non-human/mouse germline (alpaca, cow, rabbit, rat, rhesus) | 27.9% | 12.8% | 2.97 | 2e-4 | n/a. Cow 6/19 units (ultralong CDR3; 5 of the 6 from one lab), rabbit 7/62, alpaca 35/417, human 3.2%, mouse 5.2% |
| **A: SEQRES starts after IMGT 1** | 32% (19/59) | 7.8% | 5.3 | <1e-4 | **Yes.** OR 7.5, q <1e-4. 18 labs, 2016–2026 |
| **D (all): SEQRES ends inside FR4 (<IMGT 125)** | 6.8% (4/59) | 0.9% | 14.0 | 2e-4 | Yes (13.7, q 0.006) |
| **E: SEQRES ends inside FR4** | 11% (2/18) | 0.9% | 15.3 | 0.009 | Yes (14.4, q 0.03) |
| **E: V gene never before in a PDB structure** | 11% (2/18) | 0.55% | 26.8 | 0.001 | Yes (22, q 0.007) |
| E: log10 prior PDB entries with the same V gene | 1.66 | 2.06 | 0.52/SD | 0.026 | Yes (q 0.01). Includes mouse IGKV8-30 ×3 |
| Run window holds a residue at a rarely occupied IMGT column (matched expectation) | 7.0% (14/199) | 0.47% | ×15 | <1e-6 | Breakdown: A 4/59 (extra N-terminal residue), B 4/25 (atypical CDR2), D−/D0 4 (bovine ultralong CDR3), C 1 |
| Run window holds a rare (<1%) residue (matched expectation) | 29.1% | 22.6% | ×1.29 | 0.13 | Weak. D0 5/9 vs 20% |
| B: CDR2 identity to the closest prior different antibody | 0.79 | 0.92 | 0.62/SD | 0.03 | **Explained by atypical CDR2 length.** In typical-length CDR2s: OR 0.90, p = 0.63 |
| D: J-germline identity | 0.82 | 0.89 | 0.67/SD | 0.014 | Mostly FR4 truncation: excluding truncated SEQRES, OR 0.78 (p = 0.08). M3 0.63 (p = 0.003, n = 28) |
| Framework typicality (PDB-profile log-likelihood) | −0.363 | −0.328 | 0.81/SD | 0.048 | No (0.84). E: 0.57/SD (q 0.04) |

## What showed no difference
* **Germline novelty.** First-time germlines are not the link:
  * first-time V gene: 1.1% vs 0.5% (q 0.75);
  * first-time J gene: 0 vs 0.1%;
  * first-time V–J pair: 3.8% vs 3.5%;
  * first-time CDR1/2/3 length combination: 4.4% vs 2.2%;
  * first-time V + CDR1/2 lengths: 4.4% vs 2.6%;
  * number of prior relatives at ≥90% / ≥80% identity: no difference.
* **Novelty at the error location:**
  * mode A, N-terminal (IMGT 1–10) identity: 0.963 vs 0.975, OR 0.97. Adjusted for truncation it is 1.17;
  * mode C, FR3: 0.955 vs 0.959, OR 1.13;
  * mode F, CDR1: OR 0.72, q 0.38;
  * modes D+/D−, CDR3 length: 14.9 vs 13.3 for D+, OR 1.23;
  * modes D+/D−, closest prior same-length CDR3, and prior same-length CDR3 counts: no difference in M3;
  * modes D/E, FR4 mutations: no difference.
* **Rare residues at fixed position sets:**
  * canonical: 26% vs 20%, q 0.59;
  * Vernier: 31% vs 26%;
  * VH–VL interface: 16% vs 13%;
  * any framework position: no difference;
  * fixed mode windows (A, C, D, E, F): no difference.
* **Mutation load:** SHM total and germline V identity show no difference in M2.
* **Constructs:** humanisation signature (0.5% vs 1.3%), scFv or multi-domain constructs, unknown (X) residues and
  polyclonal entries show no difference.
* **J-gene usage** is homogeneous (H p = 0.40, κ 0.40, λ 0.97).
* **Period.** The 2020–21 error peak is not a novelty peak:
  * the half-year mean nearest-different-antibody identity is flat at 0.86–0.88;
  * the share with an earlier same-antibody structure is not correlated with the half-year error rate (ρ −0.34,
    p = 0.18).

## Weak or uncertain
* **V-gene heterogeneity in human/mouse conventional cryo-EM units:**
  * heavy chains, p = 0.001: IGHV4-34 16% (5/31), IGHV5-51 11%, IGHV3-53 9.5%;
  * λ chains, p = 0.013: IGLV1-44 18.5%, IGLV1-47 14%.

  Each gene's errors span several modes and labs (for IGHV4-34: 5 errors, 5 labs, 5 modes), and the test is not
  year-adjusted. So this is not a single mechanism.

## Interpretation
* **There is no "novel antibody, no template" link.** The apparent novelty of the erroneous antibodies is explained by:
  * the case definition: an earlier structure of the same antibody turns its errors into "copied" ones;
  * non-human/mouse species.
* **What links the errors at the sequence level is the record and the construct at the slipped strand:**
  * the deposited sequence is truncated at that terminus;
  * or it carries an insertion that most antibodies lack;
  * or (FR4 bulge) the V gene is rarely or never represented in the PDB.

  This fits the previous finding: the register is assigned by counting across a stretch with nothing to anchor it.

## Caveats
* **Overlap with the detector:**
  * Rare-column insertions and SEQRES termini depend on ANARCI numbering, which is also one arm of the detector. Some
    of these runs could be convention disagreements rather than modelling errors.
  * SEQRES truncation may reflect depositors entering only modelled residues. In that case the truncation and the
    unbuilt-terminus finding of `../../deepdive` partly describe the same event.
* **Sample size and reference set:**
  * Mode-level n is small: 12–59 error units, and 2–4 for several positives.
  * The PDB is the reference population (SAbDab-like). Natural-repertoire (OAS) statistics were not available offline.
  * Concurrent, unreleased structures from other labs are invisible to the "prior PDB" measures.
* **Germline assignment:**
  * Genes are human/mouse only, so nanobodies get human genes.
  * ANARCI's germline sets for cow, rabbit and alpaca are sparse.

## Files
* **Scripts:** `scripts/` holds:
  * `s01_prefilter.py`;
  * `s02_anarci.py`;
  * `s03_build_db.py`;
  * `s03b_kabat_map.py`;
  * `s04_novelty.py`;
  * `s05_units.py`;
  * `s05b_termini.py`;
  * `s06_tests.py`;
  * `s07_window.py`;
  * `s08_followups.py`;
  * `s09_typicality.py`;
  * `s10_summary.py`;
  * `s11_cases.py`.
* **Tables:** `tables/` holds:
  * `chain_novelty.csv.gz`: per-chain features;
  * `units.csv.gz`;
  * `units_typicality.csv.gz`;
  * `tests_em_units.csv`;
  * `tests_typicality.csv`;
  * `window_rare_matched.csv`;
  * `window_rare_summary.csv`;
  * `D_jsplit.csv`;
  * `followups.log`;
  * `period_composition.csv`;
  * `all_tests_bh.csv`: global BH;
  * `cases_positive_features.csv`;
  * `unit_mode_windows.csv`.
* **Cache:** the ANARCI cache is in the scratchpad at `deeper/cache/seq_novelty/` (`anarci_num.pkl`, `abdb.pkl`).
