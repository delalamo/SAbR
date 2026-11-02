# Do register errors in deposited antibodies make structure predictors reproduce them?

**Status:** proposed test, with a pilot run on this machine. This is a new hypothesis, numbered **H19** here.
* It is not the report's H16. H16 asked whether AlphaFold starting models *reduced* errors after 2021.
* H19 asks the converse. Every structure predictor is trained on the PDB, so it may have learned the
  errors (report §C: errors are rarely fixed and keep being reused).
* If it did, a predicted starting model would carry the error back into new depositions. That would
  also explain why H16 found no benefit for entries that record an AlphaFold starting model.

## Three routes by which a deposited error could reach a prediction

| Route | What it would look like | Methods exposed |
|---|---|---|
| **1. Templates at inference time** | The prediction copies a template that has the error. This is H12, done by software. | ColabFold/AF2 with templates; template-based antibody modellers |
| **2. Memorised training examples** | The same antibody sequence, seen in training with the error, is predicted with the error. | Every trained predictor, if the entry passed its filters |
| **3. Learned bias that generalises** | A new antibody gets the error because its framework resembles erroneous training examples (the identical-error families in `../sabr_anarci_scan`). | Every trained predictor |

Route 3 is the one that matters most. It would put errors into antibodies that were never deposited
(designs, repertoire models, cryo-EM starting models).

## Readout: no corrected structure needed

For each case (an antibody chain with a detected register error spanning IMGT window *w* and shift
δ), predict the Fv from sequence. Then score the prediction two independent ways.
1. **Ladder against the deposited model.**
   * Superpose the predicted chain on the deposited chain using framework Cα atoms outside *w* and
     outside the CDRs.
   * For each deposited residue in *w*, find which predicted residue lies within 3 Å.
   * A sequence offset of 0 means the prediction puts the same residue in the same place as the
     deposit: it **copies the error**. An offset of δ means it is in the **canonical register**.
2. **SAbR vs ANARCI on the prediction.** This is the same detector as the PDB scan. A run of ≥3
   disagreeing residues inside *w* means the prediction carries a register shift there.

**Positive control.** Feeding the deposited chains through the same pipeline must return "copies
error" from both readouts. It does, in 16/16 pilot cases (`tables/pilot_readout.csv`, method
`deposited`).

**Adjudication (optional, for a subset).** For cases where predictors disagree with each other, a map
check decides which register is right. Score both registers by per-residue Q-score (EM) or RSCC
(X-ray), as in `../sabr_anarci_scan/scripts/sabr_mapfit.py`.

**Exclusion found in the pilot.** Antibodies with a framework insertion relative to germline must be
excluded.
* **Example:** the 7CWU heavy chain has EVQL**L**EQ… against germline QVQL-VQ…
* **Effect:** SAbR and ANARCI place the extra residue differently. Deposit and predictions then
  "agree" on a shift that is not an error.
* **The 7CWU case:** this one antibody accounts for 16 flagged chains in 6 entries. Every predictor
  tested "reproduced" it.
* **Fix:** `scripts/h19_cases.py` now drops chains with an ANARCI insertion code at a framework
  position.

## Exposure: which errors each predictor was trained on

Exposure is decided by release date and the method's training filters. All dates below are
from the methods' own papers or technical notes.

| Method | Training cutoff (PDB release) | Antibody-relevant filters | Runs on this machine? |
|---|---|---|---|
| AlphaFold2 monomer; AF-Multimer v1, v2 | 2018-04-30 | | yes (ColabFold, CPU) |
| ESMFold | 2020-05-01 | | marginal (8.4 GB weights, 15 GB RAM) |
| IgFold | SAbDab before about Jul 2021 (its benchmark starts 2021-07-01); plus AF2-predicted OAS antibodies | | yes |
| ABodyBuilder2 | SAbDab extracted 2021-07-31 | ≤3.5 Å; VH and VL both present; no missing residues except at termini | yes |
| AF-Multimer v2.3 (ColabFold `multimer_v3`), Boltz-1, AlphaFold3 | 2021-09-30 | | AF-M v2.3 and Boltz-1 yes; AF3 no (weights on request) |
| Boltz-2 | 2023-06-01 | | yes |
| ABodyBuilder3 | SAbDab to Jan 2024 | ≤3.5 Å; no nanobodies | probably (weights on Zenodo; code on GitHub, which this sandbox cannot download from) |
| Chai-1 | | | no (needs a CUDA GPU) |

ABodyBuilder2's filters exclude most erroneous cryo-EM Fabs, because errors sit next to unbuilt
residues (H17). In the pilot, only 2 of 16 erroneous chains passed them. Methods with loose
resolution filters saw far more of the errors.

**How many candidate errors fall inside each training window.** The candidates come from the 3–5 Å
PDB scan: framework-type runs of ≤30 residues with |δ| ≤ 3. That gives 234 entries, before removing
framework-insertion artifacts. Counts are by release date (`tables/exposure_counts.csv`).

| Cutoff | Exposed entries | Unexposed entries |
|---|---|---|
| 2018-04-30 (AF2, AF-M v1/v2) | 8 | 226 |
| 2020-05-01 (ESMFold) | 23 | 211 |
| 2021-07 (IgFold) | 56 | 178 |
| 2021-07-31, ≤3.5 Å (ABodyBuilder2; internal-gap filter not yet applied) | 32 | 174 |
| 2021-09-30 (AF-M v2.3, Boltz-1, AF3) | 65 | 169 |
| 2023-06-01 (Boltz-2) | 117 | 117 |
| 2024-01, ≤3.5 Å (ABodyBuilder3) | 68 | 100 |

## Design

**A. Memorisation (route 2): the main test.** Predict every candidate error with every method.
* **Outcome:** "copies error" by both readouts.
* **Comparison within each method:** copy rate for errors in its training window vs errors released
  after its cutoff.
* **Model:** logistic, clustered by antibody sequence. Adjust for window location (N-terminal FR1
  vs internal), |δ| and distance to a CDR.
* **The cleanest contrasts are within one architecture,** as a difference-in-differences. The copy
  rate for errors released between the two cutoffs, minus the rate for errors released after both,
  cancels version-to-version differences:
  * AF-Multimer v2 (2018-04-30) vs v2.3 (2021-09-30): same network, retrained;
  * Boltz-1 (2021-09-30) vs Boltz-2 (2023-06-01);
  * ABodyBuilder2 (2021-07) vs ABodyBuilder3 (2024-01).
* **Dose:** errors deposited many times (the identical-error families) should be copied more if
  memorisation is the mechanism.
* **Power** (simulated, one-sided Fisher, α = 0.05):

  | Exposed vs unexposed entries | Detects 20% vs 5% copying | Detects 10% vs 3% |
  |---|---|---|
  | 80 vs 150 | 95% power | 62% |
  | 40 vs 190 | 80% | 42% |

  Strong memorisation is detectable; weak memorisation is not, unless the candidate set is enlarged.
  The PDB-wide scan so far covered only 3–5 Å entries. Scanning <3 Å and >5 Å antibody entries costs
  about 0.3 s per chain.

**B. Templates (route 1).** ColabFold AF-Multimer v2.3 with templates on and off. Set
`--max-template-date` just before vs just after the erroneous entry's release. Template copying is
the difference. Also log which templates were used.

**C. Generalisation (route 3).**
* **Query antibodies:** never in any training set. Use entries released after 2024-01 that the
  detector calls clean, or paired OAS sequences.
* **Groups:** antibodies whose nearest training neighbour carries a register error, vs matched
  antibodies whose nearest neighbour is clean. "Nearest neighbour" means the same V gene and ≥90%
  framework identity.
* **Outcome:** the detector on the prediction, inside the neighbour's error window.
* **Scale:** this arm is reference-free, so the fast predictors can run thousands of queries.

**D. Memorisation check (only for copied cases).** Mutate CDR-H3 residues far from the window, or
swap in a different light chain. If the error disappears, it is sequence memorisation rather than a
learned framework bias.

**E. Feedback into new depositions (link to H16).**
* **Sample:** post-2021 entries that record an AlphaFold or ColabFold starting model.
* **Test:** predict them with the version that was current at deposition. Then ask whether the
  deposited errors sit where that prediction also had them.

## Pilot on this machine (4 CPU cores, 15 GB RAM, no GPU)

**Sample.** 16 erroneous Fvs, 2–4 per release stratum (`tables/pilot_cases.json`). Both chains, V
regions only.

| Case | Released | Window, δ | ABodyBuilder2 | IgFold |
|---|---|---|---|---|
| 6BPE κ | 2018-06 | FR1 3–8, −1 | canonical | canonical |
| 6CMO κ | 2018-06 | FR1 4–12, −2 | canonical | canonical |
| 6OT0 κ | 2019-06 | FR1 4–9, −2 | canonical | canonical |
| 7JTF H | 2021-03 | FR2 55–62, −1 | canonical | canonical |
| 7E23 κ | 2021-05 | FR3 76–93, −2 | canonical | canonical |
| 7CZV κ (passes ABB2 filters) | 2021-03 | FR1 1–9, −1 | canonical | canonical |
| 8DDK κ | 2022-12 | FR1 2–9, −1 | not copied (shift 1–2) | not copied |
| 8E8L H | 2023-06 | FR1 4–16, −1 | canonical | canonical |
| 8CR9 H | 2023-11 | FR2 50–63, −3 | unclear | not copied |
| 9YC6 H | 2026-02 | FR4 120–128, +1 | not copied | not copied |
| 8TPX κ | 2024-08 | FR1 1–8, −1 | canonical | canonical |
| 26PF H | 2026-08 | FR1 8–35, −2 | not copied | not copied |
| 7CWU H | 2020-12 | FR1 1–4, −1 | *"copies"* | *"copies"* (framework-insertion artifact; excluded) |
| 6ORP, 8HHX, 22GE | | | framework superposition RMSD >2.5 Å | same |

**Result.** Neither ABodyBuilder2 nor IgFold reproduced any of the 12 interpretable real errors. That
includes 6 released before both models' cutoffs. Only 1 of those 6 (7CZV) passed ABodyBuilder2's
training filters, and IgFold's are not documented. With n = 12 this rules out only near-universal
memorisation, not a modest effect.

**Slower predictors.** Boltz-2, ColabFold AF-Multimer v2.3 and ESMFold were run on one case (6BPE,
released 2018-06, inside all three training windows). All three put the segment in the canonical
register (ladder −1/−2, no detector run). Framework RMSD to the deposit was 0.8–1.2 Å.

## Compute on this machine

Times were measured here on one Fv: 6BPE, 227 residues (H + L). Each method used its default
settings, except where noted.

| Method | Time per Fv (4 cores) | Peak RAM | Disk for weights | Notes |
|---|---|---|---|---|
| ABodyBuilder2 | 7–10 s | <2 GB | <0.5 GB | unrefined; refinement needs OpenMM (installed) |
| IgFold | 13–40 s | <3 GB | <1 GB | no refinement; transformers had to be pinned to 4.44 |
| Boltz-2 | 6 min 11 s, plus ~2 min MSA (server) | 9.5 GB | 4.2 GB (+2 GB optional affinity weights) | 1 diffusion sample, 200 steps, 3 recycles |
| ColabFold AF-Multimer v2.3 | 12.5 min per model, plus 2.3 min MSA (server) | 4.2 GB | 3.8 GB | 1 model; stopped early after 1 recycle. Defaults (5 models, up to 20 recycles) are 5–10× longer |
| ESMFold | 4 min 15 s (+80 s model load, once) | 13.8 GB of 15 | 8.4 GB | single sequence; H + 25 G linker + L. Works, but with almost no RAM to spare |
| Boltz-1, AF-Multimer v2 | presumably as Boltz-2 / AF-M v2.3 | | +3–4 GB each | not run; same code paths |
| Chai-1, AlphaFold3, Protenix | | | | not possible: CUDA GPU required (Chai-1), weights on request (AF3) |

**What the full study would cost here.**
* **Arm A:** about 200 errors (after exclusions) plus 200 matched clean chains, i.e. 400 Fvs.
* **Arm B:** about 100 errors × 2 template dates.
* **Arm C:** about 1,000 queries for the fast methods and 200 for the slow ones.

| Method | Fvs | CPU time here |
|---|---|---|
| ABodyBuilder2 + IgFold | 1,400 each | about 10 h together. **Feasible.** |
| Boltz-2 (+ Boltz-1) | 600 each | about 70 h each |
| ColabFold v2 + v2.3, 1 model | 600 each + 200 template runs | about 300 h |
| ESMFold | 600 | about 42 h |

**Conclusion.**
* **Fast antibody predictors (ABodyBuilder2, IgFold):** the whole test runs comfortably on this
  machine, including the large generalisation arm.
* **General predictors (Boltz, AF-Multimer, ESMFold):** the study is CPU-bound at roughly 2–3 weeks of
  continuous computing. That is impractical in an ephemeral container.
* **Disk:** about 35 GB in total for environments and weights, against 26 GB free at the start.
  Weights have to be downloaded, used and deleted method by method. Downloads ran at about
  30 MB/s.
* **Options for the general predictors:**
  1. a GPU, where these models are commonly reported at tens of seconds per Fv (not measured here),
     so the whole study fits in about a day;
  2. or a reduced CPU design: arm A only, Boltz-2 and AF-M v2.3 only, 1 model each, about 230
     errors and no controls (about 3–4 days);
  3. or the within-architecture contrast only (about 4 days). Run AF-M v2 vs v2.3 on the ~225
     errors released after 2018-04-30. About 60 of them fall between the two cutoffs; the rest are
     the unexposed comparison. This answers the causal question most directly.
* **MSA server:** the ColabFold public server is shared, so a few hundred queries should be paced.
  Boltz and ColabFold can reuse the same `.a3m` files.

## Files

* `scripts/h19_cases.py`: picks erroneous chains and pairs each with the nearest partner V domain.
  Writes the V-region Fv sequences.
* `scripts/h19_run_ab.py`: runs ABodyBuilder2 (unrefined) and IgFold (no refinement).
* `scripts/h19_posctl.py`: writes the deposited V regions as a "prediction" (positive control).
* `scripts/h19_readout.py`: computes the ladder and the SAbR/ANARCI detector on predictions.
* `tables/`: pilot cases and readouts, exposure counts, timings.
