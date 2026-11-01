# Register-error rates by fold class: antibodies vs other folds

Re-analysis of Supplementary Table S1 from Sánchez Rodríguez et al., *IUCrJ* 11, 938 (2024),
"Using deep-learning predictions reveals a large number of register errors in PDB depositions".
The table lists 12,674 putative register errors, found by conkit-validate in 3–5 Å X-ray and cryo-EM
entries deposited up to 5 April 2022.

## Bottom line

**Overall, antibodies do not have a higher register-error rate than comparable folds. They have one of
the lowest.** After adjusting for method × resolution, deposition year and domain length:

* Antibody **V domains** are flagged about as often as other Ig-superfamily domains (MHC, β2m, receptor
  V/C1/I-set domains). Adjusted OR for other IgSF vs antibody V is 0.84 (95% CI 0.38–1.88).
* Antibody **V domains** are flagged about **2–5× less often** than the other Ig-like/E-set and
  β-sandwich groups and than most other ECOD architectures.
* Antibody **C domains** are flagged at about half the V-domain rate (OR 0.43).

**There is one important caveat: the gap closes over time.** For most folds the flag rate fell from
about 5% (≤2017) to about 2% (2020–22). For antibody V domains it went *up*, from about 0.4% to about
1–2%. In 2020–22 cryo-EM structures, antibody V domains (2.2%) are flagged at the same rate as all
other domains (2.0%). Adjusted OR for antibody V vs everything else, by deposition era:

| Era | Adjusted OR (Ab V vs rest) | 95% CI |
|---|---|---|
| ≤2017 | 0.08 | 0.03–0.19 |
| 2018–19 | 0.35 | 0.17–0.71 |
| 2020–22 | **1.31** | 0.91–1.88 |

The study ends in April 2022, so there is nothing yet on whether this trend continued.

## Method

1. **Errors.** Table S1 gives PDB:chain and residue range for each putative error. These already
   passed the authors' filters, and redundant copies are kept as separate rows.
2. **Denominator.** Table S1 lists only the flagged chains, not the analysed ones, so I rebuilt the
   denominator. RCSB search for X-ray or EM entries at 3.0–5.0 Å deposited ≤2022-04-05 returned
   21,672 entries. I kept protein chains of 20–1000 residues, giving 201,757 chains. 99.7% of Table S1
   errors map onto these chains.
3. **Fold classification.** I used ECOD v295.2 (Aug 2026), which covers 20,414 of the entries,
   including multi-character chain IDs. Each error was assigned to the ECOD domain it overlaps most.
   This placed 10,921 of the 12,674 errors (86%). CATH 4.4 was tried first and dropped, because it
   covers only 12–55% of these chains and almost none of the large cryo-EM complexes.
4. **Antibody / TCR identification.** I ran ANARCI (HMMER) on all 29k unique sequences. A domain
   counts as "Antibody V" if it is a V-set domain in ECOD's Ig/FN3/E-set T-group on a chain with an
   ANARCI H, K or L hit. "Antibody C" is the same with C1-set.
5. **Unit and statistics.** The unit is the domain instance, and "flagged" means at least one error
   overlaps it. 95% CIs use bootstrap resampling over PDB entries. The adjusted model is a logistic
   regression with method × resolution bin, deposition-year bin and log domain length, using
   entry-clustered SEs. I also repeated the analysis after collapsing identical sequences, giving one
   observation per unique (sequence, ECOD family, domain start).

## Results

### Raw rates by group (selected rows; all groups in `table_raw.csv`)

| Group | Domains | Entries | % flagged (95% CI) | % residues in errors | % flagged, unique sequences | % cryo-EM |
|---|---|---|---|---|---|---|
| Other Ig-like X-group (nicotinic-receptor LBD, C2, prealbumin-like…) | 2,485 | 614 | 10.3 (7.5–13.1) | 0.75 | 6.6 | 32 |
| α complex topology | 6,211 | 1,958 | 8.1 (6.8–9.4) | 0.80 | 9.7 | 52 |
| α/β three-layered sandwiches | 33,898 | 6,426 | 5.3 (4.8–5.9) | 0.39 | 6.0 | 53 |
| Other β-sandwiches | 12,287 | 1,881 | 4.3 (2.3–6.7) | 0.55 | 3.9 | 58 |
| β-barrels | 26,719 | 4,051 | 4.3 (3.7–4.8) | 0.80 | 4.0 | 58 |
| Other Ig/E-set/PapD T-group (non-antibody/TCR) | 3,557 | 804 | 4.2 (3.0–5.6) | 0.62 | 4.1 | 35 |
| α/β two layers | 51,975 | 5,708 | 2.5 (2.2–2.8) | 0.43 | 4.0 | 54 |
| TCR V | 444 | 126 | 2.5 (0.5–4.8) | 0.13 | 2.8 | 2 |
| **Antibody V** | 8,637 | 2,118 | **1.3 (1.0–1.6)** | 0.13 | **1.8** | 51 |
| Other IgSF (V/C1/C2/I-set: MHC, β2m, receptors) | 2,202 | 485 | 1.2 (0.5–2.2) | 0.15 | 1.8 | 15 |
| **Antibody C** | 4,470 | 1,045 | **0.6 (0.3–0.9)** | 0.05 | **0.8** | 24 |
| Fibronectin type III | 250 | 107 | 0.4 | — | — | — |

### Adjusted ORs vs antibody V (full list in `table_adjusted_vs_abV.csv`)

Antibody C 0.43 (0.22–0.85) · other IgSF 0.84 (0.38–1.88) · TCR V 1.67 (0.63–4.4) ·
other β-sandwiches 2.0 (1.0–4.0) · other Ig/E-set 2.6 (1.7–4.1) · β-barrels 2.8 (2.1–3.8) ·
α/β 3-layer sandwich 2.6 (1.9–3.5) · α superhelices 3.4 (2.4–4.7) · other Ig-like X-group 5.4 (3.5–8.3).

### Within the same entries (same map or data, same resolution)

In entries that contain antibody V domains, the antibody V domains and the non-antibody chains
(mostly antigens) in those entries:

| Method | Antibody V | Antibody C | Non-antibody chains in the same entries |
|---|---|---|---|
| Cryo-EM | 1.95% | 1.03% | 2.25% |
| X-ray | 0.57% | 0.41% | 2.17% |

With length adjustment and entry clustering, the antigen vs antibody V OR is 1.20 (0.77–1.85), and
antibody C vs V is 0.47 (0.24–0.94). In cryo-EM, the antibody V domains and the antigens in the same
map are flagged at similar rates.

### Where antibody V-domain errors fall

134 errors; IMGT numbering via ANARCI, mapped through RCSB's author-to-sequence residue mapping.

| Chain | FR1 | CDR1 | FR2 | CDR2 | FR3 | CDR3 | FR4 |
|---|---|---|---|---|---|---|---|
| Heavy (residues) | 199 | 7 | 30 | 35 | 230 | 121 | 164 |
| Light (residues) | 218 | 91 | 28 | 9 | 99 | 20 | 65 |

* 61% of the flagged antibody errors never touch a CDR, and only 16% touch CDR3. This makes it less
  likely that the antibody signal is mostly AlphaFold2 failing on CDR-H3 loops.
* 81% are ±1-residue shifts. The median length is 7.5 residues.
* The per-error list is in `antibody_V_errors_mapped.csv`.

## Caveats

* **The denominator is my reconstruction, not the authors' analysed set.** They retrieved 19,310
  entries and analysed 16,662. Chains they excluded (no usable AF2 model, AF2 pLDDT < 65, failures)
  are counted here as "not flagged". That lowers absolute rates, and lowers them most for folds AF2
  predicts poorly. AF2 predicts antibody V/C domains confidently, so antibodies were probably analysed
  at a *higher* rate than average. If anything, this biases the comparison *against* antibodies, so
  their lower rate is unlikely to come from the denominator.
* **The flags are predictions.** Errors come from conkit-validate after the authors' filters. They
  report about 87% agreement with later high-resolution structures, plus known false positives
  (fold-switching proteins, disulfide-rich peptides, genuinely ambiguous registers).
* **Possible detection blind spots.** Register shifts inside long β-strands of very regular sandwich
  folds could be harder to detect from contact maps, because shifting by two residues preserves the
  strand-pairing pattern. That would affect antibodies and other Ig folds alike, not antibodies
  specifically.
* **Small groups.** FN3, TCR and some other groups are small, so their CIs are wide.
* **Why antibody V rates rose over time is not established.** One hypothesis is the shift toward
  cryo-EM Fab–antigen complexes (Fabs at the periphery of maps whose local resolution is worse than
  the reported global value), plus the 2020–22 surge in SARS-CoV-2 antibody structures. I have not
  tested this.

## Files

* `table_raw.csv`: per-group counts, raw rates, CIs, residue-level rates, deduplicated rates.
* `table_adjusted_vs_abV.csv`: adjusted ORs.
* `antibody_V_errors_mapped.csv`: each antibody V-domain error with its IMGT region.
* `scripts/`, in pipeline order: `gql.py` and `gql2.py` (RCSB metadata and sequences),
  `run_anarci.py`, `build.py` (chain table), `build_ecod.py` (error → domain assignment),
  `analyze.py`, `abv_regions.py`. Input data: Table S1 (Europe PMC supplementary-files API),
  ECOD v295.2, RCSB Search and GraphQL APIs (queried 2026-11-01).
