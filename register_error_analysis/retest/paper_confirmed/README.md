# Do the general trends hold in the paper-confirmed cases (≤2022)?

**Short answer: yes, the core conclusion holds.** For errors that conkit flagged in Table S1 and that SAbR/ANARCI also confirm, the
deeper dive's main finding is reproduced: the errors carry a whole-model, antibody-specific building fingerprint, not a
person, antigen or unusual-antibody signature. The local hazard reproduces too: unbuilt residues next to the slipped strand.
Every fingerprint trend that can be tested goes in the same direction, and most have CIs that exclude the null with one vote
per paper. The constant-domain cis deficit is the exception; as in the full set, it rests on one paper. The
people/competition/journal nulls stay null. The antibodies look as ordinary as in the full set.

What does not carry over:
- Most mode-specific features are untestable, with 1–9 units per mode.
- Poor map fit at the site is attenuated in cryo-EM.
- Some full-set trends disappear: the Krios deficit, and partly the sparse-deposition trend.
- A few new signals appear: China-led papers, less experienced teams, lower citations. These appear only in the conkit-confirmed
  subset, so they are best explained by conkit's selection rather than by the errors.

Verdicts below are those of the adversarial checkers wherever they disagreed with the re-test (marked †). I found no case
where the files showed the checker to be wrong.

---

## 1. Case set and controls

Built by `case_set/build_cases.py`. It was re-derived independently, with identical results, by all four verifiers.

- **Source.** Table S1 of Sanchez Rodriguez et al. (IUCrJ 2024) flags 162 errors on chains in our scan. Of these, 131 are on
  single-domain V chains with ≥50% of the error's residues in the V region.
- **Status of the 131.** A Table S1 error counts as confirmed when a SAbR/ANARCI run (≥3 consecutive disagreeing V residues)
  overlaps it. Each confirmed run was then classed with the deeper dive's ledgers:

  | Status | Errors |
  |---|---|
  | Confirmed, first occurrence | 69 |
  | Copy of an earlier model | 16 |
  | Near copy | 4 |
  | Artifact (strict ledger) | 5 |
  | Confirmed, loose-ledger doubt | 1 |
  | Not confirmed by SAbR/ANARCI | 36 |

  All 131 were released by 2022-02-02.

| Set | Chains | Entries | Antibody × paper units | Papers | EM / X-ray chains |
|---|---|---|---|---|---|
| **Strict cases** (confirmed first occurrence) | 67 | 38 | 34 | 25 | 49 / 18 |
| Including copies and near copies (sensitivity) | 87 | 53 | 42 | 32 | – |
| **Controls**: released ≤2022-04-05, entry without a Table S1 V-region error, no SAbR/ANARCI run on the chain | 6,239 | 1,534 | 2,396 | 975 | 2,679 / 3,560 |

- **One dominant paper.** Zhou et al., Cell Res 2021 (10.1038/s41422-021-00487-9; the 7CZR–7D00 series) contributes **19 of the 67
  strict chains, 39% of the EM chains**. One lens write-up says 17; the correct figure is 19. Every key test was repeated without
  this paper, with leave-one-paper-out, and with one vote per paper.
- **Control definition.** The control filter excludes only entries carrying one of the 131 V-region errors, not every Table S1
  entry. 363 control chains come from entries that Table S1 flags elsewhere. A stricter control set changes nothing material
  (e.g. EM gap OR 5.50 vs 5.36).
- **Rest of the deeper set.** Heterogeneity is tested against the deeper dive's other first-occurrence cases, mostly cryo-EM
  from 2018–26.

## 2. Results by family

Legend:
- **holds**: same direction, CI excludes the null.
- **underpowered**: same direction, CI includes the null.
- **attenuated**: same direction but clearly weaker.
- **absent**: no effect, or the reverse.
- **null still null**: the trend was null in the full set and is still null.
- **cannot test**: too few units, or data missing.

OR is case vs control unless stated. "Paper level" means one vote per paper.

### 2a. Whole-model fingerprint (deeper README §4.1, §7)

| Trend | Full set | Paper-confirmed, strict | Verdict |
|---|---|---|---|
| κ Pro8 (IMGT 8) modelled cis | EM 7% vs 64%, paper-weighted OR 0.115 (0.04–0.31) | EM **0/18** (6 papers) vs 69%, Haldane OR 0.012, p=1e-9. Paper-weighted OR 0.067 (0.011–0.40). Without Zhou 0/6, p=9e-4. Conkit-missed EM κ errors in the window are also 0/17. No heterogeneity (p≈0.3). | **holds**. Fragile: outside FR1 slips and outside Zhou the evidence is about 3 chains. X-ray cannot test. |
| Trans Pro8 as a flag | 5.7% error if trans vs 0.26% if cis | 6.0% (18/298) vs 0/615. NPV 100%, PPV 6%. | **holds** (EM only; useless in X-ray) |
| Pro8 deficit is a paper-level habit (within-paper) | Sign p=0.22 | 3 informative papers; 5 of 7 case papers are all-trans in every κ chain | **null still null**. Fits the paper-level reading. |
| κ L3 cis-Pro95 (IMGT 115) | 23% vs 76%, OR 0.077 | EM 0/17 vs 68%, p=7e-9. Paper-weighted OR 0.071 (0.011–0.46). Without runs at IMGT ≥104: 0/10. | **holds**. Fragile: outside Zhou, about 3 chains. |
| Worse Ramachandran away from the slip | +3.75 per 100 (+3.1 on this metric) | EM +8.2 (3.1–13.4), X-ray +4.7 (1.1–8.3). Paper level EM +6.0 (1.7–10.3). | **holds**. †"Larger than the rest" does not survive at paper level (p=0.11–0.18). |
| Antibody vs rest of model gap (V minus non-V % allowed) | EM +2.16 pp (1.13–3.19) | EM +6.5 pp (3.6–9.4). Paper level +6.5 (2.9–10.1). Larger than the rest even at paper level (p=0.024). X-ray +1.7 (−0.4 to 3.8). | **holds** (EM). X-ray underpowered. Non-V chains in X-ray paper entries are also slightly worse. |
| Framework Cα deviation (run excluded) | +0.39 Å in the rest of the set; OR 3.2/SD | Chain level EM +0.17 (−0.02 to 0.36). **Paper level +0.32 Å (0.11–0.52)**, vs the rest +0.47, p_het=0.26. Without Zhou, chain level +0.30. | **holds at paper level**†. The chain-level attenuation is one paper (Zhou's 19 chains sit below the controls), not demonstrably conkit selection. |
| VH error → VL error in the same entry | OR 15–25; rest of set 19.9 | EM 7/14 vs 17/554, OR 31 (10–94). Papers: OR 20.5 (5.1–82). X-ray OR 45, p=0.04. | **holds**. The size is inflated by conkit: VL is wrong in 50% of entries with a flagged VH error vs 14% with a missed one. The fair same-era reference is OR 8.9. |
| Within entry, the error chain is no worse than its clean neighbours (Ramachandran) | +0.8, n.s. | −0.2 (−2.8 to 2.3). The clean neighbours themselves are +5 vs controls. | **null still null** (supports "entry, not strand") |
| Within entry, framework RMSD | +0.17 Å, n.s. | Entry FE +0.16 to +0.17. Per paper +0.19, p=0.34. | **null still null**†. A ~0.15 Å chain-specific excess cannot be excluded. |
| Less β-sheet away from the site | 0.39 vs 0.44, p=3e-12 | EM −0.11 (−0.15 to −0.07). X-ray −0.09. Paper level EM −0.10 (−0.16 to −0.04). | **holds**. †"Stronger than the rest" fails at paper level (p=0.38). Partly overlaps the error definition. |
| Non-H3 CDRs off the canonical backbone | 0.76 vs 0.89 | EM subset, 20 chains in 14 papers: 0.65 vs 0.88, −0.23 (−0.34 to −0.12). Paper level MW p=4e-4. | **holds** (EM subset with existing predictions) |
| Constant-domain (Cκ/CH1) cis deficit | Rests on one all-trans paper | Pooled 12/32 vs 94%, all from Zhou. Without Zhou 12/13 cis, OR 0.95 (0.14–6.4). MDE OR ≈0.06. | **absent outside the one paper**. As in the full set; uninformative. |

### 2b. Local hazards and how the model was built (deeper §4.2, deepdive, junction)

Unit: antibody × paper × mode, giving 36 strict units (26 EM, 11 X-ray) from 25 papers.

| Trend | Full set | Paper-confirmed, strict | Verdict |
|---|---|---|---|
| Unbuilt residues within 5, or N-terminus unbuilt | 48% vs 16%; MH OR 4.35 | EM 50% vs 16.9%, **OR 5.36 (2.32–12.4)**. One per paper 7.2. Alignment-based definition (no SAbR labels) 4.02 (1.68–9.65). X-ray 6.8 (1.8–25). p_het=0.60. | **holds** |
| Bottom-quartile map fit at the site | OR 1.76 | EM OR 1.06 (0.44–2.54), MDE 3.0. X-ray (RSCC) OR 10.2 (2.3–44). Conkit-missed EM errors OR 2.4. | **attenuated in EM**. CI includes the full-set OR. Holds in X-ray. |
| Either unbuilt or poor fit | OR 3.49 | EM OR 3.17 (1.39–7.23). One per paper 3.9. | **holds** (all from the gap part) |
| Adjacent CDR of atypical length | MH OR 4.6 | OR 2.7 (0.48–15) | underpowered |
| Mode A: N-terminus not built | Model starts after IMGT 3: 29% vs 3.6% | On SAbR labels 22% vs 2.8%, p=0.028. On ANARCI/SEQRES positions 2/9 vs 9%, p=0.20. With copies 41% vs 9%. | **underpowered**†. The label-based measure is partly the shift itself. |
| Mode A: non-canonical N-terminal residues | Q1 low, E5 high, κ I2 low | E5 20% vs 2.3%; I2 75% vs 87%; Q1 no effect (n=5) | underpowered |
| Mode F: mouse, atypical CDR1 | 50% vs 19%; 42% vs 15% | n=3: 67% vs 23%; 67% vs 17% | underpowered |
| Mode B: atypical CDR2, nanobody, identity | e.g. atypical CDR2 24% vs 3% | n=4 units, 3 papers. 0/4 atypical CDR2; P(0/4 at the full-set rate)=0.23 | **cannot test**† |
| Mode C: mouse IGHV5 | 21% vs 2.6% | 0/5. Only 1 C unit is mouse (IGHV9). The in-window IGHV5 errors (7TE9/7TEB) were missed by conkit. | **cannot test**†. Conkit masks this trend. |
| Mode C: non-Asp at IMGT 81 | 30% vs 5% | 2/4 vs 4%, p=0.01 | holds (2 units; may mark gap placement) |
| Mode C: unbuilt nearby | 38% vs 2% | 4/5 vs 1.7% | holds |
| Mode D+: long CDR3 | 14 vs 12 | All methods 17 vs 11, but this is chain-type mix. Heavy only 18 vs 15, p=0.44. Full set also null within chain type. | **null still null**† |
| Mode D+: poor local map | Domain z and site Q low | n=4 EM; site Q 0.34 vs 0.38 (same direction) | **cannot test**† |
| Mode D+: unbuilt / IGHJ6 / SARS-CoV-2 | 61/19/39% vs 28/7/24% | Unbuilt 50% vs 15%, p=0.02. IGHJ6, heavy only, p=0.29. SARS-CoV-2 vs 2020–21 controls p=0.28 (an era effect). | **underpowered**†. Only "unbuilt" holds. |
| Mode D−: several features | – | 1 unit | cannot test |
| Mode E: mouse IGKV8, mouse, atypical CDR1 | 17% vs 0.9% … | 3 human SARS-CoV-2 units; P(0/3) 0.18–0.57 | **cannot test**† |
| CDR3/FR4 share of errors ≥ a fifth | 36–41% | 12/36 = 33% (CI 19–51%); Table S1's own share is 25% | **holds**. Narrowly; partly inflated by SAbR confirmation. |
| C-flank after CDR3 shifts positive | 87% +1 | 11/13 events positive (12 papers). Exactly +1 in only 6/13. | **holds** for a "positive shift"†. Not for "+1" specifically. |
| Template CDR3 length mismatch | 9 errors only at ≥2 difference | Only 7V2A←6Z3P, which is already in the junction README | **cannot test**† |
| Rigid-body only vs ab initio | 1.79% vs 0.23% | 5/437 vs 0/156, p=0.33; any mode null | underpowered |
| AlphaFold vs PDB template vs none | AF 0.34%, template 1.09%, none 1.70% | 0 AlphaFold starting models in the window. Template vs none OR 2.5 at chain level only (paper level p=0.79). | **cannot test** |
| Canonical pull toward common lengths | 39/84 vs 27 expected | 6/13 vs 4.0 expected, p=0.12 | underpowered |
| H2-9 / κ L1-17 | OR 2.45 / 3.1–3.6 | 1.75 (0.56–5.5) / 3.6 (0.97–13.6) | underpowered |
| λ L1-13 | n.s. without amyloid | 1 exposed case | null still null (untestable) |
| Fv-only models | OR 1.45, n.s. | OR 0.32 (0.04–2.7) | **null still null**† |
| Truncated SEQRES N-terminus (mode A) | Weak after adjusting for model start | 0/9 EM vs 5.5% | absent (it was already weak) |
| Truncated SEQRES in FR4 | Null after adjustment | 0/12 | **null still null**† |

### 2c. People, places, publication, antigen and competition (deeper §4.3, §4.4, §6)

Units: 27 EM case entries / 16–18 EM papers or studies vs 616 entries / ~300 papers.

| Trend | Full set | Paper-confirmed, strict | Verdict |
|---|---|---|---|
| Other labs on the same antigen (±1/6/12 m) | Adjusted OR 1.0–1.16 | 1.16 / 1.09 / 1.07; paper level 0.94 | null still null |
| Race (another lab within ±1 m) | 1.50 (0.98–2.30) | 1.31 (0.34–5.0); MDE 3.1 | null still null |
| SARS-CoV-2 vs other antigens | OR 0.91 | Entry level 2.78 (n.s.), carried by the 7CZR series; paper level 1.60 | null still null |
| Early SARS-CoV-2 race is one or two papers | Yes | First 25 entries clean; ranks 26–50 = 2 papers | holds (descriptive) |
| Fewer competitors / niche antigen | 0.60 / 8.3 | 0.48 / 5.1 (0.49–53) | underpowered |
| First structure of the antigen family | Adjusted 1.3–1.4, n.s. | Crude 4.0 (1.2–13.6); adjusted 2.9 (0.84–9.8) | underpowered (null after adjustment in both) |
| Capsid / enzyme targets | ≈1.0 after artifacts and networks | Capsid 4 papers from 3 co-author networks, 3 with non-human antibodies. Enzyme 2 entries; the 20.6 OR with copies is one lab's series. | **underpowered, discounted as in the full set**† |
| Teams new to antibody cryo-EM | Effect confined to 2022–23; ≤2021 null | 2.64 (0.92–7.6); the window is entirely ≤2021 | null still null |
| Team/lab antibody experience | 0.88–0.90 per doubling | 0.61–0.76, p<0.01. **Reversed in conkit-missed in-window errors (1.13–1.19); null for all in-window errors (0.94–1.02)** | **holds only in the conkit subset; produced by selection**† |
| Individuals, labs, institutions, facilities, hubs, co-author links | None beyond chance | Smallest q 0.15. Links 4 vs 2.1 expected (1 vs 1.8 without collision-prone names). | null still null |
| Country | Null | China-led 9/16 vs 22%, OR 5.5. Conkit-missed in-window errors 18% vs 22%. | **null not reproduced: new signal, attributable to conkit selection**† |
| Lab persistence / carrier | 1.69 / 1.02 | Crude 36% vs 36%; carrier 1.27 | null still null |
| Citations | 0.68 per log unit, between-lab | 0.49 (0.29–0.83). Entirely within China-led papers; non-China 0.89 (n=7). | holds (composition-sensitive) |
| Authors per antibody | 0.70 | 0.73 (0.43–1.25) | underpowered |
| Journal, funders, open access, preprints, timing, processing choices | Null | All n.s. The NIH dip is China composition. | null still null |
| Titan Krios / 300 kV before 2022 | Krios OR 0.53 | Entry 1.12; paper level 0.54–0.74; MDE 0.29. Conkit-missed in-window errors 0.17. | **attenuated / untestable; masked by selection**†. One stated fact is wrong: 7KTX (2020) used a Polara, not a Krios. |
| No funding record / half maps | 2.4 / 0.58 | 2.9 (1.24–6.7); after adjusting for China-led 1.39 (0.54–3.6). Paper level n.s. | **attenuated**† (China-led composition) |
| Recorded software / starting model | Rosetta/Coot/Phenix lower; template < none | All EM ORs <1, n.s.; AlphaFold untestable | underpowered |
| Rigid-body fitting (D−) / local resolution at site | – | 1 D− unit; 4 sites with EMDB local resolution | cannot test |
| Entry size (>500 kDa) | Null | 5.2, from 3 capsid entries | underpowered (composition) |

### 2d. Are the antibodies ordinary? (deeper §7; seq_novelty, struct_unusual, literature_dean)

Sequence tests: 34 units / 25 papers vs 2,378 window control units. Structural tests: Fvs in 19 papers. 37 EM Fv instances
are only **15 distinct Fvs**†; the checker de-duplicated these.

| Trend | Full set | Paper-confirmed, strict | Verdict |
|---|---|---|---|
| Mean VH–VL orientation (ABangle) | All n.s. | De-duplicated: HC2 −1.27°, p=0.18; X-ray LC1 p=0.09 | null still null |
| More VH–VL orientation outliers | 36% vs 16%, OR 2.7 | Distinct Fvs 4/15 (27%) vs 15%, OR 2.0; without Zhou 25% | **underpowered**† (the reported "15% vs 15% without Zhou" was copy-inflated) |
| Outliers come from distorted domain cores | Core RMSD +0.35 Å | +0.25 Å per distinct Fv (p=0.003); undistorted cores 1/10 outliers vs 11% | holds (4 distinct outliers) |
| Predicted orientation normal | +0.13, n.s. | +0.02 (−0.33 to 0.36) | null still null |
| Rare CDR lengths | OR 0.85 | 5/34 vs 26%, crude 0.49 (0.19–1.27) | null still null (the "depletion" is not robust) |
| H2-9, κ L1-17 | OR 2.45, 3.1–3.6 | 1.75, 3.6 (see 2b) | underpowered |
| Predicted H3 base kinked (VH) / VHH extended | 97% vs 98% / OR 0.31 | 31/31 vs 96% / 0.69 (0.07–7.2) | null still null / underpowered |
| ImmuneBuilder uncertainty at the site | +0.16 Å | EM +0.16 (0.03–0.29). 14/17 papers positive, Wilcoxon p=0.001. | **holds** |
| First-time V gene, J gene, V–J pair, length combination | n.s. | 2/34, 0/34, 4/34, 1/34; all n.s. crude and per paper | null still null |
| Distance to earlier PDB antibodies (adjusted) | M3 0.90 | Different antibody 0.95 (0.67–1.36) | null still null |
| Same antibody solved earlier | OR 0.32 | 0.40 (0.16–1.01); per paper 0.25 (0.09–0.70) | underpowered (largely by construction) |
| Germline identity / SHM, humanisation, rare canonical/Vernier/interface residues | n.s. | All n.s. (interface/Vernier EM point estimates ~1.9) | null still null |
| Non-human / nanobody | OR 2.97 / 1.9–2.4 | 7/34, all alpaca VHH (6 papers); M1 2.08, alpaca 3.0; per paper n.s. | underpowered |
| Rabbit / bovine | OR ≈4 / ≈11 | 0/34 with 1.7 / 1.4 expected; MDE 8 / 16 | **cannot test**†. The bovine cases (7D3M/7D3R/7D3K) *were* flagged by conkit; our own overlap matching and artifact ledger dropped them. |
| Absent from curated X-ray ≤3 Å statistics | 7% vs 21% | 3/34 vs 46% (38% method-standardised), OR 0.16. X-ray 0/10 vs 57%; EM p=0.04–0.07. | holds (EM marginal; partly circular) |

## 3. What this says

**Holds (CI excludes the null with one vote per paper):**
- The model-building fingerprint:
  - missing κ cis-Pro8 and cis-Pro95;
  - worse backbone geometry away from the slip;
  - the antibody-vs-rest gap (EM);
  - less β-sheet;
  - off-canonical non-H3 CDRs;
  - framework distortion (at paper level);
  - VH→VL co-occurrence.
- The within-entry nulls: the error chain is no worse than its error-free neighbours, and those neighbours are themselves
  far worse than controls. This again places the fingerprint on the entry or paper, not the strand.
- Unbuilt residues next to the slipped strand (cryo-EM and X-ray).
- The CDR3/FR4 share of errors and the positive C-flank shift.
- Site-level ImmuneBuilder uncertainty.
- Under-representation in curated X-ray statistics.

**Null still null:**
- Competition and racing, SARS-CoV-2 as such.
- People, labs, institutions, facilities and co-author links.
- Journals, funders, preprints, timing and processing.
- Germline novelty, distance to earlier antibodies, somatic mutation, humanisation, rare residues.
- Predicted orientation and H3 base.

**Underpowered (same direction):**
- Canonical pull and H2-9 / κ L1-17.
- Most per-mode features (mode A N-terminus, E5/I2, mode F).
- Niche antigen and fewer competitors.
- VH–VL orientation outliers.
- Nanobody excess.
- Software and starting-model records.

**Cannot test:**
- AlphaFold starting models (none existed in the window).
- Template-CDR3 mismatch (no independent case).
- Mode B, C-IGHV5, D−, D+ map and E features.
- Rabbit and bovine.
- Local resolution at the site.

**Attenuated, absent, or new:**
- **Cryo-EM poor fit at the site** is attenuated (OR 1.06, but the CI includes the full-set OR; strong in X-ray).
- **Krios deficit** is masked by selection: the conkit-missed errors carry it.
- **No-funding-record excess** is attenuated (China-led composition).
- **Constant-domain cis deficit** is absent outside the one all-trans paper, as in the full set.
- Three new or stronger signals appear: **China-led papers, less experienced teams, lower citations**. The first two are
  absent or reversed in the in-window errors that conkit missed, so they describe what conkit picks, not what causes errors.
  The citation effect sits entirely within China-led papers.

**On the general conclusion.** Every trend that defines "a model-building fingerprint, not people, antigens or unusual
antibodies" survives in this independent, paper-confirmed slice:
- Its signature measures hold.
- Its null claims stay null.
- The only person- or place-like signals that appear are explained by detector selection.

The strongest single piece of evidence is that κ Pro8 is 0% cis in both the conkit-flagged and the conkit-missed cryo-EM
errors of the same window, against 69% in controls. Two limits apply:
- The paper set cannot size the fingerprint, because conkit enriched for wholly misbuilt Fab models. That inflates the
  geometry, gap and VH→VL estimates.
- It cannot add anything about AlphaFold-era workflows.

## 4. Caveats

- **Power.** 34 units / 25 papers overall; 18 EM κ chains for the cis tests; 1–9 units per error mode. "Underpowered" and
  "cannot test" are not failures. Minimum detectable ORs per mode are ~10–50.
- **One paper dominates.** Zhou 2021 contributes 19/67 strict chains (39% of EM) and has no cis peptide anywhere. All key tests
  were re-run without it; all fingerprint trends survive except T9. Outside FR1 slips and Zhou, the cis evidence rests on about
  3 chains. Repeated Fab copies inflated several structural p-values (HC2, LC1, outliers) until de-duplicated.
- **Conkit's selection.** Conkit flagged about half of the window's first-occurrence errors (67 strict of 202 in-window
  first-occurrence chains in our scan). Relative to the errors it missed, the paper set is:
  - more often wholly misbuilt Fabs (EM 51% vs 32% of chains in entries where every V chain is wrong);
  - more often FR1 slips (43% vs 16%);
  - less often CDR3/FR4 slips (33% vs 59–64%);
  - more China-led and more SARS-CoV-2;
  - almost free of pre-2018-05 (AlphaFold2 training-set) errors: 3/67 chains.

  This selection affects the trends in different ways:
  - It inflates the geometry, gap and VH→VL effects.
  - It masks the Krios deficit and probably the mouse IGHV5 FR3 errors (7TE9/7TEB were missed).
  - It produces the China, experience and possibly the germline-like-antibody signals.
  - It cannot create the Pro8 signal.
  - Paper-level contrasts of flagged vs missed are weaker than chain-level ones (framework p=0.17–0.36; Ramachandran p=0.08–0.16).
- **The "conkit-missed" comparator is not clean.** It is defined as in-window first-occurrence errors not matched to a confirmed
  Table S1 error. It contains chains in Table S1-flagged entries: 30 of 134 chains overall, and 9 of 37 units in the antibody lens.
  In some of these, conkit flagged the error chain itself but SAbR/ANARCI overlap matching or our ledger dropped it (e.g. the
  bovine 7D3M/7D3R). Read selection statements as "suggests", not "shows".
- **SAbR/ANARCI confirmation also selects.** It confirms Table S1 errors touching IMGT ≥105 in 30/33 cases, against 16/43 at
  CDR1–FR3. The confirmed mode mix reflects both detectors.
- **Definitions differ slightly from the full set.**
  - "Beyond the site" excludes ±3 around every disagreeing residue.
  - Ramachandran and gap tests are population-wide, not the 183-unit matched sample.
  - Window controls are 61% X-ray, so compare only case-control contrasts across sets.
  - The canonical-CDR test uses only chains with existing ImmuneBuilder predictions.
  - ImmuneBuilder's training data (to 2021-07) may include some cases, which would bias its tests toward zero.
- **Multiplicity.** Several hundred tests across lenses, with no global correction. Isolated nominal results with no full-set
  counterpart (>500 kDa, letter format, X-ray Coot, first-time V gene p=0.048) should not be read as findings.
- **AlphaFold era untestable.** No entry in the window records an AlphaFold starting model. The "traced or threaded rather
  than docked" reading cannot be tested from metadata here either.

## 5. Files

All under `/home/user/SAbR/register_error_analysis/retest/paper_confirmed/`.

| Directory | Contents |
|---|---|
| `case_set/` | `build_cases.py`, `paper_errors_annotated.csv` (131 Table S1 errors with status), `chain_flags.csv.gz` (case/control flags; join to the deeper `chains.csv.gz` on pdb, chain) |
| `model_fingerprint/` | `notes.md`; `scripts/s01–s10`; tables `cis_tests.csv`, `pro8_flag.csv`, `rama_tests.csv`, `rama_within_entry.csv`, `fw_tests.csv`, `fw_or_per_sd.csv`, `fw_by_mode.csv`, `fw_within_entry.csv`, `vh_vl_tests.csv`, `sheet_tests.csv`, `canonical_tests.csv`, `selection_tests.csv`, `lopo.csv`, `chain_frame.csv.gz` |
| `local_hazards/` | `notes.md`; `scripts/lh01–lh07`; tables `pooled_hazards.csv`, `per_mode_features.csv`, `mode_distribution.csv`, `flank_asymmetry_paper.csv`, `template_cdr3_paper.csv`, `protocol_rates.csv`, `canonical_pull_paper.csv`, `length_tests_paper.csv`, `termini_paper.csv`, `fv_only_paper.csv`, `checks.log` |
| `who_where_when/` | `notes.md`; `scripts/w01–w07`; `tables/summary_all_tests.csv`, `w02_sars2_rank_bins.csv`, `w03_entity_enrichment_pseudonymised.csv` (names pseudonymised) |
| `antibody_ordinary/` | `notes.md`; `scripts/s01–s07`; tables `seq_tests.csv`, `struct_tests.csv`, `units_frame.csv.gz`, `struct_fv_frame.csv`, `struct_site_err.csv`, `dominant_paper.csv`, `summary_key_rows.csv` |
| `verify_model_fingerprint/`, `verify_local_hazards/`, `verify_who_where_when/`, `verify_antibody_ordinary/` | Adversarial checks: `notes.md`, scripts `v01–v06` (each re-derives the case set independently), logs and tables. These hold the corrected verdicts marked †. |
