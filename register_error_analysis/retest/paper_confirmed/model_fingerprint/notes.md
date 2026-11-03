# Re-test on the paper-confirmed set: the whole-model "fingerprint" (deeper README §4.1 and §7)

**Question.** Do the whole-model fingerprint trends hold if we keep only the register errors that the
Sanchez Rodriguez et al. (IUCrJ 2024) conkit-validate Table S1 flagged (PDB snapshot April 2022) and that our
SAbR/ANARCI scan also confirms?

**Short answer.** Yes, for nearly all of them.

- The κ cis-proline deficit, the backbone-geometry excess away from the slip, the antibody-vs-rest gap, the
  β-sheet deficit, the off-canonical CDRs and the VH→VL co-occurrence all reproduce on the paper-confirmed cases.
  Their 95% CIs exclude the null, also with one vote per paper.
- Several are *larger* than in the rest of the deeper set: Ramachandran beyond the site, the V-vs-rest gap and
  sheet (heterogeneity p 0.002–0.05).
- Framework Cα deviation is **attenuated**. It is about half the size of the rest of the set at chain level, and
  only +0.09 Å for CDR3/FR4 slips (full set 1.56 vs 1.02 Å). The effect is still present at paper level.
- The constant-domain cis deficit reproduces only through the same single all-trans paper. Without that paper it
  is absent, as in the full set.
- Within an entry, the error chain is again no worse than its error-free neighbours for Ramachandran geometry.
  For framework RMSD it shows the same +0.17 Å as the full set, now nominally significant in EM.
- Conkit's selection shapes the magnitudes (see the selection section below).

## Data and design

**Cases.** `paper_strict` from `paperconf/chain_flags.csv.gz` (copied in `../case_set/`): 67 chains, 38 entries,
34 antibody×paper units, 25 papers. 49 are EM and 18 X-ray. All were released by 2022-02-02.

**Sensitivity set.** `incl_copies` = strict + 1 loose-ledger case + 19 copies or near copies, giving 87 chains.

**Controls.** `paper_control`, in the paper window: 6,239 chains (2,679 EM, 3,560 X-ray). For the
framework-RMSD test a random subset was used: 300 EM entries and 300 X-ray entries, one chain each, plus 230
clean window chains already computed by the DE2 verifier.

**Heterogeneity comparator (`rest_case`).** Every deeper first-occurrence error chain that is not paper-confirmed:
311 chains (269 EM, 42 X-ray), all years. It is compared with all clean chains (`err_any==0`) of all years.
Heterogeneity is reported two ways:

- a z-test of the two estimates (log OR or difference);
- a 3-group model (paper vs rest vs clean) with a Wald test of paper = rest.

A same-era version (conkit-missed errors in the window) is in `selection_tests.csv`.

**Units.** Each test is reported at three levels:

- chain level, adjusted for year bin, resolution and chain class (+method), with SEs clustered by paper;
- **paper-weighted**: one vote per paper per class;
- **paper level**: paper means. For binary features this is a Mann-Whitney test of paper cis fractions.

**Power.** The minimum detectable OR at 80% power (`mde_OR_*`) uses the expected cells. For continuous
measures, 2.8×SE. Amyloid entries are excluded throughout.

**Dominant paper.** One paper dominates the case set: 10.1038/s41422-021-00487-9 (Zhou Q., Cell Res 2021,
7CZR–7D00). It holds 17 of the 67 strict chains and 12 of the 18 EM κ chains. Every test was re-run without it,
and with leave-one-paper-out (`lopo.csv`).

## Results

| # | Trend (full-set value, source) | Paper-confirmed strict (95% CI, p, n) | Paper-weighted / per paper | incl. copies | Heterogeneity vs rest | Verdict |
|---|---|---|---|---|---|---|
| T1 | κ Pro8 cis: 7% (5/70) vs 64%, chain OR 0.036, paper-weighted 0.115, non-FR1 unit OR 0.14 (verify/pdb_meta_cis_pro_kappa8_*) | **EM 0/18 (6 papers) vs 615/895 (69%)**, Haldane OR 0.012 (0.0007–0.21), Fisher p=1e-9. All methods 1/19 vs 78%, adj OR 0.019 (0.002–0.17), p=4e-4. Non-FR1: EM 0/8 (4 papers), p=1e-4. X-ray: 1 chain (cis). | All methods paper-weighted OR 0.067 (0.011–0.40), p=0.003; non-FR1 0.095 (0.016–0.56). Each of the 6 EM case papers has 0% cis vs 69% in control papers (MW p=3e-4). Without the dominant paper: EM 0/6, OR 0.034, p=9e-4. | EM 0/31 (10 papers), OR 0.007 | Rest EM 5/49 (10%), OR 0.057–0.071; z-test p=0.25–0.32 | **holds** (EM); X-ray cannot test (n=1) |
| T1b | Trans Pro8 as a flag: error rate 5.7% if trans vs 0.26% if cis (EM, all years) | Window EM: paper-confirmed error 18/298 (6.0%) if trans vs 0/615 if cis. Sensitivity 100%, specificity 69%, NPV 100%. Any deeper first-occurrence error in the window: 38/333 (11.4%) vs 0/629. | — | 31/311 (10.0%) vs 0/615 | — | **holds** in EM. X-ray: no use (0/162 vs 1/999) |
| T1c | Pro8 deficit is a paper-level habit (within paper 5 lower / 1 higher / 9 ties) | Only 3 case papers have clean κ chains: 1 lower / 0 higher / 2 ties. In 5 of 7 case papers every κ chain, case or clean, is trans (Zhou 2021: 22/22 κ chains trans). | — | 9 of 11 papers all-trans | — | **null still null** within paper; the paper-level habit reading is supported |
| T2 | κ L3 cis-Pro95 (IMGT 115): 23% (6/26) vs 76%, OR 0.077; non-CDR3/FR4 12.5% vs 62%, OR 0.079 (struct_unusual, pdb_meta) | EM 0/17 (5 papers) vs 445/656 (68%), Haldane OR 0.014, p=7e-9. All methods 2/19 vs 78%, adj OR 0.041 (0.005–0.36), p=0.004. No run reaching IMGT≥104: EM 0/10 (4 papers), p=1e-5; all methods adj OR 0.067 (0.008–0.53). | Paper-weighted 0.071 (0.011–0.46), p=0.006. Without the dominant paper: EM 0/5, OR 0.042, p=0.003. | 0/20 EM | Paper set stronger than rest. Rest is 50% vs 62% because it includes slips at L3: p=0.009–0.016. With the position filter, p=0.13–0.19. | **holds**. The "elsewhere in 3D" sub-claim cannot be tested: only 1 chain is non-FR1 and away from L3 |
| T3 | Ramachandran allowed beyond the site: +3.75/100 (sample), one per paper +2.6, X-ray +3.7 (verify/struct_unusual_beyond_site_*); this metric on the full EM population: +3.1 | **EM 14.7 vs 6.6%: +8.2 (3.1–13.4), p=0.002, 48 vs 2,632 chains, 16 papers.** X-ray 10.6 vs 6.7: +4.7 (1.1–8.3), p=0.01, 18 chains, 10 papers. All methods +7.3 (2.8–11.9). | Paper level: EM +6.0 (1.7–10.3); X-ray +4.6 (1.4–7.7); all +6.0 (2.9–9.0), p=1e-4. Without the dominant paper: EM +4.5, p=0.03. | EM +5.6 (0.4–10.7); all +5.5 | Rest EM +2.3 (0.8–3.7). Paper > rest: z p=0.03, 3-group p=0.017. X-ray: rest +2.4, p_het=0.26. | **holds**, EM and X-ray (larger than the full set) |
| T4 | Antibody vs rest of the same model, Ramachandran gap: +2.16 pp, population EM (verify/struct_unusual_antibody_vs_rest_*) | **EM entries +6.5 pp (3.6–9.4), p=9e-6, 27 vs 616 entries, 17 papers.** X-ray +1.7 (−0.4 to 3.8), p=0.12, 11 entries. Gap using only disagreement-free V chains: EM +4.1 (1.2–7.0), p=0.006 (full set +1.1). | Paper level: EM +6.5 (2.9–10.1); all +5.1 (2.4–7.8). Without the dominant paper: EM +5.7, p=0.001. | EM +3.8 (1.0–6.7) | Rest EM +1.6. Paper > rest: z p=0.002, 3-group p=0.001 | **holds** in EM; X-ray **same direction, underpowered** (MDE 3.0 pp) |
| T4c | "Antigen chains are mostly fine" (non-V % allowed n.s.; clash excess is paper-level) | Non-V % allowed: EM +1.7 (−0.1 to 3.4), p=0.06; **X-ray +4.2 (0.9–7.6), p=0.014** | Paper level: X-ray +4.5, p=0.003 | — | Rest all +0.8, p=0.003; p_het 0.07–0.28 | **attenuated** (the rest of these X-ray models is also worse) |
| T5 | Framework Cα RMSD (run excluded): 1.56 vs 1.02 Å (D/E), OR 3.2/SD; other modes OR 3.1–3.5 (verify/case_files_DE_DE2_*) | EM 1.20 vs 1.04 Å: +0.17 (−0.02 to 0.36), p=0.09; OR per 0.47 Å 1.62 (1.00–2.61), p=0.05. X-ray 1.12 vs 0.89: +0.15 (−0.04 to 0.35), p=0.12. D/E chains only (EM): 1.12 vs 1.04, +0.09 (−0.12 to 0.31). Non-D/E: +0.23 (0.01–0.44). | Paper-weighted EM +0.32 (0.11–0.52), p=0.002; OR/SD 2.3 (1.4–3.9). Paper level EM +0.32, p=0.003. Leave-one-paper-out: max p 0.12. | EM +0.14 (−0.01 to 0.29) | Rest EM +0.39 (0.26–0.52), OR/SD 2.5. Paper < rest: z p=0.06, 3-group p=0.02 | **attenuated** |
| T6 | VH error → VL error in the same entry: 34% vs 1.3%, paper/antibody-level OR 15–25; any mode OR 13–18 (verify/case_files_DE_DE3_*) | Window EM entries, VH paper error → VL any first-occurrence error: **7/14 vs 17/554, OR 31 (10–94), p=3e-7**; adjusted 31 (5.8–170). VL paper-confirmed outcome: 4/14 vs 7/554, OR 31. VH D/E: 4/6 vs 20/562, OR 48. X-ray 1/5 vs 5/751, OR 45, p=0.04. | Fab unit: OR 17.5, p=8e-5. Paper: 4/9 vs 10/273, OR 20.5 (5.1–82), p=4e-4; all methods 5/14 vs 15/819, OR 30 (9–96). | EM entries 7/17, OR 22 | Rest (all years) EM OR 19.9 (10–38): p_het=0.51. Rest in window OR 8.9: p=0.21 | **holds** (EM; X-ray same direction, n=5) |
| T7a | Within an entry, the error chain is no worse than its clean V chains: Ramachandran +0.8 (−0.4 to 2.1), p=0.2 | EM, 15 entries: −0.5 (−3.7 to 2.7), p=0.77; all methods (24 entries) −0.2 (−2.8 to 2.3). Error-free neighbours vs window controls: **+5.0 (1.3–8.6), p=0.008** (full set +0.6 n.s.). | 18 papers: mean paired difference −0.35, p=0.64 | −0.4 (−2.5 to 1.7) | p_het 0.71 | **null still null**. The entry-level reading is supported more strongly than in the full set. |
| T7b | Same, framework RMSD: +0.17 Å (−0.09 to 0.43), p=0.20 | EM, 15 entries: +0.17 (0.004–0.33), p=0.044; all methods +0.16 (−0.04 to 0.35), p=0.12. Neighbours vs window controls: +0.21 (0.04–0.38). | Error chain higher in 10/15 entries, Wilcoxon p=0.17 | +0.26 (0.09–0.42), p=0.003 | p_het 0.73 | Same size as the full set. **Same direction, underpowered** as a "no worse" null: a small chain-specific excess cannot be excluded. |
| T8a | Less β-sheet away from the site: 0.39 vs 0.44, OR 0.54 per 0.08; within entry, lower in 70% of 52 pairs (pdb_meta) | EM 0.33 vs 0.43: −0.11 (−0.15 to −0.07), p=2e-7. X-ray 0.38 vs 0.47: −0.09 (−0.16 to −0.03), p=0.005. Within entry: lower in 9/10 same-class pairs, p=0.02. | Paper level: EM −0.10 (−0.16 to −0.04); X-ray −0.06 (−0.10 to −0.02). Without the dominant paper: −0.10, p=0.003. | EM −0.086 | Rest EM −0.06. Paper stronger: p=0.04–0.05 | **holds** |
| T8b | Non-H3 CDRs in the predicted canonical backbone bins: 0.76 vs 0.89; −0.09 in error-free chains (struct_unusual) | Subset only: these chains have ImmuneBuilder predictions in the struct_unusual EM sample. **20 chains, 14 papers: 0.65 vs 0.88 (window controls), −0.23 (−0.34 to −0.12), p=4e-5.** Error-free chains of case entries: 0.74 vs 0.875, −0.14 (−0.26 to −0.02), n=8. | Paper level −0.24, p=4e-4 | −0.22 | Rest −0.14: p_het=0.17 | **holds** (EM subset). X-ray cannot test (no predictions) |
| T9 | Constant-domain cis deficit: Cκ 21% vs 81%, CH1 42% vs 72%. Weakened: rests on one all-trans paper; CH1 null without it (verify/pdb_meta_cis_constant_and_L3_*) | Cκ EM 1/13 vs 90% (12 of 13 chains from the dominant paper). CH1 EM 1/8 vs 88%. Pooled, all methods: 12/32 vs 94%, adj OR 0.074 (0.018–0.31). **Without the dominant paper: Cκ 4/4, CH1 8/8, pooled 12/13 (92%) vs 94%, OR 0.95 (0.14–6.4).** | Pooled paper-weighted OR 0.61 (0.14–2.6), p=0.51. Paper level p=0.51. | Without the dominant paper 24/25 cis | Rest pooled OR 0.24 (0.10–0.57). Crude p_het=0.0002, but driven by the one paper | **absent or reversed** apart from the one all-trans paper (as in the full set; low power, MDE OR ≈0.06) |

## How conkit's selection bears on these trends (`selection_tests.csv`, `selection_descriptors.csv`)

**Who conkit picks.** In the same window, the deeper scan has 110 first-occurrence error chains that are not
paper-confirmed (88 EM). Compared with them, the paper-confirmed chains differ in three ways:

- more often sit in entries where every V chain is wrong: 51% vs 32% (EM);
- are more often FR1 slips (mode A): 43% vs 16%;
- are concentrated in one paper: 39% of EM chains vs ≤11% for any single paper among the missed chains.

The missed chains are mostly CDR3/FR4 slips (64%).

**What this does to the estimates.** Against the same window controls:

| Measure | Paper-confirmed | Conkit-missed | Paper minus missed |
|---|---|---|---|
| Ramachandran allowed beyond the site | +8.3 | +1.9 (n.s.) | p=0.006 |
| Clashes | +24.8 | +10.0 | p=3e-4 |
| Framework RMSD | +0.19 Å | +0.50 Å | p=0.03 |
| Sheet | −0.11 | −0.07 | n.s. |
| κ Pro8 cis (EM) | 0/18 | 0/17 | — |

So conkit's set enriches badly refined, wholly misbuilt Fab models and FR1 slips. That inflates the
Ramachandran, clash and V-vs-rest effects, and it explains why these are *larger* than in the rest of the deeper
set. It also misses many CDR3/FR4 slips in distorted frameworks, which deflates the framework-RMSD effect. The
attenuation of T5 is therefore at least partly selection, not evidence against the full-set trend.

**Pro8.** Conkit selection cannot produce the Pro8 signal: the conkit-missed EM κ errors in the window are also
0/17 cis.

**The AlphaFold2 training set.** Conkit is nearly blind to errors released before 2018-05. Only 3 of 67 strict
chains (4.5%) are that old, against 18% of the missed chains and 68% of the missed X-ray chains. The paper X-ray
cases are therefore mostly post-2018 X-ray. This removes most of the older X-ray errors from the paper-set X-ray
estimates, but the X-ray trends T3, T8a and T4c still hold.

**AlphaFold starting models.** None of the trends here depends on AlphaFold starting models. In the paper set,
44/67 strict chains record no starting model, 23 record an experimental PDB model and none an AlphaFold model.
The "traced or threaded rather than docked" interpretation remains untestable from metadata, as in the full set.

**Possible circularity.** Conkit flags errors from AF2-contact disagreements. A globally misbuilt V domain gives
more such disagreements, so the geometry/sheet fingerprint might itself help conkit see the error. This is a
reason to expect, not to discount, the larger geometry effects in this set. The SAbR noise test in the deeper
verify shows that the geometry does not create SAbR runs.

## Caveats

- **Small numbers.** 18 EM κ chains in 6 papers carry the cis tests. 11 EM and 10 X-ray papers carry the X-ray
  arms. Many X-ray-only rows (κ cis, constant domains) have 1–7 chains.
- **Zero cells.** Zero cells (0/18 cis) make adjusted logits inestimable. The tables report Haldane ORs, Fisher p
  and paper-level Mann-Whitney instead.
- **Beyond-site definition.** "Beyond the site" here excludes ±3 residues around *every* SAbR/ANARCI-disagreeing
  residue (the verifier's stricter Vex3 bin). The full-set +3.75 excluded only the run ±3. On the full EM
  population this definition gives +3.1.
- **Population vs sample.** The Ramachandran and gap tests are population-wide: every case and control chain or
  entry with a validation report. The full-set values came from a matched sample of 183 units, so absolute sizes
  are not identical in construction.
- **Framework-RMSD controls.** These are a random 600-entry subset of the window controls plus the DE2
  verifier's window controls. CA atoms come from RCSB ModelServer. The code is the verifier's, and it
  reproduces their values to 2e-5 Å.
- **Canonical-CDR test.** T8b is limited to chains that already had ImmuneBuilder predictions. ImmuneBuilder is
  not installed here.
- **Multiple testing.** Many tests; no correction was applied across this lens.

## Files

All files are under `retest/paper_confirmed/model_fingerprint/`.

**Scripts** (`scripts/`):

| Script | What it does |
|---|---|
| `f00_fetch_vrpt_xray.py` | Fetches X-ray validation XML into the scratchpad `paperfp/vrpt`. |
| `s01_frame.py` | Builds the chain frame and groups → `tables/chain_frame.csv.gz`. |
| `s02_cis.py` | T1, T1b, T1c, T2, T9 → `cis_tests.csv`, `pro8_flag.csv`, `pro8_within_paper.csv`. |
| `s03_parse_vrpt_xray.py` | X-ray residue tallies; a copy of the verifier's `v02_parse.py`. |
| `s04_rama.py` | T3, T4, T7a → `rama_tests.csv`, `rama_within_entry.csv`, `rama_chain.csv.gz`, `rama_entry.csv.gz`. |
| `s05_fw_fetch_compute.py` | Framework RMSD → `fw_computed.csv`. |
| `s06_fw_tests.py` | T5, T7b → `fw_tests.csv`, `fw_or_per_sd.csv`, `fw_within_entry.csv`, `fw_by_mode.csv`. |
| `s07_vh_vl.py` | T6 → `vh_vl_tests.csv`. |
| `s08_sheet_canonical.py` | T8a, T8b → `sheet_tests.csv`, `sheet_within_entry.csv`, `canonical_tests.csv`. |
| `s09_selection.py` | Conkit-selection diagnostics → `selection_tests.csv`, `selection_descriptors.csv`. |
| `s10_lopo.py` | Leave-one-paper-out → `lopo.csv`. |
| `binsuite.py`, `common.py` | Shared statistics. |
| `show.py` | Compact printer; output in `tables/*_compact.txt`. |
