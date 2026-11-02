# Retesting every hypothesis with the SAbR/ANARCI detector

The earlier analyses (`../REPORT.md`, `../followup/`, `../template_propagation/`) relied on the
2024 conkit-validate flags. These flags stop in April 2022 and come from comparison against
AlphaFold2. Here every hypothesis is retested with the SAbR-vs-ANARCI disagreement detector
(`../sabr_anarci_scan/`). The detector is independent of AlphaFold and covers 16,636 antibody V
domains (or scFv chains) at 3–5 Å through 2026.

**Data added for this retest:**
* For every entry: per-residue map quality, recorded starting model, EM fitting protocol, modelling
  software, revision history, entries per paper, antigen novelty and competition, CDR lengths.
* 693 TCR V domains, as an intended same-fold comparator.
* 1,038 earlier high-resolution (<3 Å) structures of the same V regions (the templates), all scanned.

**Counting.** "Flag" means at least one run of ≥3 residues where SAbR and ANARCI disagree.
"Snapshot" means released by April 2022; "newer" means released after it. Models use single-domain
chains, adjust for method × resolution, period, chain type and V length, and cluster by entry.

## Verdicts

| # | Hypothesis (earlier evidence) | Retest | Verdict |
|---|---|---|---|
| 1 | Antibodies are flagged no more often than comparable folds | TCR V domains as comparator: SAbR's experimental TCR mode disagrees with ANARCI *systematically* in CDR2/FR3 (IMGT 57–74; α +4, β −1 in 75% of runs), so the comparison is invalid | **Untestable** with this detector |
| 2 | Antibody error rate rose through 2020–22 while other folds improved | Cryo-EM high through 2018–21 (13.2% in 2020H2, 9.5% in 2021H1), then **fell abruptly to 3.4% in 2021H2** and 2.5–7.6% after. X-ray flat at 1.3–1.9% | **Revised**: a plateau, then a sharp fall |
| 3 | About half the rise came from the shift to cryo-EM | ≤2019 → 2020–21 (3.1% → 6.1%): **all compositional** (+2.9 pp; within-stratum +0.0). 2020–21 → 2022–26 (6.1% → 3.6%): **all within-stratum** (−2.9 pp) | **Revised**: the rise was all method mix, the fall a real improvement |
| 4 | Cryo-EM worse than X-ray at the same resolution | OR 5.9 (snapshot), 3.3 (newer); unchanged after adjusting for local map quality | **Survives**, gap narrowing |
| 5 | Authors are careless with auxiliary antibodies | Snapshot: auxiliary *lower* (adjusted OR 0.30, 0.17–0.54). Newer: no difference (0.79). Within the same 40 entries, the reused reagents (Nb35, scFv16, BAG2, NabFab, sybodies…) are flagged **0/67** vs 4.5% (3/67) for the other antibodies | **Rejected**: the opposite, if anything |
| 6 | High-impact journals carry more errors (non-antibody domains, OR ~1.3) | Antibodies, snapshot: no effect. Newer: CNS 0.53 (n.s.), sister journals **0.57** (0.35–0.92) | **Not supported** for antibodies |
| 7 | More structures per paper, fewer errors per structure | n.s. in both eras (OR 1.04, 0.85 per doubling) | **Not supported** |
| 8 | Novel or contested targets carry more errors | Novel antigen: snapshot **OR 1.60** (1.03–2.47), newer n.s. Competition: no effect, or fewer errors | **Weak**: novelty only, pre-2022 |
| 9 | Short hold time (rushed) means more errors | Snapshot: 181–365 d holds had *more* (OR 1.96). Newer: 31–90 d holds had *fewer* (OR 0.17) | **Not supported** |
| 10 | SARS-CoV-2 2020 spike (6.4% vs 1.4%) | 2020: SARS-CoV-2 11.2% vs other viral 11.0%. 2021: **3.6% vs 7.9%** (adjusted OR 0.29) | **Rejected**: SARS-CoV-2 structures were as good or better |
| 11 | Antibody errors don't co-occur with errors in other chains | Another (different) antibody chain flagged in the same entry: 10.3% vs 2.9%, **OR 2.7** (1.5–4.7). Identical copies in an entry: **all copies flagged in 86%** | **Revised**: errors cluster within entries and are copied across symmetry copies |
| 12 | Errors cluster by lab | Dispersion **3.2×** expected (213 labs). Lab's snapshot excess predicts its newer rate only weakly (6.0% vs 4.7%, OR 1.27, n.s.) | **Survives**, weak persistence |
| 13 | Errors are rarely corrected | Only **9%** of conkit-flagged antibody entries ever had their coordinates revised. The flagged segment still disagrees today in **62%** of unrevised and 31% of revised ones | **Survives** |
| 14 | A prior high-res model of the same antibody protects (conkit, OR 0.32) | It depends on the template: clean template → **0.8%** flagged (n=3,651). Flagged template → **93%** (n=140), or **74%** (n=39, 9 families) excluding the HIV framework-insertion antibodies. 82% carry the *identical* error | **Revised**: templates transmit their register, right or wrong |
| 15 | Errors are copied from templates | Recorded starting model is a flagged structure of the same V: **84.5%** flagged (61% excluding HIV bnAbs) vs **0.3%** when it's clean. Flagged high-res templates' segments fit their own maps poorly (median 13th percentile; 83% < 25th) | **Survives, strongly** |
| 16 | Errors arise at chain ends, next to unbuilt loops and length-variable loops | Runs next to an unmodelled gap: **11.9% vs 0.9%** of matched windows. 30% start at the N-terminus. V domains with an unmodelled N-terminus: **18.1% vs 3.4%** flagged. N-terminal runs fit worse than N-terminal windows of clean domains (X-ray 9th vs 26th percentile). Atypical light-chain CDR1 lengths: λ L1=8 12.3% vs L1=9 1.9%; κ L1 11–12 ~5% vs 6 at 2.7%. Heavy CDR3 length: weak | **Survives, strongly** for gaps and termini |
| 17 | AlphaFold2 starting models reduced errors after 2021 | Timing fits: the drop is in 2021H2. But entries recording an AlphaFold starting model are no better (OR 0.91, 0.40–2.07; only 494 records). EM fitting: ab initio 2.6%, rigid body 3.5%, flexible fit 7.7% (newer) | **Unresolved**: timing consistent, direct test null; usage is under-recorded |
| 18 | conkit's low pre-2018 rate was an AlphaFold2 training-set artifact | conkit caught **10%** of SAbR framework errors in structures released before the AF2 training cutoff (X-ray 8%), vs **40.5%** after (X-ray 62%). SAbR's own X-ray rate is flat across the cutoff (1.4% vs 1.6%) | **Supported** |

## Why the errors arise and why they persist

**Origin of each flagged single-domain chain** (`tables/flag_origin.csv`; each run assigned to the
first category that applies):

| Likely origin | Snapshot | Newer |
|---|---|---|
| Same error already in an earlier high-res (<3 Å) template | 29% (unambiguous, excluding HIV FR3-insertion bnAbs: ~5%) | 12% (unambiguous ~3%) |
| Same error in an earlier 3–5 Å structure from another paper | 4% | 3% |
| At the N-terminus or next to an unmodelled gap | 26% | **40%** |
| FR3 insertion-convention region (possible artifact) | 3% | 6% |
| Not explained | 38% | 39% |

**How errors arise:**
1. **At chain ends and gaps.** When the first residues or a loop can't be built, the residues on
   either side are assigned to the wrong density. 7MXL's Cκ, and the light-chain N-terminal strands
   of 1B2 and many SARS-CoV-2 Fabs, are examples. This is now the largest identifiable source (40% of
   newer flags), and it is real: those segments fit the map worse than the same positions in clean
   domains.
2. **In new antibodies built into cryo-EM maps.** Cryo-EM carries 3–6× the X-ray rate at the same
   resolution, independent of how well the V domain fits the map overall. The problem is local, not
   global map quality.
3. **Not from rushing or lack of care.** Auxiliary reagents, high-impact papers, competitive targets,
   fast releases and SARS-CoV-2 structures are all as good or better.

**How errors persist:**
1. **Templates transmit their register.** Antibodies built from a clean earlier structure almost
   never carry an error (0.3–0.8%). Built from a flawed one, most do (61–93%). A handful of flawed
   high-resolution templates (18 of 1,038 scanned; e.g. 6C9U for Fab 1B2) seed a disproportionate
   share of copies.
2. **Copies within an entry.** Symmetry copies are built once and replicated (86% of copy groups are
   all flagged).
3. **Errors are almost never revised** (9% of flagged entries).
4. **Validation is blind exactly where it matters.** AlphaFold2-based checks miss errors in
   structures AlphaFold2 was trained on (10% vs 40% sensitivity). So erroneous pre-2018 templates,
   the ones most likely to be reused, look clean to the very method used to audit the archive. A
   predictor-independent check (SAbR vs ANARCI) finds them.
5. **Lab practice clusters** errors (3.2× dispersion) but persists only weakly across eras.

**Why rates fell after 2021.** The fall is a within-method, within-resolution improvement that
starts in 2021H2. Inherited errors also fell, from 29% to 12% of flags, so fewer flawed templates are
being reused. That's consistent with AlphaFold2 models replacing old experimental templates as
starting points, but entries rarely record this, and the direct test is null. In newer cryo-EM the
remaining errors increasingly sit at chain ends and gaps.

> **Update:** the "Not explained" category was investigated further in `unexplained/`. Most of it is
> CDR3–FR4 junction errors, an FR4 bulge after the W/F-G motif, and CDR1/CDR2 boundary errors. The
> genuinely unexplained share falls to 5–11% of independent error events.

## Caveats

* **Blind spots.** The detector misses errors inside CDRs, and can't adjudicate framework-insertion
  antibodies (HIV bnAbs) where SAbR and ANARCI legitimately differ. Those dominate the raw template
  numbers, so unambiguous figures are given alongside.
* **TCRs can't be used as a comparator.** SAbR's TCR mode is documented as experimental, and the
  systematic CDR2/FR3 offset shows why.
* **Sparse metadata.** Software and starting-model records are sparse (ModelAngelo is essentially
  never recorded), so the tool-level tests are weak.
* **Proxies.** "Lab" is the last author.

## Files

* `scripts/`:
  * `retest_master.py`: builds the per-V-domain table.
  * `retest_tests.py`: hypotheses 2–13 and 17.
  * `retest_tests2.py`: hypotheses 1, 14–16 and 18.
  * `retest_decompose.py`: assigns each flag a likely origin.
  * `gql_modelling.py`: modelling metadata.
  * `sabr_scan_tcr.py`: the TCR scan.
* `tables/`: `tests.log`, `tests2.log`, `decompose.log`, `flag_origin.csv`.
