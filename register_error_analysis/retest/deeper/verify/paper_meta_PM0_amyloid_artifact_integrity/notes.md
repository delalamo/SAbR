# Check of PM0_amyloid_artifact (paper_meta): data integrity and statistics

**Verdict: confirmed.** The headline numbers reproduce exactly, and an independent geometric test shows that these chains are not in the Ig fold. Two minor errors and one scope caveat are listed below.

## What reproduces exactly (`counts_check.py`, `counts_check.log`)

* **Entries.** There are 15 AL λ light-chain fibril entries in the analysis set, all cryo-EM, and all 15 have `err_first`:
  * 6ic3, 6z1o, 7nsl;
  * 9faa, 9fab, 9fac;
  * 9els;
  * 9liv, 9liw, 9lix, 9liy, 9lj0, 9lja, 9ljb;
  * 9v91.
* **Independent search.** A second search found no other fibril in the set. It used title keywords plus a structural signature: entries with no heavy chain and ≥3 identical V chains. The only extra hit, 7o85, is a normal Fab with Q = 0.91.
* **Units.** 12 of the 183 cryo-EM error units are amyloid (12 of 4,260 EM units in all).
* **Runs.** 77 of the 488 first-occurrence runs are amyloid (15.8%), which is 77 of 428 EM runs.
  * Run length: median 89 residues, against 8 for all other runs.
  * |Δ|: median 17, against 1.
  * 6ic3 runs from IMGT 16 to 122 with Δ −48, as stated.
* **Pipeline flag.** No amyloid run carries `copied_or_artifact`, so the pipeline's artifact filter missed them.
* **Exclusion in paper_meta.** In `paper_base.csv` the title regex flags exactly 10 paper keys and gives no false positives. Every downstream script removes them:
  * `analysis.py`, `competition.py`, `sensitivity.py` and `network.py` keep only the `error` and `clean` groups;
  * the GEE uses an inner merge on that set;
  * `within_paper.py` drops them explicitly.
* **Author hit before exclusion** (`author_enrich_with_amyloid.py`, which copies the `network.py` logic). With the 5 published amyloid papers counted as errors:
  * Radamaker, Fändrich, Schönland, Hegenbart, Huhn and Schmidt M are each 3/3, with 0.18 expected, p = 2.05e-4, q = 0.056;
  * "Wang K" is essentially tied (p = 2.5e-4, q = 0.058).
  * After exclusion the top hit is Wang K with q = 0.29, which matches the notes.

## The independent fold check (`fold_check.py`, `fold_check_chains.csv`)

**Chains compared:**
* all 77 amyloid chains;
* 178 non-amyloid cryo-EM error chains (one per entry);
* 150 clean cryo-EM light chains.

**Metric 1: Q_native.** This is the fraction of the consensus Ig-V Cα contacts present in the chain. Positions are taken from sequence-based ANARCI IMGT numbers.

| Group | Q_native |
|---|---|
| Amyloid | 0.014–0.091 |
| Clean | 0.86–1.00 |
| Non-amyloid errors | 0.11–1.00 |

**Metric 2: planarity.** This is the SD of Cα coordinates along the third principal axis.

| Group | Planarity (Å) |
|---|---|
| Amyloid | 1.03–2.96 (flat fibril layers) |
| All 328 Ig-domain chains | 4.98–7.81 |

The two ranges do not overlap. The amyloid chains are also elongated: first-axis SD 21 Å, against 10 Å.

**Disagreement.** SAbR disagrees with ANARCI at 69–100% of residues (median 100%), and in 6ic3 it even assigns negative IMGT numbers.

**Other whole-domain runs are a different thing.** Two non-amyloid chains have whole-domain runs: 7c80 H (Q = 0.11) and 3jcx H (Q = 0.18). Both have normal Ig geometry (planarity 5.9–6.6 Å). They are whole-chain mis-threadings of a real Ig fold, not fibrils, so they stay in the error set.

## Minor corrections to the finding

1. **Mode counts.** Per entry the modes are A_FR1Nterm 7, F_CDR1 2 and other 6, not 8/2/6 (which adds up to 16). Per run they are A 34, F 8 and other 35.
2. **"Top co-author hit."** This is true, but it is shared by 6 co-authors of the same 3 papers and is only marginally ahead of Wang K (q 0.056 vs 0.058). It is not a standout at the author level.
   * The *class*-level link, by contrast, is overwhelming: 15/15 entries against a 6.3% EM base rate, binomial p = 1e-18. This is definitional.
3. **"All 15" means all 15 that entered the analysis set.** The PDB has 3 more AL fibrils at 3–5 Å from cryo-EM:
   * 6HUD (4.0 Å) and 8CPE (4.0 Å) were scanned by SAbR, show about 90–100% disagreement, and dropped out of chains.csv (77 residues modelled);
   * 6Z1I (3.4 Å) was not scanned.
   * Every AL fibril that was scanned (17/17) is flagged.

## Why excluding them matters, beyond paper_meta

If the amyloid entries are left in, they distort several other results:

* **Run origin.** They make up 74 of the 217 runs in the origin class "chain end or next to an unbuilt loop". That class's share of all runs is 44.5% with them and 34.8% without, which inflates the "errors sit next to unbuilt residues" finding.
* **Mode "other".** They are 35 of the 47 "other" runs.
* **λ chains.** They are 12 of the 24 cryo-EM λ error units. The λ unit error rate is 5.0% with them and 2.6% without (κ is 3.2%, heavy chains 4.8%), so keeping them creates a spurious λ signal.
* **2024–25 error rate.** For cryo-EM entries it is 5.3% with them and 4.1% without.

**Not every lens has excluded them yet:**

* `struct_unusual/tables/unit_metrics.csv` has 12 amyloid units among its 183 cases;
* `literature_dean/tables/apparent_length_error_units.csv` has 10 amyloid rows out of 107;
* `seq_novelty/tables/cases_positive_features.csv` includes 6ic3.

None of these three lenses' scripts or notes mention amyloid.
