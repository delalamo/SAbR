# Do bad templates get reused? Register-error propagation across the PDB

## Method

1. **Group errors into families.** I mapped each of the 12,674 Table S1 errors to positions in the
   chain sequence, then grouped them into **2,978 error families**: same sequence, overlapping
   segment.
2. **Find every structure of that sequence.** I searched the whole PDB, any resolution, through 2026,
   for chains containing the sequence window around each error.
3. **Sample.** For the 2,308 families whose sequence appears in at least two papers, I took up to 12
   entries, preferring different papers: the flagged entries, the earliest and highest-resolution
   structures, and the most recent ones. That gave 10,812 chains.
4. **Call the register.** Each chain was superposed on the flagged reference using Cα atoms outside
   the segment. It was called **same register** if the segment Cα atoms coincide (median < 2 Å), or
   **other register** if they coincide only after a shift of k residues.
5. **Check against the map, independently of AlphaFold.** For each entry I took the per-residue RSCC
   (X-ray) or Q-score (EM) of the segment and ranked it among all residues of the same chain.
6. **Check the stated starting model.** I pulled `pdbx_initial_refinement_model` for every entry to
   see which template each one says it started from.

**Validation.** 648 of 657 (98.6%) entries flagged in Table S1 come out in the same register as their
family's reference, so the structural calls agree with the original flags.

## Results

| | All families (≥2 papers) | Strict set: both registers deposited, flagged one fits the map worse |
|---|---|---|
| Families | 2,222 | 352 |
| Flagged register in ≥2 papers / ≥2 labs | 56% / 45% | 48% / 39% |
| Error already present in the first structure of the protein | 90% | 71% |
| Later papers on the same protein that carry the flagged register | 58% | 26% |
| Still being deposited in 2023–26 | 20% | 11% |
| Later entries citing the flagged-register template as starting model: % that carry it | 84% | 63% |
| Later entries citing an other-register template: % that carry the flagged register | 0.8% | 0% |

* **The map arbitration favours AlphaFold.** In the 780 contested families, where both registers have
  been deposited, the flagged register fits the experimental map worse in 67% (Wilcoxon
  p = 2×10⁻²⁹). Using only structures at ≤3 Å, the figure is 73%.
* **Errors persist after a fix exists.** When the error came first, it took a median of 4.0 years
  before a structure with the other register appeared. After that, 20% of new papers still used the
  erroneous register.
* **There is no quality filter on reuse.** In contested families where the *flagged* register fits
  the map better, so the AlphaFold flag is probably wrong, later papers copy the flagged register at
  the same rate (23% vs 26%). Templates propagate regardless of correctness.
* **Weaker evidence.** 406 "uncontested" families (every structure shares the flagged register, and
  the segment fits poorly) and 86 likely false flags (a ≤2.5 Å structure in the flagged register fits
  well) are listed in `tables/family_summary.csv`. Poor fit alone can't separate a misregistered
  strand from a flexible loop, so treat the uncontested ones as unresolved.

## Example: Fab 1B2 (family 504)

| Year | Entry | Resolution | Lab | Register | Stated starting model |
|---|---|---|---|---|---|
| 2018 | 6C9U | 2.09 Å X-ray | Khosla | **flagged** | 2QO3, 1SBS |
| 2018 | 6MLK | 2.45 Å X-ray | Khosla | other | 1MN6, 4LRI |
| 2020 | 7KRA, 7KTX | 3.2, 4.3 Å EM | Frost | other | – |
| 2021 | 7M7E–J (6 entries) | 3.2–4.3 Å EM | Khosla | **flagged** | – |
| 2021 | 7S6C, 7S6D | 3.1 Å EM | Fromme | **flagged** | 7S6B, **6C9U** |
| 2022 | 8EE0 | 2.65 Å X-ray | Khosla | **flagged** | **6C9U** |
| 2023 | 8TPW, 8TPX | 3.4–3.5 Å EM | Khosla | **flagged** | **6C9U** (8TPX) |
| 2025 | 9PAT, 9PAV, 9PC6 | 3.2–4.0 Å EM | Chen | **flagged** | – |
| 2026 | 11SZ | 2.94 Å EM | Cogan | **flagged** | – |

The register that independent builds produced (6MLK, Frost) and that AlphaFold2 predicts was
abandoned. The register of the higher-resolution 6C9U was copied by everyone after 2021.

The map evidence for this case is modest. The segment fits about equally well in the two crystal
structures (31st vs 29th percentile of their chains), and poorly in 8EE0 (4th percentile). Settling
1B2 needs a direct look at the 6C9U and 6MLK density.

## Caveats

* **Sampling.** At most 12 entries per family, so heavily studied proteins are under-sampled.
* **Unit of propagation.** "Papers" include follow-up papers from the same lab; ≥2 labs is the
  stricter measure.
* **Starting models are often unrecorded.** 58% of entries give none, so inheritance is inferred from
  structure in most cases.
* **Map fit is a relative measure.** It is averaged over the segment and is noisy at low resolution.
* **Time windows differ.** The flags come from the April 2022 snapshot. The structural tracing runs
  through 2026, but later entries were never flagged themselves; their register was only called by
  comparison.

## Files

* `scripts/`, in pipeline order:
  * `tmpl1_families.py` → `tmpl2_candidates.py` → `tmpl3_fetch.py` and `tmpl3b_fetch_pdbe.py`
    (`ca_fetch.py` is the shared fetch helper)
  * `tmpl_irm.py` → `tmpl4_register.py` → `tmpl5_analyze.py`
  * `tmpl6_mapfit.py` (uses `../followup/scripts/gql_local.py`) → `tmpl7_arbitrate.py` →
    `tmpl8_extras.py`
* `tables/`: per-entry register calls, map fit, family summaries with verdicts, starting-model
  inheritance, the list of likely-real error families, and logs.
