# Do bad templates get reused? Register-error propagation across the PDB

> **Corrected version.** The first version compared each PDB entry using only the first chain that
> contained the error's sequence. In 215 of 2,308 families (9%) the reference chain was therefore an
> *unflagged* copy, for example 7MXL chain K instead of the flagged chain B. This version uses the
> flagged chain as reference and checks up to four copies per entry. An entry counts as carrying the
> flagged register if any copy has it. All numbers below come from the corrected run
> (`scripts/v2/`). The headline patterns are unchanged; individual cases changed (see "Cases").

## Method

1. **Group errors into families.** I mapped each of the 12,674 Table S1 errors to positions in the
   chain sequence, then grouped them into 2,978 families: same sequence, overlapping segment.
2. **Find every structure of that sequence.** I searched the whole PDB, any resolution, through 2026,
   for chains containing the sequence window around each error. That gave 319,735 chains in 182,141
   entries.
3. **Sample.** For the 2,308 families whose sequence appears in at least two papers, I took up to 12
   entries, preferring different papers: the flagged entries, the earliest and highest-resolution
   structures, and the most recent ones. Up to 4 chains per entry, flagged chains first, gave
   17,786 chains.
4. **Call the register.** The reference is the flagged chain of the earliest flagged entry. Each
   chain was superposed on it using Cα atoms outside the segment. It was called **same register** if
   the segment Cα atoms coincide (median < 2 Å), or **other register** if they coincide only after a
   shift of k residues.
5. **Check against the map, independently of AlphaFold.** For each entry I took the per-residue RSCC
   (X-ray) or Q-score (EM) of the segment and ranked it among all residues of the same chain.
6. **Check the stated starting model.** I pulled `pdbx_initial_refinement_model` for every entry to
   see which template each one says it started from.

**Validation.** 661 of 669 (98.8%) entries flagged in Table S1 come out in the same register as their
family's reference. 83 entries contain both registers in different copies.

## Results

| | All families (≥2 papers) | Strict set: both registers deposited, flagged one fits the map worse |
|---|---|---|
| Families | 2,234 | 355 |
| Flagged register in ≥2 papers / ≥2 labs | 55% / 44% | 45% / 37% |
| Error already present in the first structure of the protein | 89% | 67% |
| Later papers on the same protein that carry the flagged register | 56% | 26% |
| Still being deposited in 2023–26 | 19% | 10% |
| Later entries citing the flagged-register template: % that carry it | 83% | 62% |
| Later entries citing an other-register template: % that carry it | 0% | 0% |

* **The map arbitration favours AlphaFold.** In the 815 contested families, where both registers
  have been deposited, the flagged register fits the experimental map worse in 67% (Wilcoxon
  p = 5×10⁻³²). Using only structures at ≤3 Å, the figure is 79%.
* **Errors persist after a fix exists.** When the error came first (239 strict-set families), it took
  a median of 4.2 years before a structure with the other register appeared. After that, 19% of new
  papers still used the erroneous register.
* **There is no quality filter on reuse.** In the 111 contested families where the *flagged*
  register fits the map better (so the AlphaFold flag is probably wrong), later papers copy it at a
  similar rate: 20% vs 26%.
* **Weaker evidence.** 393 "uncontested" families (every structure shares the flagged register and
  the segment fits poorly) and 79 likely false flags (a ≤2.5 Å structure in the flagged register fits
  well) are listed in `tables/family_summary.csv`. Poor fit alone can't separate a misregistered
  strand from a flexible loop, so treat the uncontested ones as unresolved.

### How much of Table S1 is explained (`tables/attribution_*.csv`)

| | Flagged chains (of 12,674) | Flagged entries (of 2,954) |
|---|---|---|
| Any protein: flagged register recurs in ≥2 papers | 48% | 60% |
| Any protein: recurs in ≥2 labs | 41% | 52% |
| Any protein: ≥2 papers, map supports a real error | 15% | 27% |
| Any protein: ≥2 labs, strict set | 5.8% | 11% |
| SARS-CoV-2 entries (any flagged chain) | 0.7% | 1.4% |
| Antibody chains, all | 1.3% | 2.9% |

**Antibody errors** (165 flagged chains, 87 entries):

| Group | Chains | Entries |
|---|---|---|
| SARS-CoV-2 complex, deposited 2020 | 41 (25%) | 13 |
| SARS-CoV-2 complex, deposited 2021–22 | 28 (17%) | 14 |
| Error copied across labs | 14 (8%) | 12 |
| Error repeated in the same lab's papers | 1 | 1 |
| One-off: antibody re-used elsewhere, error not copied | 31 (19%) | 19 |
| One-off: antibody appears in one paper only | 47 (28%) | 29 |
| Could not be mapped | 3 | 3 |

Of the five antibodies whose error recurs across labs:
* **Fab 1B2:** the strict set, map supports a real error.
* **The 3H02 Fab:** fits poorly, uncontested.
* **A Fab light chain from 6OT0:** the flagged register fits *better*, so the flag is probably false.
* **Two undetermined:** a nanobody in 7M74 and the C10 Fab light chain.

## Cases

### Fab 1B2 (family 504): an erroneous high-resolution template copied for 8 years

| Year | Entry | Resolution | Lab | Register | Stated starting model |
|---|---|---|---|---|---|
| 2018 | 6C9U | 2.09 Å X-ray | Khosla | **flagged** | 2QO3, 1SBS |
| 2018 | 6MLK (Fab 3A6) | 2.45 Å X-ray | Khosla | other | 1MN6, 4LRI |
| 2020 | 7KRA, 7KTX (Fab DH4) | 3.2, 4.3 Å EM | Frost | other | – |
| 2021 | 7M7E–J (6 entries) | 3.2–4.3 Å EM | Khosla | **flagged** | – |
| 2021 | 7S6C, 7S6D | 3.1 Å EM | Fromme | **flagged** | 7S6B, **6C9U** |
| 2022 | 8EE0 | 2.65 Å X-ray | Khosla | **flagged** | **6C9U** |
| 2023 | 8TPW, 8TPX | 3.4–3.5 Å EM | Khosla | **flagged** | **6C9U** (8TPX) |
| 2025 | 9PAT, 9PAV, 9PC6 | 3.2–4.0 Å EM | Chen | **flagged** | – |
| 2026 | 11SZ | 2.94 Å EM | Cogan | **flagged** | – |

That is 15 Fab 1B2 entries after 6C9U, from 4 labs, carrying 6C9U's register.

> **Correction.** Family 504 is defined by a shared light-chain FR1 sequence window, not by antibody.
> 6MLK is Fab 3A6, and 7KRA/7KTX are Fab DH4. These are different antibodies whose light chains share
> 1B2's first 14 residues. All the entries carrying the flagged register are Fab 1B2. The other two
> antibodies, built independently on the same sequence, agree with AlphaFold2.

The map evidence is modest. The segment fits about equally well in 6C9U and 6MLK (31st vs 29th
percentile of their chains), and poorly in 8EE0 (4th percentile).

### REGN5713 light chain in 7MXL (families 200/201): a one-off error next to an unbuilt loop

7MXL is a 3.2 Å cryo-EM structure of Bet v 1 with Fabs of REGN5713, 5714 and 5715. The error is in
**chain B** (the REGN5713 light chain), in its κ constant domain (Cκ).

* **Residues 149–158 are not modelled.** That is the rest of strand C, the C–C′ loop and strand C′.
* **The flanking segments are shifted.** Residues 140–148 (end of strand B, the B–C loop, strand C)
  sit 1–2 positions downstream. Residues 159–171 (strand D and the D–E loop) sit 2 positions
  upstream. The register is restored at Thr172.
* **The numbering doesn't fit the gap.** About six positions are physically missing, but ten
  residues are numbered as missing.
* **The core tryptophan is misplaced.** In chain B, Trp148's ring is 14 Å from the domain centre;
  in chain K it is 3.9 Å from the centre and buried.
* **Chain K is correct.** It is the REGN5715 light chain, in the same map with the identical Cκ
  sequence and no gap, and it matches a 1.45 Å Cκ structure (7YV1) at 0.74 Å.
* **The segments fit the map worse in chain B than in chain K.** Q-score is 0.34 vs 0.43 for
  residues 141–148, and 0.36 vs 0.48 for 159–171. The two Cκ domains fit equally well overall.
* **The flag is correct and the error is not propagated.** The first version of this analysis called
  it a likely false flag copied across 10 labs, because it compared against chain K.

## Caveats

* **Sampling.** At most 12 entries and 4 chains per entry per family, so heavily studied proteins
  are under-sampled.
* **Unit of propagation.** "Papers" include follow-up papers from the same lab; ≥2 labs is the
  stricter measure.
* **Families are defined by a shared sequence window.** For conserved regions such as antibody
  constant domains, a family can span many different proteins that share that stretch.
* **Starting models are often unrecorded.** 58% of entries give none, so inheritance is inferred from
  structure in most cases.
* **Map fit is a relative measure.** It is averaged over the segment and is noisy at low resolution.
* **Time windows differ.** The flags come from the April 2022 snapshot. The structural tracing runs
  through 2026, but later entries were never flagged themselves; their register was only called by
  comparison.

## Files

* **`scripts/`:** the original pipeline (`tmpl1`–`tmpl8`). `tmpl4_register.py` provides the
  register-calling function used by v2.
* **`scripts/v2/`:** the corrected run.
  * `v2_candidates.py`: all chains per entry.
  * `v2_expand.py`: flagged-chain references and up to 4 copies per entry.
  * `v2_register.py`: chain-level calls aggregated to entries.
  * `v2_analyze.py`, `v2_mapfit.py`, `v2_arbitrate.py`, `v2_extras.py`, `v2_attribution.py`.
  * These scripts write to a session scratch directory set at the top of each one.
* **`tables/`:** corrected outputs. Chain- and entry-level register calls, map fit, family summaries
  with verdicts, starting-model inheritance, the likely-real error families, the attribution tables
  and logs.
