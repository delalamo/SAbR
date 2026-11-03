# Verification: pdb_meta / amyloid_artifact (integrity)

The verifier returned this verdict without writing a notes file. The text below is its structured return, copied unchanged from the workflow log.

**Verdict: confirmed**

## Reason

I tried to refute this finding and could not. Every headline count reproduces exactly from the shared data. A separate structural test that does not use SAbR shows the 77 fibril chains are not in an Ig fold, and they separate completely from every other error chain. Excluding them is therefore justified on structural grounds, not just by definition. There are a few minor numerical and wording errors, but none changes the conclusion.

Checks on the case set and the joins:
- The investigator's title regex `amyloid(?!-beta)` picks out exactly the 15 entries they list, and the same 77 chains at chain level. A broader search for amyloid, fibril, light chain, myeloma or AL titles, and for entries with only light chains, finds no other fibril in the analysis set; the only other amyloid hit is 4M1C, an X-ray Fab–IDE complex with Abeta that is correctly excluded.
- The exclusion is applied in every downstream table I checked: scripts 05, 06 and 08 (and so 09), and script 13 through the analysis frame. There are 0 fibril rows in `chain_cis_sheet` and in `local_resolution_sites`.
- The definition is not circular. It is based on titles, which are independent of the error flags.

Recounts:
- 15/193 EM error entries.
- 0/3,782 clean entries (all methods) are fibrils.
- 77/488 first-occurrence runs (15.8%).
- 12/218 error units. 218 counts all methods; within cryo-EM it is 12/183.
- 12/26 λ error units and 6/66 mode-A units, both across all methods (cryo-EM: 12/24 and 6/59).
- All 77 fibril chains are flagged (`err_first`). Fibrils also account for 5/10 'other'-mode units and 35/47 'other' runs.
- Note that the denominators are mixed: the 193 is EM entries, while the unit and run counts cover all methods.

Antibody-alone reproduces exactly:
- With fibrils: 16/193 vs 14/2,574 (Fisher OR 16.5, p = 1e-11; adjusted for year, resolution and number of V chains with paper clusters, OR 20.7, p = 1.6e-7).
- Without fibrils: 1/178 vs 14/2,574 (p = 1; adjusted OR 1.16, p = 0.89).
- The one remaining error entry, 9ZXD, is mislabelled. It is a nanobody–AT1R fusion with a BRIL Fab, so the true residual is 0/178. Some controls are mislabelled the same way: 9ZXC and 9D7U–9D7X.

λ CDR1-length signal (L1 = 8 vs 9):
- Chain level, all methods: 12.3% (59/480) vs 1.9% (19/986) with fibrils; 2.8% (12/433) vs 1.0% (10/977) without. Both reproduce exactly.
- The per-unit figures (7.9% falling to 4.1%, vs 1.5%) only reproduce if each unit is flagged by its first chain. That undercounts mixed units such as 8HHX. Flagging a unit if any chain has an error gives 8.4% (15/178) falling to 4.7% (8/171), vs 1.5% (5/330). The direction is unchanged.
- What remains after removing fibrils is weak:
  - all-method units, Fisher p = 0.042 (BH q = 0.07);
  - cryo-EM units, p = 0.11;
  - cryo-EM chains adjusted for year and resolution, OR 2.17, p = 0.28;
  - leaving out one paper at a time, p ranges from 0.08 to 0.30.
- So "largely produce the λ L1 signal" holds.

The artifact claim is supported independently of SAbR. I computed Cα geometry for all 459 error chains in all 227 error entries:
- **Thickness** (spread along the smallest principal axis): fibril chains 1.0–3.0 Å (median 1.3) vs 4.8–7.8 Å for every non-fibril error chain.
- **Radius of gyration:** 24.5 Å vs 13.5 Å.
- **Long-range Cα contacts** (pairs at least 10 residues apart and within 6 Å, per residue): 0.00–0.04 vs a median of 0.80.
- Clean EM λ, κ and heavy-chain controls look like the non-fibril chains.
- **Run shape:** fibril runs are 63–107 residues long with a median |Δ| of 17. Non-fibril runs have a median length of 8 and median |Δ| of 1, and only 4 of 411 are 63 residues or longer.

Coverage: 18 cryo-EM AL fibrils at 3–5 Å exist in the PDB, and 15 are in the analysis set.
- SAbR processed 6HUD and 8CPE but they are missing from `entries.csv`, probably dropped by an upstream filter (only 77 residues modelled). Recomputed from their per-residue records, 69/77 and 77/77 residues fall in disagreement runs, so every fibril that reaches the detector is flagged (17/17).
- 6Z1I was never processed.
- This supports "detector artifact". It also means "15" is the count in the analysis set, not the total number of fibrils in scope.

Minor errors in the finding:
- "IGLV1-44 (x7)" is wrong under any counting: it is 6 entries, 4 units or 32 chains.
- The per-unit λ numbers are slightly off, as described above.
- "1/178 after" is really 0/178.

Effect not stated by the investigator: fibrils are 12/63 (19%) of EM error entries deposited in 2024 or later. Removing them lowers the EM error-entry rate from 4.9% to 4.2% in 2024 and from 5.9% to 4.2% in 2025, which strengthens the decline after 2021.

Framing: fibrils are better described as "not assessable by SAbR-vs-ANARCI" rather than proven error-free, but excluding them is correct either way. Two non-fibril chains have unusually few long-range contacts (8HHX I, 8G6I C). They are globular, not fibril-like.

Files are in `/home/user/SAbR/register_error_analysis/retest/deeper/verify/pdb_meta_amyloid_artifact_integrity/`:
- scripts `v01_recount.py` to `v07_bh.py`, each with a `.log`;
- `lambda_L1_variants.csv`, `fold_geometry.csv`, `fold_geometry_all_error_chains.csv.gz`, `verification_tests_bh.csv`.

## Recomputed values

Reproduced exactly:
- EM error entries that are fibrils: 15/193.
- Clean entries that are fibrils: 0/3,782.
- First-occurrence runs: 77/488 (15.8%).
- Error units: 12/218 across all methods; 12/183 within cryo-EM.
- λ error units: 12/26 across all methods; 12/24 within cryo-EM.
- Mode-A units: 6/66 across all methods; 6/59 within cryo-EM.
- Antibody-alone, with fibrils: 16/193 vs 14/2,574 (Fisher p = 1e-11; adjusted OR 20.7, p = 1.6e-7).
- Antibody-alone, without fibrils: 1/178 (truly 0/178) vs 14/2,574 (p = 1).
- λ L1 = 8 vs 9, chain level: 12.3% vs 1.9% with fibrils; 2.8% vs 1.0% without (p = 0.019).

Per-unit λ, corrected: 8.4% falling to 4.7%, vs 1.5%, instead of 7.9% falling to 4.1%.

The residual λ L1 = 8 effect after exclusion is weak to null: cryo-EM units p = 0.11; adjusted cryo-EM chains OR 2.2, p = 0.28.

SAbR-free fold geometry separates fibrils completely:
- thickness: fibrils at most 2.96 Å vs at least 4.77 Å for all 382 non-fibril error chains;
- radius of gyration: 24.5 vs 13.5 Å;
- long-range contacts per residue: 0.02 vs 0.80.

All 17 AL fibrils that SAbR processed are flagged, including 6HUD and 8CPE, which are missing from the analysis set.
