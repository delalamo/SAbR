# Verification of `enzyme_target` (antigen_competition lens): data integrity and statistics

**Verdict: weakened.** The core association reproduces exactly and is robust, but the claim as worded is
broader and stronger than the data support. It should be **moderate**, not **strong**, and narrowed to
"non-transporter, non-receptor enzymes, targeted mostly by tool antibodies, in cryo-EM".

## What reproduces (independent rebuild from `entries.csv` + the investigator's antigen table)

* **Data integrity checks passed.**
  * The joins are clean: the antigen table has one row per PDB entry, and every focal entry is present.
  * Entry error flags match the chain-level flags exactly.
  * The case/control set matches the investigator's: 3,994 entries, 2,752 EM with 178 errors, and the 15 amyloid
    entries excluded.
* **Host enzymes, entry level:** 15/81 vs 163/2671. The year + resolution logit gives OR 3.83 (1.84–8.00),
  p = 0.00035, exactly as claimed. Clustering by lab instead gives p = 0.0008.
* **Host enzymes, paper level:**
  * 12/45 vs 115/1266. Adjusting for year, resolution and log(entries) gives OR 4.1, p = 0.0004.
  * With lab experience added: OR 3.9, p = 0.0008.
  * A within-year permutation gives p = 0.0007.
* **Host enzymes, unit level** (each antibody's *own* contacted antigen):
  * 13/108 vs 158/4099, OR 3.7, p = 0.0002.
  * Non-nanobody units: OR 2.6 (p = 0.017). Nanobody units: OR 6.6 (p = 0.0035).
* **Period:**
  * 2020–21: OR 4.1. 2022+: OR 3.7. ≤2019 has only 2 entries.
  * Mantel–Haenszel homogeneity p = 0.74.
  * Without SARS-CoV-2: OR 3.7.
* **Leave one out:**
  * Leave-one-paper-out OR 3.1–3.6 (max p 0.003).
  * Leave-one-lab-out (normalised names) OR 3.1–3.6.
  * Leave-one-paper-out at unit level: OR 3.2–3.5.
* **Combined with "viral other":** 23/97 entries, OR 5.1. Papers: 15/51, with error papers from 12 labs.
* **BH correction:** all core tests have q < 0.002 across the 71 verification tests (`all_verify_tests_bh.csv`).

## What weakens it

1. **The result depends on how "enzyme" is defined.** "Enzyme" is the class left over after the
   GPCR, channel/transporter, toxin, cytokine and receptor rules have been applied.
   * Defining it simply as "the target has a UniProt EC number" (non-viral) gives **no effect**: 14/168 vs
     164/2584, OR 1.42, p = 0.35.
   * That is because 97 EC-bearing targets classed elsewhere have only 2 errors (2.1%). These include ABC
     transporters, P-type ATPases, NOX2, erbB2/ROS RTKs, ADAM17 and TMPRSS2.
2. **The labs cannot be separated from the targets.** Most enzyme error papers come from labs with only one cryo-EM
   antibody paper.
   * Within labs that published both kinds of paper, enzyme papers err 5/22 and other papers 7/58. The
     conditional logit gives OR 1.25 (0.26–5.9), p = 0.78. Only 7 labs had both clean and error papers, so this
     test has little power.
   * The lab-experience adjustment does not reduce the OR, which argues against a pure lab effect, but it does not
     rule one out.
3. **The paper-level result is fragile.**
   * Fragility index = 4: re-labelling 4 of the 12 host-enzyme error papers as clean brings Fisher to p ≥ 0.05.
   * Dropping the 5 error papers that contribute the most error entries gives entry-level p = 0.30 and
     paper-level p = 0.09.
4. **"Viral enzymes" (the investigator's "viral other" class):**
   * Errors come from only 2 labs.
   * **A copy leak:** Zimmer's Nanobody 881 has the identical run (IMGT 62–67, +1) in 7sp6/7sp7 (2021) and
     8snc/8snd (2023). The 2023 SEQRES differs by 1–2 residues, so the exact-sequence copy filter missed it. The
     2023 errors are same-lab reuse, not first occurrences.
     * After correction, viral other is 6/14 entries and 3/7 papers.
     * The combined class is 14/51 papers, OR 3.9, p = 0.0002.
     * A full-dataset scan found only one other leak (9pv5), so this is not widespread.
   * 3 of the 16 "viral other" entries are not viral enzymes: 24xy (P2Y13), 7rfp (TNFR18) and 9nz3. All 3 are clean.
   * The 2 Grimes 2021 error nanobodies (7nj5, 7nk2) have **no antigen contact** in the assembly-1 interfaces.
     Their "target" is only the largest entity in the entry.
5. **Sub-claims that do not hold:**
   * **"CDR2/FR3 (B) errors are over-represented":** in host enzymes the B mode accounts for 2/15 entries vs 23/155
     elsewhere (OR 0.88, p = 1). The excess comes entirely from Zimmer Nb881 (4 entries, 2 of them leaked copies)
     and from Grimes.
   * **"X-ray: 0 of 50":** this is uninformative. At the X-ray base rate, 1.4 errors are expected, and
     P(0) = 0.25. With viral other included, X-ray is 3/61 vs 31/1181 (OR 1.9, p = 0.23).
   * **"Mostly tool antibodies":** 47% of the host-enzyme units are "Auxiliary" by the text rule. Tool status is not
     the explanation:
     * auxiliary antibodies against transporters/GPCRs have 14/379 errors (3.7%);
     * auxiliary antibodies against enzymes have 9/51 (17.6%), OR 5.6, p = 0.0005.
     * Within enzymes, errors concentrate in auxiliary antibodies (9/51) rather than antibodies of interest (1/29),
       p = 0.085.
6. **Possible mechanism:** host-enzyme errors are more often at a chain end or next to an unbuilt loop (origin 3):
   11/15 vs 65/155, p = 0.028. 8/15 are A_FR1Nterm. This fits the known local-disorder mechanism. The
   chain-mean `lq_z` covariate the investigator adjusted for would not capture it.
7. **How entries are counted:** within-paper duplicates inflate the entry counts. Melcher contributes 3 entries
   with the same Fab light-chain error, and Southworth, Versees, Grimes and Zimmer 2 each. The paper and unit
   levels are the honest ones.

## No difference / null in this verification

* **EC-number definition of "enzyme":** OR 1.4, p = 0.35.
* **Within-lab comparison:** OR 1.25, p = 0.78.
* **Lab-level collapse** (one paper per lab and class): OR 2.0, p = 0.08.
* **Combined class in X-ray:** p = 0.23.
* **Host enzymes ≤2019:** 0/2.
* **Error modes:** the B mode in host enzymes (p = 1) and the A mode (OR 2.4, p = 0.15).

## Files

* **Scripts:** `v01_recompute.py`, `v02_robustness.py`, `v03_composition.py` and `v04_bh.py`. Their logs are the
  matching `v0*.log` files.
* **Tables:**
  * `all_verify_tests_bh.csv`;
  * `near_duplicate_copy_leaks.csv`;
  * `em_host_enzyme_entries.csv`;
  * `em_papers.csv`;
  * `units_em_enzyme.csv.gz`.
