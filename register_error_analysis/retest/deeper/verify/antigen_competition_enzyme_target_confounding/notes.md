# Verification: "enzyme_target" (antigen_competition lens), checked for confounding

**Verdict: weakened.** The association is real in the lens's data and no confounder I tried removes it. But
three things go beyond what the data support:
* the label "antibodies against enzymes";
* the "strong" rating;
* the merge with viral enzymes.

## Reproduced exactly (cryo-EM entries; logit on year and resolution; SEs clustered by paper)

| Comparison | Enzyme | Other | OR | p |
|---|---|---|---|---|
| Host enzyme vs other | 15/81 (18.5%) | 163/2671 (6.1%) | 3.83 (1.84–8.00) | 3.5e-4 |
| Host + viral enzyme | 23/97 (23.7%) | 155/2655 (5.8%) | 5.1 | |
| Paper level | 15/51 (29%) | 112/1260 (8.9%) | 4.5 (2.3–8.7) | |

## Survives (`v02`, `v03`; one BH correction over 61 tests in `all_tests_bh.csv`)

**Choice of control:**
* antigen-bound entries only: OR 4.65;
* non-SARS-CoV-2 only: OR 5.0;
* non-SARS-CoV-2, bound, capsids removed: OR 5.5;
* host enzymes vs other non-viral protein targets: OR 4.6.

**Period:**
* 2020–21: 9/26 vs 52/592, OR 6.8;
* 2022+: 12/67 vs 92/1886, OR 4.4;
* Mantel–Haenszel over period × resolution bin: OR 5.2.

**Clustering:**
* standard errors clustered by lab: OR 5.1;
* paper level, clustered by lab: OR 4.5;
* lab cluster bootstrap: median OR 4.6 (2.3–8.6);
* one row per lab × class: OR 3.4 (p = 0.001);
* papers that reuse the same antibody in the same lab merged (Zimmer Nb881, Naismith megabody): OR 3.8.

**Known factors (entry level, all together):**
* the factors are unbuilt V residues, local map fit (lq_z), nanobody, megabody, atypical CDR, germline
  identity, mouse origin, starting model, fitting protocol, lab experience, SARS-CoV-2 and capsid;
* OR 6.5 (3.4–12.3).

**Chain type (antibody-in-paper units):**
* within VHH: 11/36 vs 27/531, OR 6.8;
* within Fab heavy chains: 6/48 vs 72/1730, OR 3.8.

**Known mechanisms excluded:**
* rigid-body-only fits removed: OR 4.0;
* rigid-body and experimental starting models removed: OR 3.7;
* entries whose only errors are CDR3/FR4 junction slips removed: OR 5.1.

**Outcome definition:** with any disagreement run as the outcome (copied runs included), OR 6.2.

## Weakening points

1. **It is not "enzymes".** It is the lens's residual "enzyme" class, which is applied after the GPCR,
   channel/transporter, toxin and receptor rules (`v04`).
   * Biochemical definition (any target with an EC number, non-viral, non-GPCR): 17/160 vs 161/2592, OR 1.87,
     p = 0.07. At paper level, OR 1.88, p = 0.046.
   * EC enzymes that the rules place elsewhere have no excess: **2/79 (2.5%)**, 2/50 papers. These include ABC
     transporters, P-type ATPases, receptor tyrosine kinases, ADAM17, NOX2 and ceramide synthase, and many
     of them use custom tool Fabs or nanobodies.
2. **The whole excess is in chains with unbuilt V-region residues** (`v03`).

   | Chain | Enzyme | Other | OR |
   |---|---|---|---|
   | Fully built | 4/242 (1.7%) | 203/8662 (2.3%) | 0.82 |
   | With a gap or N-terminal truncation | 21/43 (49%) | 94/1355 (6.9%) | 8.2 |

   * The interaction p is 0.004.
   * This is **not confounding.** Clean enzyme chains are *less* often gapped than other clean chains (8.5% vs
     13%).
   * Enzyme errors are almost all of the gap-adjacent type: at run level, 16/25 vs 107/326 sit next to a gap
     or chain end (p = 0.004). So these antibodies are not generally error-prone.
   * Because a register shift can itself force unbuilt residues, gap status is partly a mediator or part of the
     error definition. Use it to stratify, not to adjust.
3. **Few independent units.** The excess comes from 15 error papers, 12 labs and about 15 distinct V regions.
   * Removing the top 2 / 3 / 4 / 6 enzyme-error labs gives OR 3.5 / 2.9 / 2.5 (p = 0.03) / 1.8 (p = 0.19).
   * Among "antibody of interest" chains it is not significant: 10/122 vs 243/8071, OR 2.3, p = 0.17. The
     excess is in auxiliary or tool chains: 11/96 vs 43/1687, OR 6.4.
   * Light chains: OR 2.4, p = 0.25.
4. **The viral-enzyme merge is post hoc.** "Viral other" is 8/16 entries from 2 labs.
   * Zimmer's 4 error entries are one nanobody (Nb881) in two papers, with the same run (IMGT 62–67, +1).
   * The 2023 SEQRES has a 1-residue deletion at the error site (YADSVKG → YDSVKG). That is why the copy
     detector did not link 8snc/8snd to 7sp6/7sp7, although it did link the later 9yq4.
   * The class also holds non-enzymes: P2Y13 (24xy) and TNFRSF18 (7rfp).
5. **"Mode B over-represented" is not significant** per distinct erroneous V region: 5/19 vs 20/147, p = 0.17.
   Mode A shows no difference either: 7/19 vs 45/147, p = 0.60.

## No difference (enzyme vs other cryo-EM entries)

* Resolution.
* Local map fit (lq_z: 0.16 vs 0.08).
* N-terminal truncation (15.5% vs 16.3%).
* Mouse origin.
* Atypical CDR flags.
* Megabody (4.1% vs 1.8%, p = 0.10).
* EC enzymes in other classes.
* Light chains.
* Antibody-of-interest chains.
* Error-mode mix per distinct V region.
* X-ray: 0/50, underpowered.

## Files

* `v01_chain_gaps.py`: per-chain unbuilt-residue features, written to the cache `chain_gaps.csv.gz`.
* `v02_reanalysis.py` (log): reproduction, choice of control, period, paper and lab level, lab bootstrap,
  covariates, unit and chain kind, EC definition, leave-labs-out.
* `v03_definition_role_gap.py` (log): outcome definition, role, gap stratification, rigid-body fitting, modes.
* `v04_enzyme_definition.py` (log): broad EC-based enzyme definition.
* `v05_combine_bh.py`: the single BH correction, written to `all_tests_bh.csv`.
* Tables: `tests_*.csv` and `em_entries_verify.csv.gz`.
