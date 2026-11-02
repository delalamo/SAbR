# First-occurrence register errors, section by section: what links the wrong models?

**Population.**
* **Scan:** the AlphaFold-independent 3–5 Å scan (15,865 single-domain V chains, to 2026).
* **Errors kept:** SAbR-vs-ANARCI runs that are *not* copied from an earlier structure of the same antibody, and not
  known detector artifacts. That leaves 488 runs, or 251 independent events, counted as one unit per antibody per
  paper.
* **Comparison:** each error mode is compared with every unit of the same chain type modelled correctly at that
  location, in cryo-EM (about 90% of the errors).
* **Features compared:**
  * local sequence;
  * how similar the sequence is to itself shifted by one;
  * germline and CDR lengths;
  * unbuilt residues and truncated termini;
  * map fit at the site, as a percentile within the domain;
  * resolution, starting model, fitting protocol, software and version, era, role and SARS-CoV-2.

Scripts: `scripts/deepdive.py`, `deepdive_compare.py`, `deepdive_summary.py`, `deepdive_overall.py`,
`software_versions.py`. Tables: `tables/`.

## The modes, and what sets each apart from correct models at the same spot

| Mode (units, all methods) | Distinguishing features, errors vs correct (EM) |
|---|---|
| **A. FR1 N-terminal strand, shift −1** (66) | N-terminus not built: first built residue IMGT 3 vs 1; start after residue 3 in 31% vs 4% (q = 10⁻¹⁰). Unbuilt residues nearby: 63% vs 24% (with the truncations). Non-canonical N-terminal residues: heavy Q1 24% vs 56%, E5 12% vs 2%; κ I2 60% vs 87% |
| **F. CDR1 boundaries** (15) | Mouse 50% vs 19%. Atypical CDR1 length 42% vs 15% |
| **B. CDR2→FR3** (32) | Atypical CDR2 length 24% vs 3% (q = 0.002). Nanobodies 28% vs 14%. Germline identity 83% vs 90%. Unusual CDR2 residues (S56, R57, S60, G64, an extra Cys55). Unbuilt residues nearby 20% vs 2% |
| **C. FR3 around IMGT 81** (31) | Mouse IGHV5 21% vs 2.6% (q = 0.006). Not Asp at IMGT 81: 30% vs 5%. Atypical CDR3 length 92% vs 63%. Unbuilt residues nearby 38% vs 2% |
| **D+. CDR3 built one residue short, FR4 +1** (40) | Unbuilt residues nearby 61% vs 28% (q = 10⁻⁶). Poor local map: domain z −0.41 vs +0.40; site Q-score 0.30 vs 0.45. Long CDR3s: 14 vs 12; IGHJ6 19% vs 7%. Fv-only models 55% vs 40%. SARS-CoV-2 39% vs 24% |
| **D−. CDR3 built one residue long, FR4 −1** (22) | Worst site fit: Q-score 0.23 vs 0.45 (q = 0.001). Nanobodies 40% vs 13%. Germline identity 76% vs 90%. Rigid-body-only fitting 30% vs 11%. Resolution 3.7 vs 3.4 Å |
| **E. FR4 bulge after W/F-G** (19) | Mouse IGKV8 17% vs 0.9% (q = 0.03). Mouse 44% vs 19%. Atypical CDR1 33% vs 15%. Fv-only 61% vs 40% |

## What links them across all sections

**Pooled over all modes** (189 EM units), the error site compared with the same location in correctly modelled
domains:

| | Errors | Correct |
|---|---|---|
| Unbuilt residues within 5 positions, or N-terminus not built | **48%** | 16% |
| Bottom-quartile map fit at the site | 32% | 25% |
| Either of the above | **66%** | 37% |
| Adjacent CDR of atypical length | 44% | 38% |

1. **The register is assigned across a stretch the modeller could not see.**
   * When the residues next to a strand are missing (an unbuilt loop, an unbuilt N-terminus) or the density there is
     poor, the sequence has to be placed on the strand by counting from the other side of the gap.
   * The count is off by one in a way that matches what the antibody's sequence would suggest (below).
   * This is the same failure in every section. It is the 7MXL mechanism (`../../template_propagation`).
2. **The antibody is unusual exactly at that spot.** Examples: an atypical CDR2 next to the slipped FR3, long CDR3s
   and nanobody CDR3s before the slipped FR4, mouse IGKV8 or IGHV5, unusual N-terminal residues. A template- or
   germline-based expectation of how many residues fill the gap is then wrong.
3. **What does not link them:**

   | Candidate | Result |
   |---|---|
   | Software or version | No program or version is over-represented. In 2018–21, entries recording Rosetta, Coot or Phenix have *fewer* errors (1.5–7.6% vs 9.6% base). Chimera-only fitting is 12.9% (n.s.). No Phenix or Coot version stands out |
   | Lab or paper | Errors are spread out; e.g. mode A covers 59 units from 50 papers and 42 labs, with at most 7 units from one lab |
   | Lineage | No shared germline lineage (`../lineage`) |
   | Copying | No copying from earlier models (by construction, and `../../template_propagation`) |
   | Sequence self-similarity | Not consistently higher at error sites: slightly higher in mode A, not elsewhere |

**What remains.** About a third of first-occurrence errors have neither a gap nor poor fit at the site. For those,
the only consistent marker is a location-specific atypical feature. These are mainly in modes B, C and F, where most
errors are not next to a gap.

**Caveats.**
* Software versions are rarely recorded.
* Map fit is a relative measure (percentile within the domain) and Q-scores at 3–4 Å are noisy.
* The "next to a gap" feature partly overlaps how the earlier origin classes were defined. The non-gap subset was
  therefore tested separately: CDR3 errors not next to a gap still have poor site fit, 70% vs 25% (p = 0.003).
