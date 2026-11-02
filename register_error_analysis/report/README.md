# Standalone report: register errors in deposited antibody structures

`index.html` is a self-contained report for readers new to the topic. It covers the background,
the methods, all eighteen hypotheses tested (with verdicts), the mechanisms behind the errors, a
diagnosis and recommendations. It has 29 figures in `img/`. Open it in a browser; images load from
`img/` by relative path.

## Figures

* **Charts and register ladders** (`img/f_*.png`) are drawn by `scripts/figs.py` from the tables in
  `../`, `../followup/tables/`, `../sabr_anarci_scan/tables/`, `../retest/tables/` and
  `../retest/unexplained/tables/`.
  * Run `python figs.py` to draw all of them, or `python figs.py f_folds f_w118` to draw selected ones.
  * The register ladders (`scripts/ladder.py`) superpose a 3–5 Å chain on a high-resolution
    reference using framework Cα outside the window. For each reference position they report the
    nearest residue within 3 Å and that residue's own sequence position.
* **Structure images** (`img/s_*.png`) are 3Dmol.js 2.4.0 renders, captured with Playwright
  (`scripts/shot.js`) through the helpers in `scripts/render.py`.
  * Per-chain coordinates and SAbR/ANARCI numbering come from `scripts/rep_struct_prep.py`.
  * `scripts/render_cdr3fr4.py` reproduces Figure 26.
  * The context and anatomy renders (7CWU, 8ITF, 7YV1, 7MXL) were made interactively with the same
    helpers.
* **Paths.** The scripts point at a session scratch directory set at the top of `render.py` and
  `figs.py`. Edit `SP` and `R` to rerun them elsewhere.

## Correction made while preparing the report

The Fab 1B2 case in `../template_propagation/README.md` had listed 6MLK and 7KRA/7KTX as 1B2. They
are Fabs 3A6 and DH4, which share 1B2's light-chain FR1 sequence. The template family was defined by
that sequence window. The README and this report now say so. The conclusion is unchanged: every 1B2
entry after 6C9U carries 6C9U's register, and the two other antibodies, built on the same sequence,
use the register AlphaFold2 predicts.
