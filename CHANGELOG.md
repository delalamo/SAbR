# Changelog

All notable changes to SAbR (Structure-based Antibody Renumbering) are documented in this file.

## [Unreleased]

## 2026-01-07

### Fixed

- Fix FR1 double-assignment bug and add chain-type-aware pattern (#158)

## 2026-01-06

### Fixed

- Fix aln2hmm offset that introduced off-by-one errors in cases with N-terminal insertions (#157)
- Fix PDB fetch failure handling in GitHub Action (#152)

### Changed

- Remove shark antibodies and scFvs from benchmark datasets (#156)
- Update SAbDab script gap checking logic (#155)
- Update gap definition to use constant value (#153)
- Refactor softalign code to be added directly to SAbR (#149)

## 2026-01-04

### Fixed

- Fix bug that truncated matches to zero even when they were 0.9999 (#146)

### Performance

- Replace BioPython file I/O with Gemmi for 12x faster performance (#144)

## 2026-01-02

### Added

- Add structural gap detection to skip deterministic CDR renumbering (#145)

## 2026-01-01

### Changed

- Enhance CDR anchor warning to show closest residue (#142)
- Change C-terminus IMGT position warning to debug level (#141)
- Update model weights to remove positive gap extend penalty/bonus (#138)
- Filter comparison to IMGT positions 1-128 only (#140)

## 2025-12-30

### Added

- Add browser-like headers to PDB fetch requests (#133)
- Add CLAUDE.md with pre-commit and type hint guidelines (#136)
- Add function to renumber BioPython objects (#123)
- Add SAbDab test set execution as a GitHub Action (#131)
- Implement analysis workflow on pre-2021 IMGT-renumbered entries from SAbDab (#130)

### Fixed

- Fix GitHub Actions for test set (#137)
- Fix RTD homepage to show README instead of **init**.py docstring (#128)
- Fix ReadTheDocs build by enabling submodule checkout (#127)
- Yet another ReadTheDocs homepage fix (#129)

### Changed

- Remove uv.lock from version control (#135)
- Optimize test suite for faster execution (#134)
- Rename /review and /ask commands to /review_claude and /ask_claude (#132)
- Compartmentalize JAX/Haiku into single jax_backend module (#124)

## 2025-12-29

### Added

- Add .readthedocs.yaml for pdoc documentation (#125)
- Add documentation (#122)

## 2025-12-27

### Fixed

- Fix CDR1 renumbering issues (#120)

### Changed

- Remove unused constants and related tests (#121)

## 2025-12-26

### Added

- Add CLI arguments for ANARCI species and chain type (#117)
- Add deterministic C-terminus numbering correction (#115)
- Add unified embeddings support (#114)
- Add N-terminal truncation integration test (#111)
- Allow /ask about PRs (#113)
- Add AI code review GitHub Action (#110)

### Fixed

- Fix code review model (#112)

### Changed

- Remove redundant tests and consolidate with parametrization (#119)
- Refactor CLI main function for better modularity (#118)
- Remove unnecessary comments across codebase (#116)

## 2025-12-24

### Added

- Add end-to-end integration test for N-terminal extension numbering (#109)

### Fixed

- Debug SAbR test cases (#108)

### Changed

- Implement soft CDR boundary detection (#106)

## 2025-12-23

### Fixed

- Fix off-by-one alignment bug (#104)
- Fix insertion code logic (#103)
- Fix CDR loop slice bounds in deterministic renumbering (#102)

## 2025-12-22

### Fixed

- Fix misleading error messages in SoftAlignOutput validation (#99)

## 2025-12-21

### Added

- Add status badges to README (#101)

### Changed

- Update README with corrected Docker instructions (#100)
