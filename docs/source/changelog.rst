Changelog
=========

All notable changes to SAbR will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

Unreleased
----------

Added
~~~~~

- Sphinx documentation with API reference
- Documentation for all numbering schemes
- Contributing guide

Changed
~~~~~~~

- Improved module docstrings for better autodoc generation

[0.1.0] - Initial Release
-------------------------

Added
~~~~~

- Structure-based antibody renumbering using MPNN embeddings
- Support for IMGT, Chothia, Kabat, Martin, Aho, and Wolfguy numbering schemes
- Automatic chain type detection (heavy, kappa, lambda)
- PDB and mmCIF input/output support
- Extended insertion codes for very long CDR loops (mmCIF only)
- Deterministic corrections for CDR, DE loop, FR1, and C-terminus regions
- Docker container support
- Command-line interface via Click
