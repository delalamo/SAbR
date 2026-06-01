# Development

Install development dependencies:

```bash
pip install -e .[test,docs]
pip install pre-commit ruff
pre-commit install
```

Run fast tests:

```bash
pytest
```

Run slow real-model tests:

```bash
JAX_PLATFORMS=cpu pytest -m slow
```

Build docs locally:

```bash
sphinx-build -W -b html docs docs/_build/html
```

Run formatting and linting:

```bash
ruff check .
ruff format --check .
prettier --check "**/*.{js,ts,css,md,yml,yaml,json}"
```

Benchmark scripts live in `scripts/`, including `scripts/imgt_benchmark.py` for
IMGT-numbered SAbDab comparisons.

## Design invariants

SAbR keeps the renumbering path deliberately direct:

```text
structure file or BioPython structure
  -> select chain and extract [N, CA, C, computed CB] coordinates
  -> compute query embeddings
  -> align query embeddings to labelled H/K/L reference embeddings
  -> convert the alignment matrix to ANARCI states
  -> apply numbering
  -> thread numbering back onto a structure
```

- Query coordinates are `[N, CA, C, computed CB]`; the packaged model was
  trained with that fourth channel.
- Reference embeddings must be split and labelled exactly `H`, `K`, and `L`.
- `chain_type=None` and CLI `--chain-type auto` mean SAbR tries all three
  labelled references and passes the best-scoring label to ANARCI.
- Custom gap penalties set gap-open to zero in IMGT CDR positions only; gap
  extension remains normal.
- PDB output cannot represent multi-character insertion codes; use mmCIF when
  long CDR insertions are expected.
