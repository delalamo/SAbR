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
