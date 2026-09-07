README badge maintenance
=======================

The coverage and Docker badges use Shields endpoint JSON files on the `badges`
branch. Workflows create this branch automatically using `GITHUB_TOKEN`; no
additional service account or secret is required. The endpoints become available
after these changes reach `main` and the publishing jobs first run successfully.

Coverage comes from the Python 3.12 test job on `main`, excludes vendored ANARCI
according to `pyproject.toml`, and is published only after all Python checks pass.
It represents the last successful checked revision, not a fixed threshold.

Docker freshness compares the version and source commit labels of
`ghcr.io/delalamo/sabr:latest` with the highest `v*` tag using Git's version sort,
matching the Docker build workflow's default tag selection. Every platform must
match. New tags, published releases, completed Docker builds (including failures),
manual runs, and a six-hour schedule refresh it. Missing metadata or registry
inspection failures show `unknown`. The badge measures the published image's
freshness; the linked build workflow reports build and smoke-test success.

Both publishing jobs share one concurrency group and update only their own JSON
file. Publication uses a temporary Git index, preserving the checkout and the
other badge. Unchanged badge values do not produce commits. Shields and GitHub
cache badge responses, so changes can take a few minutes to appear.

Run the badge logic and publication tests with:

```bash
python -m unittest discover -s .github/scripts -p 'test_*.py'
```
