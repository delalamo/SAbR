#!/usr/bin/env bash
# Test the installed package in disposable containers; never modify the image.
set -euo pipefail

image="$1"
source_dir="$(cd "$2" && pwd)"
export EXPECTED_VERSION="$3"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

docker run --rm --entrypoint python --env EXPECTED_VERSION "$image" \
  -I -c 'import importlib.metadata as m, os, sabr; assert m.version("sabr-kit") == os.environ["EXPECTED_VERSION"]'

# Run the real CLI in the pristine runtime, before installing any test tools.
docker run --rm --entrypoint python \
  --mount "type=bind,source=$source_dir/tests/data,target=/data,readonly" \
  --mount "type=bind,source=$script_dir,target=/checks,readonly" \
  "$image" -I /checks/smoke_test.py /data

docker run --rm --entrypoint sh --workdir /tmp \
  --mount "type=bind,source=$source_dir,target=/source,readonly" \
  "$image" -ec '
    python -m pip install -c /source/constraints.txt pytest pytest-cov
    python -m pip check
    python -I -m pytest /source/tests -c /source/pyproject.toml \
      --cov=sabr --cov-config=/source/pyproject.toml \
      -o cache_dir=/tmp/pytest-cache
  '
