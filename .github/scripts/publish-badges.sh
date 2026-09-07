#!/usr/bin/env bash
# Both publishing jobs share a concurrency group to preserve each other's files.
set -euo pipefail

BADGE_OUTPUT="${1:?Expected a directory of badge JSON files}"
BADGE_INDEX="$(mktemp)"
rm "$BADGE_INDEX"
trap 'rm -f "$BADGE_INDEX"' EXIT
export GIT_INDEX_FILE="$BADGE_INDEX"
export GIT_AUTHOR_NAME='github-actions[bot]'
export GIT_AUTHOR_EMAIL='41898282+github-actions[bot]@users.noreply.github.com'
export GIT_COMMITTER_NAME="$GIT_AUTHOR_NAME"
export GIT_COMMITTER_EMAIL="$GIT_AUTHOR_EMAIL"

BADGE_PARENT=""
BADGE_REF="$(git ls-remote --heads origin refs/heads/badges)"
if [ -n "$BADGE_REF" ]; then
  git fetch origin refs/heads/badges
  BADGE_PARENT="$(git rev-parse FETCH_HEAD)"
  git read-tree "$BADGE_PARENT"
else
  git read-tree --empty
fi

for BADGE_FILE in "$BADGE_OUTPUT"/*.json; do
  BADGE_BLOB="$(git hash-object -w "$BADGE_FILE")"
  git update-index --add --cacheinfo 100644 "$BADGE_BLOB" "$(basename "$BADGE_FILE")"
done
BADGE_TREE="$(git write-tree)"
if [ -n "$BADGE_PARENT" ]; then
  if [ "$BADGE_TREE" = "$(git rev-parse "$BADGE_PARENT^{tree}")" ]; then
    exit 0
  fi
  BADGE_COMMIT="$(git commit-tree "$BADGE_TREE" -p "$BADGE_PARENT" -m 'Update README badges')"
else
  BADGE_COMMIT="$(git commit-tree "$BADGE_TREE" -m 'Update README badges')"
fi
git push origin "$BADGE_COMMIT:refs/heads/badges"
