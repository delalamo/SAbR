"""Structure-consistent numbering of immunoglobulin-fold domains.

A single reference covers V, C1, C2/I-set and FN3 domains. Each reference row is
one structural anchor of the unified Ig frame, ordered along the chain; its
vector is trained so that its dot product with a SAbR residue embedding is the
log-odds of that residue occupying the anchor rather than a loop. Domains align
to the subset of anchors they possess, and everything between two matched
anchors is numbered by loop length alone.

Labels follow these rules:

* an integer is a structural site shared across domain types (e.g. 23, 41, 104);
* a letter suffix (``8A``) is a site present only in some types or classes;
* a decimal (``111.3``) is a loop insertion attached to the preceding label;
* loop residues take the integers between their flanking anchors, filled from
  both ends so gaps fall at the loop centre; letters are never loop slots;
* N-terminal residues count down from the first anchor (0 and negatives are
  allowed) and C-terminal residues count up from the last.

Labels sort as 8 < 8.1 < 8A < 8A.1 < 8B < 9, so numbering is always in chain
order. Additional domains in the same chain are offset by
``constants.DOMAIN_NUMBERING_STRIDE``.
"""

import functools
import logging
import re
from importlib.resources import files

import numpy as np

from sabr import constants
from sabr.alignment import _align_reference
from sabr.types import FloatArray

LOGGER = logging.getLogger(__name__)

REFERENCE_FILE = "unified_reference.npz"
# A segment is accepted as a domain when it matches at least
# MIN_DOMAIN_ANCHORS anchors and either its alignment score (summed anchor
# log-odds minus gap costs) reaches MIN_DOMAIN_SCORE or it matches
# CONFIDENT_DOMAIN_ANCHORS anchors. On held-out data this keeps ~98% of Ig
# domains and admits ~1% of chains from non-Ig folds absent from training.
MIN_DOMAIN_ANCHORS = 30
MIN_DOMAIN_SCORE = 400.0
CONFIDENT_DOMAIN_ANCHORS = 50
# Segments shorter than this are not searched for further domains.
MIN_SEGMENT_RESIDUES = 40
# Consecutive anchors further apart than this belong to different domains
# (longer than any loop of a single domain except rare ultralong CDR3s).
MAX_ANCHOR_SPACING = 45
_LABEL = re.compile(r"(-?\d+)([A-Z]*)(?:\.(\d+))?$")


@functools.cache
def load_reference() -> tuple[FloatArray, tuple[str, ...]]:
    """Return read-only [M, 64] anchor vectors and their M ordered labels."""
    path = files("sabr.assets") / REFERENCE_FILE
    with path.open("rb") as handle, np.load(handle, allow_pickle=False) as data:
        embeddings = np.asarray(data["embeddings"], dtype=np.float32)
        labels = tuple(str(label) for label in data["labels"])
    embeddings.flags.writeable = False
    return embeddings, labels


def label_key(label: str) -> tuple[int, tuple[int, str], int]:
    """Sort key: integer, then letter suffix (A < Z < AA), then decimal."""
    match = _LABEL.fullmatch(label)
    if match is None:
        raise ValueError(f"Invalid unified label {label!r}.")
    letters = match.group(2)
    return (
        int(match.group(1)),
        (len(letters), letters),
        int(match.group(3) or 0),
    )


def split_label(label: str) -> tuple[int, str]:
    """Map a label to a residue number and insertion code: 8A.1 -> (8, 'A1')."""
    match = _LABEL.fullmatch(label)
    if match is None:
        raise ValueError(f"Invalid unified label {label!r}.")
    return int(match.group(1)), match.group(2) + (match.group(3) or "")


def join_label(number: int, insertion: str) -> str:
    """Inverse of split_label: (8, 'A1') -> 8A.1, relative to the domain."""
    letters = insertion.rstrip("0123456789")
    decimal = insertion[len(letters) :]
    return f"{number}{letters}" + (f".{decimal}" if decimal else "")


def fill_loops(length: int, anchors: dict[int, str]) -> list[str]:
    """Number every row of a domain from its anchored rows.

    ``anchors`` maps zero-based rows to anchor labels and must increase along
    the rows. Rows between two anchors take the integers strictly between the
    anchors' integer parts, alternately from the N- and C-terminal ends (the
    N side takes the extra residue when the count is odd). Surplus rows become
    insertions on the last N-side number. Rows outside the first and last
    anchors count outward through integers.
    """
    if not anchors:
        raise ValueError("At least one anchor is required.")
    rows = sorted(anchors)
    labels = dict(anchors)
    used = set(anchors.values())
    first, last = rows[0], rows[-1]
    start = label_key(anchors[first])[0]
    for offset, row in enumerate(range(first - 1, -1, -1), start=1):
        labels[row] = str(start - offset)
    end = label_key(anchors[last])[0]
    for offset, row in enumerate(range(last + 1, length), start=1):
        labels[row] = str(end + offset)
    for left, right in zip(rows, rows[1:]):
        loop = list(range(left + 1, right))
        if not loop:
            continue
        low = label_key(anchors[left])[0] + 1
        high = label_key(anchors[right])[0]
        slots = [str(k) for k in range(low, high) if str(k) not in used]
        filled = min(len(loop), len(slots))
        n_left = (filled + 1) // 2
        n_right = filled - n_left
        for index in range(n_left):
            labels[loop[index]] = slots[index]
        for index in range(n_right):
            labels[loop[-1 - index]] = slots[-1 - index]
        extra = loop[n_left : len(loop) - n_right]
        if extra:
            base = slots[n_left - 1] if n_left else anchors[left]
            number, insertion = split_label(base)
            letters = insertion.rstrip("0123456789")
            decimal = int(insertion[len(letters) :] or 0)
            for index, row in enumerate(extra, start=1):
                labels[row] = f"{number}{letters}.{decimal + index}"
    return [labels[row] for row in range(length)]


def assign_anchors(
    embeddings: FloatArray, gap_indices: frozenset[int] = frozenset()
) -> tuple[dict[int, str], float]:
    """Align [Q, 64] embeddings to the unified reference.

    Returns the anchor label of every matched row and the alignment score.
    ``gap_indices`` are rows before structural breaks, as from extract_chain.
    """
    reference, labels = load_reference()
    if len(embeddings) < 2:
        return {}, float("-inf")
    soft, _, score = _align_reference(
        embeddings, reference, query_gap_indices=frozenset(gap_indices)
    )
    rows, columns = np.nonzero(soft > 0.5)
    return {int(r): labels[int(c)] for r, c in zip(rows, columns)}, score


def find_domains(
    embeddings: FloatArray, gap_indices: frozenset[int] = frozenset()
) -> list[dict[int, str]]:
    """Return per-domain anchor maps (chain rows) in chain order.

    The whole chain is aligned first. An alignment is cut wherever consecutive
    anchors are more than MAX_ANCHOR_SPACING rows apart, the run with most
    anchors is kept, and the unmatched segments on either side are searched
    again, so V-C chains, Fc fragments and multi-domain receptors yield one
    anchor map per Ig domain.
    """
    domains = []
    segments = [(0, len(embeddings))]
    while segments:
        start, stop = segments.pop()
        if stop - start < MIN_SEGMENT_RESIDUES:
            continue
        gaps = frozenset(
            i - start for i in gap_indices if start <= i < stop - 1
        )
        anchors, score = assign_anchors(embeddings[start:stop], gaps)
        runs, run = [], []
        for row in sorted(anchors):
            if run and row - run[-1] > MAX_ANCHOR_SPACING:
                runs.append(run)
                run = []
            run.append(row)
        if run:
            runs.append(run)
        rows = max(runs, key=len, default=[])
        if len(rows) < MIN_DOMAIN_ANCHORS or (
            score < MIN_DOMAIN_SCORE and len(rows) < CONFIDENT_DOMAIN_ANCHORS
        ):
            continue
        anchors = {row: anchors[row] for row in rows}
        domains.append({start + r: anchors[r] for r in rows})
        LOGGER.info(
            "Unified domain at rows %d-%d: %d anchors, score %.2f.",
            start + rows[0],
            start + rows[-1],
            len(anchors),
            score,
        )
        segments.append((start, start + rows[0]))
        segments.append((start + rows[-1] + 1, stop))
    return sorted(domains, key=min)


def number_embeddings(
    embeddings: FloatArray, gap_indices: frozenset[int] = frozenset()
) -> list[tuple[int, int, str]]:
    """Number every row of a chain: (row, residue number, insertion code).

    Linkers between domains are split: the first half continues the previous
    domain's C-terminal count and the second half precedes the next domain.
    """
    domains = find_domains(embeddings, gap_indices)
    if not domains:
        raise ValueError(
            "No immunoglobulin-fold domain matched the unified reference."
        )
    length = len(embeddings)
    bounds = [(min(anchors), max(anchors)) for anchors in domains]
    starts = [0]
    for (_, last), (first, _) in zip(bounds, bounds[1:]):
        starts.append(last + 1 + (first - last - 1) // 2)
    stops = starts[1:] + [length]
    records = []
    for index, (anchors, start, stop) in enumerate(zip(domains, starts, stops)):
        local = {row - start: label for row, label in anchors.items()}
        labels = fill_loops(stop - start, local)
        offset = index * constants.DOMAIN_NUMBERING_STRIDE
        for row, label in enumerate(labels, start=start):
            number, insertion = split_label(label)
            records.append((row, number + offset, insertion))
    return records
