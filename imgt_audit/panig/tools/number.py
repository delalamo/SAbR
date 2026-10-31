"""IMGT-style deterministic numbering from matched anchors, with selectable honesty policies."""
import re

CDR_INTERIOR = set(range(28, 38)) | set(range(57, 65)) | set(range(108, 116))


def imgt_fill(res, free):
    """IMGT CDR rule generalised: fill free numbers alternately from both ends (front first);
    overflow at the centre pair, first insertion on the right; left insertions n.1, n.2 ... (after n),
    right insertions m.k ... m.1 (before m)."""
    k, m = len(res), len(free)
    lab = {}
    nf = min(k, m)
    front = (nf + 1) // 2
    back = nf - front
    for q in range(front):
        lab[res[q]] = str(free[q])
    for q in range(back):
        lab[res[k - 1 - q]] = str(free[m - 1 - q])
    extra = res[front:k - back]
    if extra:
        if m == 0:
            return None  # caller attaches to the left anchor
        left = free[front - 1] if front else None
        right = free[m - back] if back else None
        n_right = (len(extra) + 1) // 2 if right is not None else 0
        n_left = len(extra) - n_right
        if left is None:
            n_left, n_right = 0, len(extra)
        if right is None:
            n_left, n_right = len(extra), 0
        for q in range(n_left):
            lab[extra[q]] = f"{left}.{q + 1}"
        for q in range(n_right):
            lab[extra[len(extra) - 1 - q]] = f"{right}.{q + 1}"
    return lab


def number(n, site_of, exclude, max_pos=128):
    """site_of: {residue index: anchor int}; exclude: numbers never used for fill."""
    idx = sorted(site_of)
    lab = {i: str(site_of[i]) for i in idx}
    if not idx:
        return [None] * n
    used = set(site_of.values()) | set(exclude)
    first, last = idx[0], idx[-1]
    # N-terminus: free integers counting down to 1, then 1.1, 1.2, ... (IMGT C-domain style)
    free = [p for p in range(site_of[first] - 1, 0, -1) if p not in used]
    for q, i in enumerate(range(first - 1, -1, -1)):
        lab[i] = str(free[q]) if q < len(free) else f"1.{q - len(free) + 1}"
    # C-terminus: free integers up to max_pos, then max_pos+1, ...
    free = [p for p in range(site_of[last] + 1, max_pos + 1) if p not in used]
    extra = max_pos
    for q, i in enumerate(range(last + 1, n)):
        if q < len(free):
            lab[i] = str(free[q])
        else:
            extra += 1; lab[i] = str(extra)
    for a, b in zip(idx, idx[1:]):
        res = list(range(a + 1, b))
        if not res:
            continue
        free = [p for p in range(site_of[a] + 1, site_of[b]) if p not in used]
        fill = imgt_fill(res, free)
        if fill is None:
            fill = {i: f"{site_of[a]}.{q + 1}" for q, i in enumerate(res)}
        lab.update(fill)
    return [lab[i] for i in range(n)]
