"""Pan-Ig numbering with global segment apices (context-free label order)."""
import re

CDR_INTERIOR = set(range(28, 38)) | set(range(57, 65)) | set(range(108, 116))


class Scheme:
    """anchors: structural anchor positions (ints in 1..max_pos). Every maximal run of non-anchor
    integers is a segment with one fixed apex pair (IMGT CDR convention: left = after, right = before)."""

    def __init__(self, anchors, max_pos=128):
        self.anchors = sorted(set(anchors))
        self.max_pos = max_pos
        aset = set(self.anchors)
        self.segments, run = [], []
        for p in range(1, max_pos + 1):
            if p in aset:
                if run: self.segments.append(run)
                run = []
            else:
                run.append(p)
        if run: self.segments.append(run)
        self.apex = {}      # segment index -> (left, right) or (left, None)
        self.orient = {}    # number -> +1 (insertions after) / -1 (insertions before)
        for k, seg in enumerate(self.segments):
            m = len(seg)
            if m >= 2:
                c = (m + 1) // 2
                L, R = seg[c - 1], seg[c]
                if L == 1:          # number 1 is reserved for N-terminal extension (1.k before 1)
                    L, R = seg[c], seg[c + 1] if c + 1 < m else None
                self.apex[k] = (L, R)
            elif m == 1 and seg[0] != 1:
                self.apex[k] = (seg[0], None)
            for p in seg:
                ap = self.apex.get(k)
                self.orient[p] = -1 if (ap and ap[1] is not None and p >= ap[1]) else 1
        for p in self.anchors:
            self.orient[p] = 1
        self.orient[1] = -1
        self.seg_of = {p: k for k, seg in enumerate(self.segments) for p in seg}

    def key(self, label):
        m = re.fullmatch(r"(\d+)(?:\.(\d+))?", label)
        n, k = int(m.group(1)), int(m.group(2) or 0)
        return (n, self.orient.get(n, 1) * k)

    def number(self, n_res, site_of):
        idx = sorted(site_of)
        if not idx:
            return [None] * n_res
        aset = set(self.anchors)
        lab = {i: str(site_of[i]) for i in idx}
        first, last = idx[0], idx[-1]
        free = [p for p in range(site_of[first] - 1, 0, -1) if p not in aset]
        for q, i in enumerate(range(first - 1, -1, -1)):
            lab[i] = str(free[q]) if q < len(free) else f"1.{q - len(free) + 1}"
        free = [p for p in range(site_of[last] + 1, self.max_pos + 1) if p not in aset]
        nxt = self.max_pos
        for q, i in enumerate(range(last + 1, n_res)):
            if q < len(free):
                lab[i] = str(free[q])
            else:
                nxt += 1; lab[i] = str(nxt)
        for a, b in zip(idx, idx[1:]):
            res = list(range(a + 1, b))
            if res:
                lab.update(self._fill(res, site_of[a], site_of[b]))
        return [lab[i] for i in range(n_res)]

    def _fill(self, res, A, B):
        F = [p for p in range(A + 1, B) if p not in set(self.anchors)]
        k, m = len(res), len(F)
        out = {}
        if k <= m:                      # alternate from both ends; gaps at the centre
            front = (k + 1) // 2
            for q in range(front):
                out[res[q]] = str(F[q])
            for q in range(k - front):
                out[res[k - 1 - q]] = str(F[m - 1 - q])
            return out
        if m == 0:                      # no free number: insertions after the left anchor
            return {i: f"{A}.{q + 1}" for q, i in enumerate(res)}
        # choose the global apex nearest the span centre whose numbers lie inside F
        cands = [ap for s, ap in self.apex.items() if ap[0] in F and (ap[1] is None or ap[1] in F)]
        centre = (F[0] + F[-1]) / 2
        L, R = min(cands, key=lambda ap: abs((ap[0] + (ap[1] or ap[0])) / 2 - centre))
        left = [p for p in F if p <= L]
        right = [p for p in F if R is not None and p >= R]
        e = k - len(left) - len(right)
        n_right = (e + 1) // 2 if R is not None else 0
        n_left = e - n_right
        seq = [str(p) for p in left] + [f"{L}.{q + 1}" for q in range(n_left)] + \
              [f"{R}.{q}" for q in range(n_right, 0, -1)] + [str(p) for p in right]
        return dict(zip(res, seq))
