"""Modal structural partner of each IMGT label, per (class X, class Y) pair."""
import json, os, sys
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(__file__))
from analyze import load, lab_key, CUTOFF

ROOT = os.path.join(os.path.dirname(__file__), "..")


def classpair(tag, doms, min_n=8, min_frac=0.5, classes=None):
    recs = json.load(open(os.path.join(ROOT, f"pairs_{tag}.json")))
    lab = {d["id"]: set(d["lab"]) for d in doms}
    cls = {d["id"]: d["cls"] for d in doms}
    stat = defaultdict(lambda: defaultdict(Counter))   # (X,Y) -> p -> Counter(partner)
    for r in recs:
        if r["pairs"] is None:
            continue
        for x, y, flip in ((r["a"], r["b"], False), (r["b"], r["a"], True)):
            if x not in cls or y not in cls:
                continue
            m = {}
            for la, lb, dd in r["pairs"]:
                if flip: la, lb = lb, la
                if dd <= CUTOFF: m[la] = lb
            for p in lab[x]:
                stat[(cls[x], cls[y])][p][m.get(p, "-")] += 1
    out = []
    for (X, Y), d in sorted(stat.items()):
        if classes and (X not in classes or Y not in classes):
            continue
        for p in sorted(d, key=lab_key):
            c = d[p]; n = sum(c.values())
            q, k = c.most_common(1)[0]
            if q == "-" and len(c) > 1:
                q, k = [t for t in c.most_common() if t[0] != "-"][0]
            if n >= min_n and q != p and q != "-" and k / n >= min_frac:
                out.append((X, Y, p, q, k, n, c.get(p, 0)))
    return stat, out


if __name__ == "__main__":
    doms = load()
    tag = sys.argv[1]
    stat, out = classpair(tag, doms)
    for X, Y, p, q, k, n, same in out:
        print(f"{X:>14} -> {Y:<14} {p:>6} ~ {q:<6} {k}/{n}  (same={same})")
