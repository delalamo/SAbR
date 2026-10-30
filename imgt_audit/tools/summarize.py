"""Per-IMGT-position structural consistency from pairs_<tag>.json."""
import json, os, sys
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(__file__))
from analyze import load, lab_key, CUTOFF

ROOT = os.path.join(os.path.dirname(__file__), "..")


def summarize(tag, doms):
    recs = json.load(open(os.path.join(ROOT, f"pairs_{tag}.json")))
    lab = {d["id"]: set(d["lab"]) for d in doms}
    cls = {d["id"]: d["label"] for d in doms}
    N, same, shift, unal = Counter(), Counter(), defaultdict(Counter), Counter()
    shift_cls = defaultdict(Counter)
    for r in recs:
        if r["pairs"] is None:
            continue
        for x, y, flip in ((r["a"], r["b"], False), (r["b"], r["a"], True)):
            m = {}
            for la, lb, dd in r["pairs"]:
                if flip:
                    la, lb = lb, la
                if dd <= CUTOFF:
                    m[la] = lb
            for p in lab[x] & lab[y]:
                N[p] += 1
                q = m.get(p)
                if q == p:
                    same[p] += 1
                elif q is None:
                    unal[p] += 1
                else:
                    shift[p][q] += 1
                    shift_cls[p][(cls[x], cls[y], q)] += 1
    rows = []
    for p in sorted(N, key=lab_key):
        s = sum(shift[p].values())
        rows.append(dict(pos=p, n=N[p], same=same[p] / N[p], shifted=s / N[p], unaligned=unal[p] / N[p],
                         top_shift=shift[p].most_common(3),
                         top_cls=[(f"{a}->{b}:{q}", c) for (a, b, q), c in shift_cls[p].most_common(4)]))
    json.dump(rows, open(os.path.join(ROOT, f"summary_{tag}.json"), "w"))
    return rows


if __name__ == "__main__":
    doms = load()
    for tag in sys.argv[1:]:
        rows = summarize(tag, doms)
        print(f"== {tag}")
        for r in rows:
            flag = "  <<" if r["shifted"] >= 0.10 else ""
            if r["n"] >= 20:
                print(f'{r["pos"]:>7} n={r["n"]:5d} same={r["same"]:.2f} shift={r["shifted"]:.2f} unal={r["unaligned"]:.2f} {r["top_shift"]} {r["top_cls"][:3]}{flag}')
