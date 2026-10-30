"""Agreement of IMGT labels with structural partners, per class pair and strand."""
import json, os, re, sys
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(__file__))
from analyze import load, lab_key, CUTOFF

ROOT = os.path.join(os.path.dirname(__file__), "..")
STRANDS = {
    "V": {"A": (3, 15), "B": (16, 26), "C": (39, 46), "C'": (47, 55), "C''": (66, 74), "D": (75, 84), "E": (85, 96), "F": (97, 104), "G": (118, 128)},
    "C": {"A": (1, 15), "B": (16, 26), "C": (39, 45), "D": (77, 84), "E": (85, 96), "F": (97, 104), "G": (118, 128)},
    "F": {"A": (1, 15), "B": (16, 26), "C": (39, 45), "D": (77, 84), "E": (85, 96), "F": (97, 104), "G": (118, 128)},
    "G": {"A": (1, 9), "B": (10, 18), "C": (19, 28), "D": (29, 38), "helix-1": (50, 72), "helix-2": (73, 92)},
}


def strand_of(t, lab):
    m = re.match(r"(\d+)$", lab)
    if not m:
        return None
    n = int(m.group(1))
    for s, (a, b) in STRANDS[t].items():
        if a <= n <= b:
            return s


def matrix(tag, t, doms, merge=lambda c: c):
    recs = json.load(open(os.path.join(ROOT, f"pairs_{tag}.json")))
    cls = {d["id"]: merge(d["cls"]) for d in doms}
    labs = {d["id"]: set(d["lab"]) for d in doms}
    agree = defaultdict(Counter); tot = defaultdict(Counter)
    for r in recs:
        if r["pairs"] is None or r["a"] not in cls or r["b"] not in cls:
            continue
        for x, y, flip in ((r["a"], r["b"], False), (r["b"], r["a"], True)):
            m = {}
            for la, lb, dd in r["pairs"]:
                if flip: la, lb = lb, la
                if dd <= CUTOFF: m[la] = lb
            for p in labs[x] & labs[y]:
                s = strand_of(t, p)
                if s is None: continue
                q = m.get(p)
                if q is None: continue           # only count residues with a structural partner
                key = (cls[x], cls[y])
                tot[key][s] += 1; agree[key][s] += (q == p)
    return agree, tot


if __name__ == "__main__":
    tag, t = sys.argv[1], sys.argv[2]
    merge = (lambda c: "C-BETA" if c.startswith("C-BETA") else c)
    doms = load()
    agree, tot = matrix(tag, t, doms, merge)
    classes = sorted({k[0] for k in tot})
    strands = list(STRANDS[t])
    print("pair".ljust(36) + " ".join(f"{s:>6}" for s in strands))
    for X in classes:
        for Y in classes:
            if (X, Y) not in tot or X > Y:
                continue
            row = []
            for s in strands:
                n = tot[(X, Y)][s]
                row.append(f"{agree[(X, Y)][s]/n:6.2f}" if n >= 10 else "     .")
            print(f"{X[:17]:>17}>{Y[:17]:<18}" + " ".join(row))
