"""Fast scoring of numbering variants on cached pairwise superpositions."""
import pickle, sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, "/tmp/pan")
import pan, ulib
from number2 import Scheme, CDR_INTERIOR

_P = None


def pairs():
    global _P
    if _P is None:
        _P = pickle.load(open("/tmp/pan/pairs.pkl", "rb"))
    return _P


def anchor_set(rows, occ=0.5, spread=1.9, types=("V", "C1", "C2/I"), cdr_free=True, extra=(), drop=()):
    A = set()
    for s, r in rows.items():
        n = int(s)
        if cdr_free and n in CDR_INTERIOR:
            continue
        if max(r["occ"][t] for t in types) >= occ and r["spread"] <= spread:
            A.add(n)
    return (A | set(extra)) - set(drop)


def label_all(doms, per_dom, scheme):
    out, claims = {}, {}
    A = set(scheme.anchors)
    for d in doms:
        m = {i: int(s) for i, (s, dist) in per_dom[d["id"]].items() if int(s) in A}
        lab = scheme.number(len(d["target"]), m)
        out[d["id"]] = [l if l is not None else f"?{d['id']}:{i}" for i, l in enumerate(lab)]
        claims[d["id"]] = set(m)
    return out, claims


def score(doms, labels, claims=None, cut=3.0):
    P = pairs()
    ids = P["ids"]
    typ = {d["id"]: pan.type_of(d["cls"]) for d in doms}
    agg = defaultdict(lambda: np.zeros(6))
    for ia, ib, tm, D, eq in P["pairs"]:
        a, b = ids[ia], ids[ib]
        if a not in labels or b not in labels:
            continue
        la, lb = labels[a], labels[b]
        pos = {l: q for q, l in enumerate(lb)}
        same = [(i, pos[l]) for i, l in enumerate(la) if l in pos]
        close = sum(D[i, j] <= cut for i, j in same)
        rec = sum(la[i] == lb[j] for i, j in eq)
        if claims is not None:
            cl = [(i, j) for i, j in same if i in claims[a] and j in claims[b]]
            cc = sum(D[i, j] <= cut for i, j in cl)
        else:
            cl, cc = same, close
        v = np.array([len(eq), rec, len(same), close, len(cl), cc], float)
        scope = "within-type" if typ[a] == typ[b] else "cross-type"
        agg["all"] += v; agg[scope] += v; agg[f"{typ[a]}~{typ[b]}" if typ[a] <= typ[b] else f"{typ[b]}~{typ[a]}"] += v
    return {k: dict(precision=v[3] / v[2], recall=v[1] / v[0], claimed=v[5] / max(v[4], 1)) for k, v in agg.items()}


def insertion_rate(doms, labels):
    by = defaultdict(list)
    for d in doms:
        t = pan.type_of(d["cls"])
        for l in labels[d["id"]]:
            by[t].append("." in l); by["all"].append("." in l)
    return {k: float(np.mean(v)) for k, v in by.items()}
