"""Compare pan-Ig numbering policies against IMGT and the letter scheme with pairwise TM-align."""
import json, pickle, random, sys
from collections import Counter, defaultdict
from multiprocessing import Pool
import numpy as np
import tmtools
sys.path.insert(0, "/tmp/pan")
import pan, ulib
from number import number, CDR_INTERIOR

G = {}


def anchor_sets(rows, u_occ=0.7, r_occ=0.5, r_spread=1.9):
    U, R = set(), set()
    for s, r in rows.items():
        n = int(s)
        if n in CDR_INTERIOR:
            continue
        o = r["occ"]
        if min(o["V"], o["C1"], o["C2/I"]) >= u_occ:
            U.add(n)
        if max(o["V"], o["C1"], o["C2/I"]) >= r_occ and r["spread"] <= r_spread:
            R.add(n)
    return U, R | U


def labelings(d, matches, U, R):
    n = len(d["target"])
    m = {i: int(s) for i, (s, dist) in matches.items()}
    mU = {i: s for i, s in m.items() if s in U}
    mR = {i: s for i, s in m.items() if s in R}
    out = {"IMGT": d["imgt"], "letter-scheme": d["target"],
           "O2 universal": number(n, mU, U), "O3 rich, IMGT-compatible": number(n, mR, U),
           "O4 rich, strict": number(n, mR, R)}
    out = {k: [l if l is not None else f"?{d['id']}:{i}" for i, l in enumerate(v)] for k, v in out.items()}
    claims = {"O2 universal": set(mU), "O3 rich, IMGT-compatible": set(mR), "O4 rich, strict": set(mR),
              "IMGT": set(range(n)), "letter-scheme": set(d["gt_sites"])}
    return out, claims


def _pair(args):
    ia, ib = args
    a, b = G["data"][ia], G["data"][ib]
    r = tmtools.tm_align(a["ca"], b["ca"], a["seq"], b["seq"])
    if max(r.tm_norm_chain1, r.tm_norm_chain2) < 0.5:
        return None
    Xa = a["ca"] @ r.u.T + r.t
    eq, i, j = {}, 0, 0
    for x, y in zip(r.seqxA, r.seqyA):
        if x != "-" and y != "-" and np.linalg.norm(Xa[i] - b["ca"][j]) <= 3.0:
            eq[i] = j
        i += x != "-"; j += y != "-"
    res = {}
    for k in a["lab"]:
        la, lb = a["lab"][k], b["lab"][k]
        pos = {l: q for q, l in enumerate(lb)}
        same = [(i, pos[l]) for i, l in enumerate(la) if l in pos]
        close = [np.linalg.norm(Xa[i] - b["ca"][j]) <= 3.0 for i, j in same]
        claimed = [(i, j) for i, j in same if i in a["claims"][k] and j in b["claims"][k]]
        cclose = [np.linalg.norm(Xa[i] - b["ca"][j]) <= 3.0 for i, j in claimed]
        res[k] = (len(eq), sum(la[i] == lb[j] for i, j in eq.items()), len(same), sum(close), len(claimed), sum(cclose))
    return a["type"] == b["type"], res


def run(doms, cache, per_dom, U, R, n_partners=10, seed=0):
    data = []
    for d in doms:
        if d["id"] not in per_dom:
            continue
        ca, _ = pan.geometry(d, cache)
        lab, claims = labelings(d, per_dom[d["id"]], U, R)
        c = cache[(d["pdb"], d["chain"])]; a0, b0 = d["rows"]
        data.append(dict(ca=ca, seq="".join(c["seq"][a0:b0]), lab=lab, claims=claims, type=pan.type_of(d["cls"]), cls=d["cls"]))
    G["data"] = data
    rnd = random.Random(seed)
    jobs = [(i, j) for i in range(len(data)) for j in rnd.sample(range(len(data)), n_partners) if j != i]
    with Pool(4) as p:
        res = [x for x in p.map(_pair, jobs, chunksize=20) if x]
    agg = defaultdict(lambda: np.zeros(6))
    for same_type, r in res:
        for k, v in r.items():
            agg[(k, "all")] += v; agg[(k, "within-type" if same_type else "cross-type")] += v
    ins = {k: np.mean([("." in l and not l.startswith("?")) for d in data for l in d["lab"][k]]) for k in data[0]["lab"]}
    unnum = {k: sum(any(l.startswith("?") for l in d["lab"][k]) for d in data) for k in data[0]["lab"]}
    print("domains with unnumbered residues:", unnum)
    return agg, ins, len(res)


if __name__ == "__main__":
    z = pickle.load(open("/tmp/pan/a12.pkl", "rb")); rows = pickle.load(open("/tmp/pan/rows.pkl", "rb"))
    doms, cache, _ = ulib.load_dataset()
    U, R = anchor_sets(rows)
    print(f"universal anchors {len(U)}: {sorted(U)}")
    print(f"rich anchors {len(R)}")
    agg, ins, npairs = run(doms, cache, z["per_dom"], U, R)
    print(f"pairs with TM>=0.5: {npairs}")
    for k in ["IMGT", "letter-scheme", "O2 universal", "O3 rich, IMGT-compatible", "O4 rich, strict"]:
        for scope in ("all", "within-type", "cross-type"):
            v = agg[(k, scope)]
            print(f"  {k:26s} {scope:12s} precision {v[3] / v[2]:.3f} recall {v[1] / v[0]:.3f} claimed-precision {v[5] / max(v[4], 1):.3f}"
                  + (f"  insertions {ins[k]:.3f}" if scope == "all" else ""))
    pickle.dump(dict(U=sorted(U), R=sorted(R), agg={f"{k}|{s}": v.tolist() for (k, s), v in agg.items()}, ins=ins),
                open("/tmp/pan/eval1.pkl", "wb"))
