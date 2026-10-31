import pickle, sys
sys.path.insert(0, "/tmp/pan")
import numpy as np, pan, ulib
from collections import defaultdict, Counter
from score import anchor_set, label_all, pairs
from number2 import Scheme

def per_anchor(doms, lab, cl, typ):
    P = pairs(); ids = P["ids"]
    st = defaultdict(lambda: np.zeros(4))   # within n, within close, cross n, cross close
    for ia, ib, tm, D, eq in P["pairs"]:
        a, b = ids[ia], ids[ib]
        if a not in lab or b not in lab: continue
        cross = typ[a] != typ[b]
        pos = {l: q for q, l in enumerate(lab[b])}
        for i in cl[a]:
            l = lab[a][i]; j = pos.get(l)
            if j is None or j not in cl[b]: continue
            k = 2 if cross else 0
            st[int(l)][k] += 1; st[int(l)][k + 1] += D[i, j] <= 3.0
    return st

if __name__ == "__main__":
    z = pickle.load(open("a12.pkl", "rb")); rows = pickle.load(open("rows.pkl", "rb")); s2 = pickle.load(open("sweep2.pkl", "rb"))
    doms, cache, _ = ulib.load_dataset(); doms = [d for d in doms if d["id"] in z["per_dom"]]
    per = s2["variants"][("class", 3.5)]
    typ = {d["id"]: pan.type_of(d["cls"]) for d in doms}
    A = anchor_set(rows, 0.5, 1.9); sc = Scheme(A)
    lab, cl = label_all([d for d in doms if d["id"] in per], per, sc)
    st = per_anchor(doms, lab, cl, typ)
    q = {s: (v[1] / max(v[0], 1), int(v[0]), v[3] / max(v[2], 1), int(v[2])) for s, v in st.items()}
    pickle.dump(q, open("anchor_quality.pkl", "wb"))
    print("anchor: within-type precision (n) | cross-type precision (n)")
    for s in sorted(q):
        w, nw, c, nc = q[s]
        flag = " <-- weak" if w < 0.9 else ""
        print(f"{s:4d}: {w:.3f} ({nw:6d}) | {c:.3f} ({nc:6d}){flag}")
