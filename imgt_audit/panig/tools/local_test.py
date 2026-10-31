"""Leave-one-out local superposition test of every claimed number, within and across domain types."""
import pickle, sys
sys.path.insert(0, "/tmp/pan")
import numpy as np, pan, ulib
from collections import defaultdict
from score import pairs

def kabsch_rmsd_point(P, Q, p, q):
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return np.linalg.norm((p - pc) @ R.T + qc - q)

def local_precision(doms, cache, lab, claims, window=12, min_ref=6, keyfn=None):
    """For each number claimed by both domains, superpose on other shared claimed numbers within +-window."""
    P = pairs(); ids = P["ids"]
    typ = {d["id"]: pan.type_of(d["cls"]) for d in doms}
    ca = {d["id"]: pan.geometry(d, cache)[0] for d in doms}
    keyfn = keyfn or (lambda l: int(l.split(".")[0]))
    st = defaultdict(lambda: np.zeros(4))
    for ia, ib, tm, D, eq in P["pairs"]:
        a, b = ids[ia], ids[ib]
        if a not in lab or b not in lab: continue
        cross = typ[a] != typ[b]
        pos = {l: q for q, l in enumerate(lab[b])}
        shared = [(i, pos[l], keyfn(l), l) for i, l in enumerate(lab[a]) if l in pos and i in claims[a] and pos[l] in claims[b]]
        if len(shared) < min_ref + 1: continue
        nums = np.array([s[2] for s in shared])
        A = ca[a][[s[0] for s in shared]]; B = ca[b][[s[1] for s in shared]]
        for k, (i, j, n, l) in enumerate(shared):
            ref = np.where((np.abs(nums - n) <= window) & (np.arange(len(shared)) != k))[0]
            if len(ref) < min_ref: continue
            dd = kabsch_rmsd_point(A[ref], B[ref], A[k], B[k])
            c = 2 if cross else 0
            st[l][c] += 1; st[l][c + 1] += dd <= 3.0
    return st

def summarize(st, keys=None):
    w = sum(v[0] for k, v in st.items() if keys is None or k in keys); wc = sum(v[1] for k, v in st.items() if keys is None or k in keys)
    c = sum(v[2] for k, v in st.items() if keys is None or k in keys); cc = sum(v[3] for k, v in st.items() if keys is None or k in keys)
    return wc / max(w, 1), cc / max(c, 1), int(w), int(c)
