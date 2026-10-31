"""Pan-immunoglobulin numbering: V-frame consensus, structural site matching, anchor statistics."""
import json, re, sys
from collections import Counter, defaultdict
import numpy as np

sys.path.insert(0, "/tmp/sabr_unified/tools")
import ulib  # dataset of domains with coordinates (SAbR extraction) and IMGT / structural labels

CORE = [str(i) for i in list(range(18, 26)) + list(range(39, 45)) + list(range(86, 92)) + list(range(99, 105))]
SITES = [str(i) for i in range(1, 129)]
V_CANON = {"VH", "V-KAPPA", "V-LAMBDA", "V-ALPHA", "V-BETA", "V-GAMMA", "V-DELTA"}


def type_of(cls):
    if cls in V_CANON: return "V"
    if cls == "V-LIKE": return "V-like"
    if cls == "C-LIKE:IgSF": return "C2/I"
    if cls.startswith("F:"): return "FN3"
    return "C1"


def kabsch(P, Q, w=None):
    w = np.ones(len(P)) if w is None else np.asarray(w, float)
    w = w / w.sum()
    pc, qc = (w[:, None] * P).sum(0), (w[:, None] * Q).sum(0)
    H = ((P - pc) * w[:, None]).T @ (Q - qc)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, qc - R @ pc


def dp_assign(S, thr=0.05):
    n, m = S.shape
    F = np.zeros((n + 1, m + 1))
    for i in range(1, n + 1):
        cand = np.maximum(F[i - 1, 1:], F[i - 1, :-1] + np.where(S[i - 1] > thr, S[i - 1], -1))
        F[i, 1:] = np.maximum.accumulate(cand)
    i, j, out = n, m, []
    while i > 0 and j > 0:
        if F[i, j] == F[i - 1, j]:
            i -= 1
        elif F[i, j] == F[i, j - 1]:
            j -= 1
        else:
            out.append((i - 1, j - 1)); i -= 1; j -= 1
    return out[::-1]


def geometry(d, cache):
    X = ulib.dom_coords(d, cache).astype(np.float64)
    ca, cb = X[:, 1], X[:, 3]
    u = cb - ca
    return ca, u / np.linalg.norm(u, axis=1)[:, None]


def core_pairs(d, labels_key="target"):
    """Residue indices labelled with verified-shared core sites (structural scheme labels)."""
    lab = d[labels_key]
    return [(i, l) for i, l in enumerate(lab) if l in CORE]


class Frame:
    def __init__(self, names, xyz, cbu):
        self.names = list(names); self.xyz = np.asarray(xyz, float); self.cbu = np.asarray(cbu, float)
        self.index = {n: k for k, n in enumerate(self.names)}


def match(ca, cbu, frame, R, t, sites=None, sigma=1.5, dmax=3.0, rounds=3, refine_on=None):
    """Order-preserving residue -> site matching after superposition (R, t); refines R, t on matches."""
    idx = [frame.index[s] for s in (sites or frame.names)]
    C, B = frame.xyz[idx], frame.cbu[idx]
    names = [frame.names[k] for k in idx]
    pairs = []
    for _ in range(rounds):
        X = ca @ R.T + t; U = cbu @ R.T
        D = np.linalg.norm(X[:, None] - C[None], axis=2)
        S = np.exp(-D ** 2 / (2 * sigma ** 2)) * np.clip((1 + U @ B.T) / 2, 0, 1)
        S[D > dmax] = 0
        pairs = dp_assign(S)
        use = [(i, j) for i, j in pairs if refine_on is None or names[j] in refine_on]
        if len(use) >= 12:
            w = np.exp(-np.array([D[i, j] for i, j in use]) ** 2 / 8.0) + 1e-3
            R, t = kabsch(ca[[i for i, _ in use]], C[[j for _, j in use]], w)
    X = ca @ R.T + t
    return {i: (names[j], float(np.linalg.norm(X[i] - C[j]))) for i, j in pairs}, R, t


def superpose_on_core(d, ca, frame):
    cp = [(i, l) for i, l in core_pairs(d) if l in frame.index]
    if len(cp) < 12:
        return None
    return kabsch(ca[[i for i, _ in cp]], frame.xyz[[frame.index[l] for _, l in cp]])


def build_v_frame(doms, cache, iters=3):
    """Consensus coordinates of IMGT V-frame sites 1-128 from canonical V domains (IMGT labels first,
    then structural matches)."""
    V = [d for d in doms if d["cls"] in V_CANON]
    # initial: per-label means after core superposition onto the first domain
    ref = V[0]; ca0, _ = geometry(ref, cache)
    lab0 = {l: i for i, l in enumerate(ref["imgt"])}
    names = [s for s in SITES if s in lab0]
    frame = Frame(names, [ca0[lab0[s]] for s in names], [[0, 0, 1]] * len(names))
    for it in range(iters):
        acc, accb = defaultdict(list), defaultdict(list)
        for d in V:
            ca, cbu = geometry(d, cache)
            Rt = superpose_on_core(d, ca, frame)
            if Rt is None:
                continue
            if it == 0:
                X = ca @ Rt[0].T + Rt[1]; U = cbu @ Rt[0].T
                for i, l in enumerate(d["imgt"]):
                    if l in SITES:
                        acc[l].append(X[i]); accb[l].append(U[i])
            else:
                m, R, t = match(ca, cbu, frame, *Rt, refine_on=set(CORE))
                X = ca @ R.T + t; U = cbu @ R.T
                for i, (s, dist) in m.items():
                    acc[s].append(X[i]); accb[s].append(U[i])
        names = [s for s in SITES if len(acc[s]) >= 10]
        xyz = [np.mean(acc[s], 0) for s in names]
        cb = [np.mean(accb[s], 0) for s in names]
        frame = Frame(names, xyz, [v / np.linalg.norm(v) for v in cb])
    return frame


def site_table(doms, cache, frame):
    """Match every domain to the V frame; return per-domain matches and per-class site statistics."""
    per_dom = {}
    pos = defaultdict(lambda: defaultdict(list))
    ncls = Counter(d["cls"] for d in doms)
    for d in doms:
        ca, cbu = geometry(d, cache)
        Rt = superpose_on_core(d, ca, frame)
        if Rt is None:
            continue
        m, R, t = match(ca, cbu, frame, *Rt, refine_on=set(CORE))
        X = ca @ R.T + t
        per_dom[d["id"]] = m
        for i, (s, dist) in m.items():
            pos[s][d["cls"]].append(X[i])
    stats = {}
    for s in frame.names:
        row = {}
        for c, P in pos[s].items():
            P = np.array(P); mu = P.mean(0)
            row[c] = dict(occ=len(P) / ncls[c], spread=float(np.sqrt(((P - mu) ** 2).sum(1).mean())), mean=mu.tolist())
        stats[s] = row
    return per_dom, stats


def class_frames(doms, cache, frame, per_dom, min_n=8):
    """Per-class consensus coordinates of V-frame sites, in the V frame (from pooled matches)."""
    acc = defaultdict(lambda: defaultdict(list)); accb = defaultdict(lambda: defaultdict(list))
    for d in doms:
        if d["id"] not in per_dom:
            continue
        ca, cbu = geometry(d, cache)
        Rt = superpose_on_core(d, ca, frame)
        if Rt is None:
            continue
        m, R, t = match(ca, cbu, frame, *Rt, refine_on=set(CORE))
        X = ca @ R.T + t; U = cbu @ R.T
        for i, (s, dist) in m.items():
            acc[d["cls"]][s].append(X[i]); accb[d["cls"]][s].append(U[i])
    out = {}
    ncls = Counter(d["cls"] for d in doms)
    for c in acc:
        if ncls[c] < min_n:
            continue
        names = [s for s in frame.names if len(acc[c][s]) >= max(3, 0.3 * ncls[c])]
        xyz = [np.mean(acc[c][s], 0) for s in names]
        cb = [np.mean(accb[c][s], 0) for s in names]
        out[c] = Frame(names, xyz, [v / (np.linalg.norm(v) + 1e-9) for v in cb])
    return out


def site_matches(doms, cache, frame, cframes=None, dmax=3.0):
    per = {}
    for d in doms:
        ca, cbu = geometry(d, cache)
        f = cframes.get(d["cls"], frame) if cframes else frame
        Rt = superpose_on_core(d, ca, f)
        if Rt is None:
            Rt = superpose_on_core(d, ca, frame); f = frame
            if Rt is None:
                continue
        m, R, t = match(ca, cbu, f, *Rt, refine_on=set(CORE), dmax=dmax)
        per[d["id"]] = m
    return per
