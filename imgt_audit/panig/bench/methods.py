"""Label-free pan-Ig anchor assignment algorithms. Each returns {residue index: anchor int}."""
import pickle, sys, time
sys.path.insert(0, "/tmp/pan"); sys.path.insert(0, "/tmp/sabr_unified/tools"); sys.path.insert(0, "/home/user/SAbR/src")
import numpy as np, tmtools, pan, ulib
from sabr.alignment import _align_reference

S = pickle.load(open("/tmp/pan/bench/setup.pkl", "rb"))
A = S["A"]; AS = set(A)
FRAME = pan.Frame(*S["frame"])
CFR = {c: pan.Frame(*v) for c, v in S["cfr"].items()}
FRAMES = dict(CFR, pooled=FRAME)
_z = np.load("/tmp/pan/bench/emb_ref.npz")
EMB_NAMES = [int(x) for x in _z["names"]]; EMB_REF = _z["ref"]
CORE = set(pan.CORE)
TEMPL = {}  # id -> dict(ca, seq, site {i: anchor})


def load_templates(doms, cache):
    for d in doms:
        if d["id"] in S["templates"]:
            ca, _ = pan.geometry(d, cache)
            c = cache[(d["pdb"], d["chain"])]; a, b = d["rows"]
            TEMPL[d["id"]] = dict(ca=ca, seq="".join(c["seq"][a:b]), site=S["oracle"][d["id"]], cls=d["cls"])


def _aligned(r):
    i = j = 0
    for x, y in zip(r.seqxA, r.seqyA):
        if x != "-" and y != "-":
            yield i, j
        i += x != "-"; j += y != "-"


# ---------------------------------------------------------------- M1/M2: TM-align label transfer
def tm_transfer(ca, seq, k=1, dmax=3.5):
    hits = []
    for tid, T in TEMPL.items():
        r = tmtools.tm_align(ca, T["ca"], seq, T["seq"])
        hits.append((r.tm_norm_chain2, tid, r))
    hits.sort(key=lambda h: -h[0])
    votes = np.zeros((len(ca), len(A))); col = {a: q for q, a in enumerate(A)}
    for tm, tid, r in hits[:k]:
        X = ca @ r.u.T + r.t; T = TEMPL[tid]
        for i, j in _aligned(r):
            if j in T["site"] and np.linalg.norm(X[i] - T["ca"][j]) <= dmax:
                votes[i, col[T["site"][j]]] += tm
    if k == 1:
        return {i: A[int(np.argmax(v))] for i, v in enumerate(votes) if v.max() > 0}, hits[0][1]
    pairs = pan.dp_assign(votes / max(votes.max(), 1e-9), thr=0.3 * hits[0][0] * 0 + 0.25)
    return {i: A[j] for i, j in pairs}, hits[0][1]


# ---------------------------------------------------------------- M3: SAbR-style embedding alignment
def emb_assign(E, gaps=frozenset()):
    soft, _, score = _align_reference(E, EMB_REF * 3.0, query_gap_indices=frozenset(gaps))
    rows, cols = np.nonzero(soft > 0.5)
    return {int(r): EMB_NAMES[int(c)] for r, c in zip(rows, cols)}, float(score)


def emb_logodds(E):
    return E @ (EMB_REF * 3.0).T  # [n, 82]


# ---------------------------------------------------------------- M4: consensus-frame matching (TM init)
def _score_match(m, sigma=1.5):
    return sum(np.exp(-dist ** 2 / (2 * sigma ** 2)) for s, dist in m.values())


def frame_match(ca, cbu, init="tm", seed=None, frames=None, dmax=3.5):
    """Superpose onto every class consensus frame, match residues to frame sites, keep the best frame."""
    best = None
    for c, f in (frames or FRAMES).items():
        if init == "tm":
            r = tmtools.tm_align(ca, f.xyz, "A" * len(ca), "A" * len(f.xyz))
            R, t = r.u, r.t
        else:  # seed: {i: anchor} from another method
            pr = [(i, f.index[str(a)]) for i, a in seed.items() if str(a) in f.index]
            if len(pr) < 8:
                continue
            R, t = pan.kabsch(ca[[i for i, _ in pr]], f.xyz[[j for _, j in pr]])
        m, R, t = pan.match(ca, cbu, f, R, t, refine_on=CORE, dmax=dmax)
        sc = _score_match(m)
        if best is None or sc > best[0]:
            best = (sc, c, m)
    if best is None:
        return {}, None, 0.0
    return {i: int(s) for i, (s, dist) in best[2].items() if int(s) in AS}, best[1], best[0]


# ---------------------------------------------------------------- M5b: fused structure + embedding DP
def fused_match(ca, cbu, E, seed, frames=None, dmax=3.5, lam=0.5, sigma=1.5, rounds=3):
    L = np.clip(emb_logodds(E), -50, 50); P = 1 / (1 + np.exp(-L))  # residue-anchor compatibility in (0,1)
    col = {a: q for q, a in enumerate(EMB_NAMES)}
    best = None
    for c, f in (frames or FRAMES).items():
        pr = [(i, f.index[str(a)]) for i, a in seed.items() if str(a) in f.index]
        if len(pr) < 8:
            continue
        R, t = pan.kabsch(ca[[i for i, _ in pr]], f.xyz[[j for _, j in pr]])
        names = f.names; Pe = np.zeros((len(ca), len(names)))
        for j, s in enumerate(names):
            if int(s) in col: Pe[:, j] = P[:, col[int(s)]]
        for _ in range(rounds):
            X = ca @ R.T + t; U = cbu @ R.T
            D = np.linalg.norm(X[:, None] - f.xyz[None], axis=2)
            Ss = np.exp(-D ** 2 / (2 * sigma ** 2)) * np.clip((1 + U @ f.cbu.T) / 2, 0, 1)
            Ss[D > dmax] = 0
            Sf = Ss * ((1 - lam) + lam * Pe)
            pairs = pan.dp_assign(Sf)
            use = [(i, j) for i, j in pairs if names[j] in CORE]
            if len(use) >= 12:
                w = np.exp(-np.array([D[i, j] for i, j in use]) ** 2 / 8.0) + 1e-3
                R, t = pan.kabsch(ca[[i for i, _ in use]], f.xyz[[j for _, j in use]], w)
        sc = sum(Sf[i, j] for i, j in pairs)
        if best is None or sc > best[0]:
            best = (sc, c, {i: int(names[j]) for i, j in pairs})
    if best is None:
        return {}, None, 0.0
    return {i: a for i, a in best[2].items() if a in AS}, best[1], best[0]


# ---------------------------------------------------------------- M6: dual-init fused matching
def _fused_from(ca, cbu, Pfull, f, R, t, lam, dmax=3.5, sigma=1.5, rounds=3, cbw=1.0):
    col = {a: q for q, a in enumerate(EMB_NAMES)}
    names = f.names; Pe = np.zeros((len(ca), len(names)))
    for j, s in enumerate(names):
        if int(s) in col: Pe[:, j] = Pfull[:, col[int(s)]]
    pairs, Sf = [], None
    for _ in range(rounds):
        X = ca @ R.T + t; U = cbu @ R.T
        D = np.linalg.norm(X[:, None] - f.xyz[None], axis=2)
        Ss = np.exp(-D ** 2 / (2 * sigma ** 2)) * ((1 - cbw) + cbw * np.clip((1 + U @ f.cbu.T) / 2, 0, 1))
        Ss[D > dmax] = 0
        Sf = Ss * ((1 - lam) + lam * Pe)
        pairs = pan.dp_assign(Sf)
        use = [(i, j) for i, j in pairs if names[j] in CORE]
        if len(use) >= 12:
            w = np.exp(-np.array([D[i, j] for i, j in use]) ** 2 / 8.0) + 1e-3
            R, t = pan.kabsch(ca[[i for i, _ in use]], f.xyz[[j for _, j in use]], w)
    struct = sum(Ss[i, j] for i, j in pairs)
    return sum(Sf[i, j] for i, j in pairs), {i: int(names[j]) for i, j in pairs}, struct


def dual_match(ca, cbu, E, seed, lam=0.5, frames=None, cbw=1.0, select="fused"):
    """M6: initialisations from (a) embedding-predicted anchors and (b) one TM-align to the pooled
    consensus frame (all class frames share its coordinates); fused structure x embedding DP on every
    class frame; the highest fused score wins."""
    P = 1 / (1 + np.exp(-np.clip(emb_logodds(E), -50, 50)))
    r = tmtools.tm_align(ca, FRAME.xyz, "A" * len(ca), "A" * len(FRAME.xyz))
    inits = [("tm", r.u, r.t)]
    pr = [(i, FRAME.index[str(a)]) for i, a in seed.items() if str(a) in FRAME.index]
    if len(pr) >= 8:
        inits.append(("emb",) + tuple(pan.kabsch(ca[[i for i, _ in pr]], FRAME.xyz[[j for _, j in pr]])))
    best = None
    for c, f in (frames or FRAMES).items():
        for tag, R, t in inits:
            sc, m, st = _fused_from(ca, cbu, P, f, R, t, lam, cbw=cbw)
            if select == "struct": sc = st
            if best is None or sc > best[0]:
                best = (sc, c, m, tag)
    return {i: a for i, a in best[2].items() if a in AS}, best[1], best[0], best[3]


def frame_match2(ca, cbu, E=None, seed=None, init="tm_each", lam=0.0, cbw=0.5, select="struct", frames=None, sigma=1.5, dmax=3.5):
    """Variants of consensus-frame matching for the tuning study."""
    P = 1 / (1 + np.exp(-np.clip(emb_logodds(E), -50, 50))) if E is not None else np.zeros((len(ca), len(EMB_NAMES)))
    inits_pooled = []
    if init in ("tm_pooled", "tm_pooled+emb"):
        r = tmtools.tm_align(ca, FRAME.xyz, "A" * len(ca), "A" * len(FRAME.xyz)); inits_pooled.append((r.u, r.t))
    if init in ("tm_pooled+emb", "tm_each+emb") and seed:
        pr = [(i, FRAME.index[str(a)]) for i, a in seed.items() if str(a) in FRAME.index]
        if len(pr) >= 8: inits_pooled.append(tuple(pan.kabsch(ca[[i for i, _ in pr]], FRAME.xyz[[j for _, j in pr]])))
    best = None
    for c, f in (frames or FRAMES).items():
        inits = list(inits_pooled)
        if init.startswith("tm_each"):
            r = tmtools.tm_align(ca, f.xyz, "A" * len(ca), "A" * len(f.xyz)); inits.append((r.u, r.t))
        for R, t in inits:
            sc, m, st = _fused_from(ca, cbu, P, f, R, t, lam, cbw=cbw, sigma=sigma, dmax=dmax)
            sc = st if select == "struct" else sc
            if best is None or sc > best[0]: best = (sc, c, m)
    return {i: a for i, a in best[2].items() if a in AS}, best[1], best[0]


def adaptive_match(ca, cbu):
    """M7: consensus-frame matching with per-class TM-align initialisation; the C-beta direction term is
    halved when the backbone is geometrically irregular (std of consecutive CA-CA distances > 0.3 A)."""
    dd = np.linalg.norm(np.diff(ca, axis=0), axis=1); dd = dd[dd < 4.5]
    cbw = 1.0 if (len(dd) < 5 or np.std(dd) < 0.3) else 0.5
    return frame_match2(ca, cbu, cbw=cbw)
