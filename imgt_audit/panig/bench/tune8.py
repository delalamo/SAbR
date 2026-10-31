import sys; sys.path[:0] = ["/tmp/pan/bench", "/tmp/pan", "/tmp/sabr_unified/tools", "/home/user/SAbR/src"]
import numpy as np, pan, ulib, methods as M, robust as RB
from multiprocessing import Pool
from sabr.model import encode
VAR2 = {"cb.5 s1.5 d3.5": dict(), "cb.5 s2 d4": dict(sigma=2.0, dmax=4.0), "cb.5 s1.5 d4": dict(dmax=4.0), "cb.5 s2 d3.5": dict(sigma=2.0), "cb.5 s2.5 d4.5": dict(sigma=2.5, dmax=4.5)}
VAR_OLD = {"lam.5 cb1 fused": dict(lam=0.5, cbw=1.0), "lam.5 cb.5 fused": dict(lam=0.5, cbw=0.5),
       "lam.5 cb0 fused": dict(lam=0.5, cbw=0.0), "lam.5 cb.5 struct": dict(lam=0.5, cbw=0.5, select="struct"),
       "lam.25 cb.5 struct": dict(lam=0.25, cbw=0.5, select="struct"), "lam0 cb.5": dict(lam=0.0, cbw=0.5),
       "lam0 cb1": dict(lam=0.0, cbw=1.0)}
VAR = VAR2
CONDS = ["domain only", "noise 0.5 A", "noise 1.0 A"]
def one(d):
    out = {}
    for cond in CONDS:
        Y, idx, s, gaps = RB.perturb(d, cond)
        ca = Y[:, 1].astype(np.float64); u = (Y[:, 3] - Y[:, 1]).astype(np.float64); cbu = u / np.linalg.norm(u, axis=1)[:, None]
        E = encode(Y); seed = M.emb_assign(E, gaps)[0]; o = M.S["oracle"][d["id"]]
        for k, kw in VAR.items():
            p = M.frame_match(ca, cbu, "tm")[0] if kw is None else M.frame_match2(ca, cbu, E, seed, **kw)[0]
            out[(cond, k)] = np.array([sum(o.get(i) == a for i, a in p.items()), len(p), len(o)])
    return out
if __name__ == "__main__":
    doms = RB.test[::4]
    with Pool(4) as pool: R = pool.map(one, doms)
    print(len(doms), "domains")
    for cond in CONDS:
        for k in VAR:
            v = sum(r[(cond, k)] for r in R); print(f"{cond:12s} {k:20s} P {v[0]/v[1]:.3f} R {v[0]/v[2]:.3f}")
