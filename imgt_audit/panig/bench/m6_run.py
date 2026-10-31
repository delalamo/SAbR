import sys; sys.path[:0] = ["/tmp/pan/bench", "/tmp/pan", "/tmp/sabr_unified/tools", "/home/user/SAbR/src"]
import pickle, sys, time
# Used twice: once with M.frame_match2 (M6, NAME "M6 consensus-frame matcher v2 (new)") and, as here, with M.adaptive_match (M7).
sys.path.insert(0, "/tmp/pan/bench")
import numpy as np, pan, ulib, methods as M
from multiprocessing import Pool
import robust as RB
from sabr.model import encode
NAME = "M7 adaptive consensus-frame matcher (new)"
byid = RB.byid; cache = RB.cache

def clean_one(did):
    d = byid[did]; ca, cbu = pan.geometry(d, cache); E = ulib.dom_emb(d, cache)
    t = time.time(); p, c, sc = M.adaptive_match(ca, cbu)
    return did, (p, time.time() - t, c, "tm")

def robust_one(did):
    d = byid[did]; out = {}
    for cond in RB.COND:
        Y, idx, s, gaps = RB.perturb(d, cond)
        ca = Y[:, 1].astype(np.float64); u = (Y[:, 3] - Y[:, 1]).astype(np.float64); cbu = u / np.linalg.norm(u, axis=1)[:, None]
        p = M.adaptive_match(ca, cbu)[0]
        out[cond] = {i: (idx[i], a) for i, a in p.items()}
    return did, out

if __name__ == "__main__":
    with Pool(4) as pool:
        c = dict(pool.map(clean_one, M.S["test"], chunksize=4))
        clean = pickle.load(open("/tmp/pan/bench/clean.pkl", "rb"))
        for did, v in c.items(): clean[did][NAME] = (v[0], v[1], v[2])
        pickle.dump(clean, open("/tmp/pan/bench/clean.pkl", "wb"))
        print("init chosen:", {k: sum(1 for v in c.values() if v[3] == k) for k in ("tm", "emb")}, "ms/dom", 1000 * np.mean([v[1] for v in c.values()]))
        rob = pickle.load(open("/tmp/pan/bench/robust.pkl", "rb"))
        r = dict(pool.map(robust_one, sorted(rob), chunksize=2))
        for did, out in r.items():
            for cond, p in out.items(): rob[did][cond][NAME] = p
        pickle.dump(rob, open("/tmp/pan/bench/robust.pkl", "wb"))
    print("done")
