"""Robustness: coordinate noise, missing termini, internal chain break, flanking chain context."""
import pickle, sys, time, zlib
sys.path.insert(0, "/tmp/pan/bench"); sys.path.insert(0, "/tmp/pan"); sys.path.insert(0, "/tmp/sabr_unified/tools"); sys.path.insert(0, "/home/user/SAbR/src")
import numpy as np, pan, ulib, methods as M
from multiprocessing import Pool
from sabr.model import encode
doms, cache, _ = ulib.load_dataset(); byid = {d["id"]: d for d in doms}
M.load_templates([byid[t] for t in M.S["templates"]], cache)
test = [byid[i] for i in M.S["test"]][::2]
COND = ["domain only", "noise 0.5 A", "noise 1.0 A", "termini -7/-7", "internal break (5 missing)", "chain context +-30"]

def with_cb(bb):
    n, ca, cc = bb[:, 0], bb[:, 1], bb[:, 2]; b, e = ca - n, cc - ca
    cb = -0.58273431 * np.cross(b, e) + 0.56802827 * b - 0.54067466 * e + ca
    return np.concatenate([bb, cb[:, None]], 1).astype(np.float32)

def perturb(d, cond):
    c = cache[(d["pdb"], d["chain"])]; a, b = d["rows"]; X = c["coords"].astype(np.float64); seq = c["seq"]
    n = b - a; idx = list(range(n)); gaps = set(); lo, hi = a, b
    if cond == "chain context +-30":
        lo, hi = max(0, a - 30), min(len(X), b + 30)
        idx = [k - a if a <= k < b else None for k in range(lo, hi)]
    Y = X[lo:hi].copy(); s = list(seq[lo:hi])
    if cond.startswith("noise"):
        sig = float(cond.split()[1]); rng = np.random.default_rng(zlib.crc32(d["id"].encode()))
        Y = with_cb(Y[:, :3] + rng.normal(0, sig, Y[:, :3].shape))
    keep = list(range(len(Y)))
    if cond == "termini -7/-7": keep = keep[7:-7]
    if cond == "internal break (5 missing)":
        m = n // 2; keep = [k for k in keep if not m - 2 <= k <= m + 2]; gaps = {m - 3 - 0}
    Y = Y[keep]; idx = [idx[k] for k in keep]; s = "".join(s[k] for k in keep)
    if cond == "internal break (5 missing)": gaps = {keep.index(m - 3)}
    return Y.astype(np.float32), idx, s, frozenset(gaps)

def run(d):
    out = {}
    for cond in COND:
        Y, idx, s, gaps = perturb(d, cond)
        ca = Y[:, 1].astype(np.float64); u = (Y[:, 3] - Y[:, 1]).astype(np.float64); cbu = u / np.linalg.norm(u, axis=1)[:, None]
        E = encode(Y)
        r = {}
        r["M1 TM-align best template"] = M.tm_transfer(ca, s, 1)[0]
        r["M2 TM-align top-5 vote"] = M.tm_transfer(ca, s, 5)[0]
        r["M3 SAbR embedding (pan-trained)"] = seed = M.emb_assign(E, gaps)[0]
        r["M4 consensus-frame match (TM init)"] = M.frame_match(ca, cbu, "tm")[0]
        r["M5a embedding seed + frame match"] = M.frame_match(ca, cbu, "seed", seed)[0]
        r["M5b fused structure+embedding DP"] = M.fused_match(ca, cbu, E, seed)[0]
        out[cond] = {m: {i: (idx[i], a) for i, a in p.items()} for m, p in r.items()}
        out[cond]["_present"] = [x for x in idx if x is not None]
    return d["id"], out

if __name__ == "__main__":
    t0 = time.time()
    with Pool(4) as p:
        res = dict(p.map(run, test, chunksize=2))
    pickle.dump(res, open("/tmp/pan/bench/robust.pkl", "wb"))
    clean = pickle.load(open("/tmp/pan/bench/clean.pkl", "rb"))
    print(f"{len(res)} held-out domains, {time.time()-t0:.0f}s")
    meths = [k for k in next(iter(res.values()))["domain only"] if not k.startswith("_")]
    summary = {}
    for cond in COND:
        print("==", cond)
        for m in meths:
            v = np.zeros(5)
            for did, out in res.items():
                o = M.S["oracle"][did]; present = set(out[cond]["_present"]); pred = out[cond][m]
                mapped = {oi: a for (oi, a) in pred.values() if oi is not None}
                spurious = sum(1 for (oi, a) in pred.values() if oi is None)
                cl = clean[did][m][0]
                v += [sum(o.get(i) == a for i, a in mapped.items()), len(pred), sum(1 for i in o if i in present),
                      sum(cl.get(i) == mapped.get(i) for i in present), len(present)]
            summary[(cond, m)] = dict(P=v[0] / v[1], R=v[0] / v[2], self=v[3] / v[4])
            print(f"   {m:38s} anchor P {v[0]/v[1]:.3f} R {v[0]/v[2]:.3f} | same as clean {v[3]/v[4]:.3f}")
    pickle.dump(summary, open("/tmp/pan/bench/robust_summary.pkl", "wb"))
