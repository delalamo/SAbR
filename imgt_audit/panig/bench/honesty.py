"""Oracle-free evaluation: local leave-one-out honesty of each method's anchor claims on held-out pairs."""
import pickle, sys, random
sys.path.insert(0, "/tmp/pan/bench"); sys.path.insert(0, "/tmp/pan"); sys.path.insert(0, "/tmp/sabr_unified/tools")
import numpy as np, pan, ulib, tmtools
from multiprocessing import Pool
from local_test import kabsch_rmsd_point
doms, cache, _ = ulib.load_dataset(); byid = {d["id"]: d for d in doms}
res = pickle.load(open(sys.argv[1] if len(sys.argv) > 1 else "/tmp/pan/bench/clean.pkl", "rb"))
S = pickle.load(open("/tmp/pan/bench/setup.pkl", "rb"))
ids = sorted(res); CA = {i: pan.geometry(byid[i], cache)[0] for i in ids}
SEQ = {}
for i in ids:
    d = byid[i]; c = cache[(d["pdb"], d["chain"])]; a, b = d["rows"]; SEQ[i] = "".join(c["seq"][a:b])
rnd = random.Random(0)
cand = [(a, b) for a in ids for b in rnd.sample(ids, 25) if a < b and byid[a]["pdb"] != byid[b]["pdb"]]

def tm(p):
    a, b = p
    r = tmtools.tm_align(CA[a], CA[b], SEQ[a], SEQ[b])
    return p if max(r.tm_norm_chain1, r.tm_norm_chain2) >= 0.5 else None

def local(claims_a, claims_b, a, b, window=12, min_ref=6):
    inv = {s: j for j, s in claims_b.items()}
    sh = sorted((s, i, inv[s]) for i, s in claims_a.items() if s in inv)
    if len(sh) < min_ref + 1: return 0, 0
    nums = np.array([s for s, _, _ in sh]); Pa = CA[a][[i for _, i, _ in sh]]; Pb = CA[b][[j for _, _, j in sh]]
    n = ok = 0
    for k, (s, i, j) in enumerate(sh):
        ref = np.where((np.abs(nums - s) <= window) & (np.arange(len(sh)) != k))[0]
        if len(ref) < min_ref: continue
        n += 1; ok += kabsch_rmsd_point(Pa[ref], Pb[ref], Pa[k], Pb[k]) <= 3.0
    return n, ok

if __name__ == "__main__":
    with Pool(4) as p:
        prs = [x for x in p.map(tm, cand, chunksize=50) if x]
    meths = list(next(iter(res.values())).keys())
    claims = {m: {i: res[i][m][0] for i in ids} for m in meths}
    claims["oracle (known class + prior core labels)"] = {i: S["oracle"][i] for i in ids}
    typ = {i: pan.type_of(byid[i]["cls"]) for i in ids}
    print(f"{len(prs)} held-out pairs with TM>=0.5 ({sum(typ[a] != typ[b] for a, b in prs)} cross-type)")
    out = {}
    for m, C in claims.items():
        v = np.zeros(4)
        for a, b in prs:
            n, ok = local(C[a], C[b], a, b); c = 2 if typ[a] != typ[b] else 0
            v[c] += n; v[c + 1] += ok
        cov = np.mean([len(C[i]) for i in ids])
        out[m] = dict(within=v[1] / v[0], cross=v[3] / v[2], n_within=int(v[0]), n_cross=int(v[2]), anchors_per_domain=cov)
        print(f"{m:44s} anchors/domain {cov:5.1f} | local honesty within {v[1]/v[0]:.4f} (n={int(v[0])})  cross {v[3]/v[2]:.4f} (n={int(v[2])})")
    pickle.dump(out, open(sys.argv[1].replace(".pkl", "_honesty.pkl") if len(sys.argv) > 1 else "/tmp/pan/bench/clean_honesty.pkl", "wb"))
