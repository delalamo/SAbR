"""Cache pairwise TM-align superpositions and structural equivalences once."""
import pickle, random, sys
from multiprocessing import Pool
import numpy as np, tmtools
sys.path.insert(0, "/tmp/pan")
import pan, ulib
G = {}
def _pair(args):
    ia, ib = args
    a, b = G["data"][ia], G["data"][ib]
    r = tmtools.tm_align(a["ca"], b["ca"], a["seq"], b["seq"])
    tm = max(r.tm_norm_chain1, r.tm_norm_chain2)
    if tm < 0.5:
        return None
    Xa = a["ca"] @ r.u.T + r.t
    D = np.linalg.norm(Xa[:, None] - b["ca"][None], axis=2).astype(np.float32)
    eq, i, j = [], 0, 0
    for x, y in zip(r.seqxA, r.seqyA):
        if x != "-" and y != "-" and D[i, j] <= 3.0:
            eq.append((i, j))
        i += x != "-"; j += y != "-"
    return ia, ib, tm, D, eq
if __name__ == "__main__":
    doms, cache, _ = ulib.load_dataset()
    data = []
    for d in doms:
        ca, _ = pan.geometry(d, cache); c = cache[(d["pdb"], d["chain"])]; a0, b0 = d["rows"]
        data.append(dict(id=d["id"], ca=ca, seq="".join(c["seq"][a0:b0])))
    G["data"] = data
    rnd = random.Random(0)
    jobs = [(i, j) for i in range(len(data)) for j in rnd.sample(range(len(data)), 10) if j != i]
    with Pool(4) as p:
        res = [x for x in p.map(_pair, jobs, chunksize=20) if x]
    pickle.dump(dict(ids=[d["id"] for d in data], pairs=res), open("/tmp/pan/pairs.pkl", "wb"))
    print("pairs", len(res))
