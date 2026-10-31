"""Clean held-out benchmark of label-free assignment algorithms against pan-Ig oracle anchors."""
import pickle, sys, time, json
sys.path.insert(0, "/tmp/pan/bench"); sys.path.insert(0, "/tmp/pan"); sys.path.insert(0, "/tmp/sabr_unified/tools")
import numpy as np, pan, ulib, methods as M
from multiprocessing import Pool
from number2 import Scheme
doms, cache, _ = ulib.load_dataset()
byid = {d["id"]: d for d in doms}
M.load_templates([byid[t] for t in M.S["templates"]], cache)
test = [byid[i] for i in M.S["test"]]
if len(sys.argv) > 1: test = test[::int(sys.argv[1])]
SC = Scheme(M.A)

def run(d):
    ca, cbu = pan.geometry(d, cache); E = ulib.dom_emb(d, cache)
    c = cache[(d["pdb"], d["chain"])]; a, b = d["rows"]; seq = "".join(c["seq"][a:b])
    out = {}
    def rec(name, f):
        t = time.time(); r = f(); out[name] = (r[0], time.time() - t, r[1] if len(r) > 1 else None)
    rec("M1 TM-align best template", lambda: M.tm_transfer(ca, seq, 1))
    rec("M2 TM-align top-5 vote", lambda: M.tm_transfer(ca, seq, 5))
    rec("M3 SAbR embedding (pan-trained)", lambda: M.emb_assign(E))
    rec("M4 consensus-frame match (TM init)", lambda: M.frame_match(ca, cbu, "tm")[:2])
    seed = out["M3 SAbR embedding (pan-trained)"][0]
    rec("M5a embedding seed + frame match", lambda: M.frame_match(ca, cbu, "seed", seed)[:2])
    rec("M5b fused structure+embedding DP", lambda: M.fused_match(ca, cbu, E, seed)[:2])
    t1 = out["M3 SAbR embedding (pan-trained)"][1]
    out["M5a embedding seed + frame match"] = out["M5a embedding seed + frame match"][:1] + (out["M5a embedding seed + frame match"][1] + t1,) + out["M5a embedding seed + frame match"][2:]
    out["M5b fused structure+embedding DP"] = out["M5b fused structure+embedding DP"][:1] + (out["M5b fused structure+embedding DP"][1] + t1,) + out["M5b fused structure+embedding DP"][2:]
    return d["id"], out

if __name__ == "__main__":
    t0 = time.time()
    with Pool(4) as p:
        res = dict(p.map(run, test, chunksize=4))
    pickle.dump(res, open("/tmp/pan/bench/clean.pkl", "wb"))
    f2 = pickle.load(open("/tmp/pan/final2.pkl", "rb"))
    print(f"{len(res)} test domains, {time.time()-t0:.0f}s")
    meths = list(next(iter(res.values())).keys())
    for m in meths:
        rows = {}
        for did, out in res.items():
            pred, dt, extra = out[m]; o = M.S["oracle"][did]; d = byid[did]; t = pan.type_of(d["cls"])
            lab = SC.number(len(d["target"]), pred) if pred else [None] * len(d["target"])
            v = np.array([sum(o.get(i) == a for i, a in pred.items()), len(pred), len(o),
                          sum(x == y for x, y in zip(lab, f2["labels"][did])), len(lab), dt, 1])
            for k in ("all", t): rows[k] = rows.get(k, 0) + v
        line = " | ".join(f"{k} P{v[0]/max(v[1],1):.3f} R{v[0]/v[2]:.3f} L{v[3]/v[4]:.3f}" for k, v in sorted(rows.items(), key=lambda x: x[0] != "all"))
        print(f"{m:38s} {rows['all'][5]/rows['all'][6]*1000:7.0f} ms/dom | {line}")
