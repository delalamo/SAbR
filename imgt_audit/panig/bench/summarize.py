"""Tables for the algorithm benchmark: oracle agreement (clean), robustness, honesty."""
import pickle, sys, json
sys.path.insert(0, "/tmp/pan/bench"); sys.path.insert(0, "/tmp/pan"); sys.path.insert(0, "/tmp/sabr_unified/tools")
import numpy as np, pan, ulib
from number2 import Scheme
doms, cache, _ = ulib.load_dataset(); byid = {d["id"]: d for d in doms}
S = pickle.load(open("/tmp/pan/bench/setup.pkl", "rb")); f2 = pickle.load(open("/tmp/pan/final2.pkl", "rb"))
SC = Scheme(S["A"]); out = {}
clean = pickle.load(open("/tmp/pan/bench/clean.pkl", "rb"))
meths = list(next(iter(clean.values())).keys())
print("## clean held-out: anchor P/R vs oracle, full-label agreement L")
for m in meths:
    rows = {}
    for did, o in clean.items():
        pred, dt = o[m][0], o[m][1]; orc = S["oracle"][did]; d = byid[did]; t = pan.type_of(d["cls"])
        lab = SC.number(len(d["target"]), pred) if pred else [None] * len(d["target"])
        v = np.array([sum(orc.get(i) == a for i, a in pred.items()), len(pred), len(orc),
                      sum(x == y for x, y in zip(lab, f2["labels"][did])), len(lab), dt, 1])
        for k in ("all", t): rows[k] = rows.get(k, 0) + v
    out[m] = {k: dict(P=v[0] / max(v[1], 1), R=v[0] / v[2], L=v[3] / v[4], n=int(v[6])) for k, v in rows.items()}
    print(f"{m:38s} | " + " | ".join(f"{k} {v[0]/max(v[1],1):.3f}/{v[0]/v[2]:.3f}/{v[3]/v[4]:.3f}" for k, v in
                                      sorted(rows.items(), key=lambda x: ["all", "V", "V-like", "C1", "C2/I", "FN3"].index(x[0]))))
rob = pickle.load(open("/tmp/pan/bench/robust.pkl", "rb"))
conds = list(next(iter(rob.values())).keys())
print("\n## robustness (anchor P vs oracle / R vs oracle / same as own clean)")
R = {}
for cond in conds:
    print("==", cond)
    for m in meths:
        v = np.zeros(5)
        for did, o in rob.items():
            if m not in o[cond]: continue
            orc = S["oracle"][did]; present = set(o[cond]["_present"]); pred = o[cond][m]
            mapped = {oi: a for (oi, a) in pred.values() if oi is not None}
            cl = clean[did][m][0]
            v += [sum(orc.get(i) == a for i, a in mapped.items()), len(pred), sum(1 for i in orc if i in present),
                  sum(cl.get(i) == mapped.get(i) for i in present), len(present)]
        if v[4]:
            R[f"{cond}|{m}"] = dict(P=v[0] / max(v[1], 1), R=v[0] / v[2], self=v[3] / v[4])
            print(f"   {m:38s} {v[0]/max(v[1],1):.3f} / {v[0]/v[2]:.3f} / {v[3]/v[4]:.3f}")
json.dump(dict(clean=out, robust=R), open("/tmp/pan/bench/summary.json", "w"), indent=1)
