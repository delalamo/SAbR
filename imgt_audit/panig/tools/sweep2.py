import pickle, sys, time
sys.path.insert(0, "/tmp/pan")
import numpy as np, pan, ulib
from score import anchor_set, label_all, score, insertion_rate
from number2 import Scheme
t0 = time.time()
z = pickle.load(open("a12.pkl", "rb")); rows = pickle.load(open("rows.pkl", "rb"))
doms, cache, _ = ulib.load_dataset(); doms = [d for d in doms if d["id"] in z["per_dom"]]
frame = pan.Frame(*z["frame"])
cfr = pan.class_frames(doms, cache, frame, z["per_dom"])
print("class frames", len(cfr), f"{time.time()-t0:.0f}s", flush=True)
V = [d for d in doms if pan.type_of(d["cls"]) == "V"]
def imgt_agree(lab):
    return np.mean([l == i for d in V for l, i in zip(lab[d["id"]], d["imgt"])])
variants = {}
for model in ("pooled", "class"):
    for dmax in (3.0, 3.5, 4.0):
        per = pan.site_matches(doms, cache, frame, cfr if model == "class" else None, dmax)
        variants[(model, dmax)] = per
        for occ, spread in ((0.5, 1.6), (0.5, 1.9)):
            sc = Scheme(anchor_set(rows, occ, spread))
            lab, cl = label_all([d for d in doms if d["id"] in per], per, sc)
            dd = [d for d in doms if d["id"] in lab]
            s = score(dd, lab, cl); ins = insertion_rate(dd, lab)
            print(f"{model:6s} dmax {dmax} anchors {len(sc.anchors):3d}: P {s['all']['precision']:.3f} R {s['all']['recall']:.3f} claimed {s['all']['claimed']:.3f} | "
                  f"within P {s['within-type']['precision']:.3f} R {s['within-type']['recall']:.3f} | cross P {s['cross-type']['precision']:.3f} R {s['cross-type']['recall']:.3f} | "
                  f"ins V {ins['V']:.3f} C1 {ins['C1']:.3f} C2/I {ins['C2/I']:.3f} FN3 {ins['FN3']:.3f} | V=IMGT {imgt_agree(lab):.3f}", flush=True)
pickle.dump(dict(cfr={c: (f.names, f.xyz, f.cbu) for c, f in cfr.items()}, variants=variants), open("sweep2.pkl", "wb"))
print(f"{time.time()-t0:.0f}s")
