import pickle, sys, time
sys.path.insert(0, "/tmp/pan")
import numpy as np, pan, ulib
from score import anchor_set, label_all, score, insertion_rate
from number2 import Scheme
t0 = time.time()
z = pickle.load(open("a12.pkl", "rb")); rows = pickle.load(open("rows.pkl", "rb"))
doms, cache, _ = ulib.load_dataset()
doms = [d for d in doms if d["id"] in z["per_dom"]]
imgt = {d["id"]: d["imgt"] for d in doms}
base = score(doms, imgt)
ins = insertion_rate(doms, imgt)
print(f"IMGT: P {base['all']['precision']:.3f} R {base['all']['recall']:.3f} | within P {base['within-type']['precision']:.3f} cross P {base['cross-type']['precision']:.3f} R {base['cross-type']['recall']:.3f} | ins {ins}", flush=True)
for occ in (0.4, 0.5, 0.6, 0.7):
    for spread in (1.6, 1.9, 2.2):
        A = anchor_set(rows, occ, spread)
        sc = Scheme(A)
        lab, cl = label_all(doms, z["per_dom"], sc)
        s = score(doms, lab, cl); ins = insertion_rate(doms, lab)
        print(f"occ {occ} spread {spread}: anchors {len(A):3d} | P {s['all']['precision']:.3f} R {s['all']['recall']:.3f} claimed {s['all']['claimed']:.3f} | "
              f"within P {s['within-type']['precision']:.3f} R {s['within-type']['recall']:.3f} | cross P {s['cross-type']['precision']:.3f} R {s['cross-type']['recall']:.3f} | "
              f"ins all {ins['all']:.3f} V {ins['V']:.3f} C1 {ins['C1']:.3f} C2/I {ins['C2/I']:.3f} FN3 {ins['FN3']:.3f}", flush=True)
print(f"{time.time()-t0:.0f}s")
