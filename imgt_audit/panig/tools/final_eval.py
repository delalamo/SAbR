import json, pickle, sys
sys.path.insert(0, "/tmp/pan")
import numpy as np, pan, ulib
from collections import Counter, defaultdict
from score import anchor_set, label_all, score, insertion_rate
from number2 import Scheme
from local_test import local_precision, summarize
z = pickle.load(open("a12.pkl", "rb")); rows = pickle.load(open("rows.pkl", "rb")); s2 = pickle.load(open("sweep2.pkl", "rb"))
q = pickle.load(open("anchor_local.pkl", "rb"))
doms, cache, _ = ulib.load_dataset(); doms = [d for d in doms if d["id"] in z["per_dom"]]
per = s2["variants"][("class", 3.5)]
A0 = anchor_set(rows, 0.5, 1.9)
fail = {s for s, (w, nw, c, nc) in q.items() if w < 0.9 or (nc >= 200 and c < 0.8)}
A = sorted(A0 - fail)
sc = Scheme(A)
print("final anchors", len(A), A)
print("segments:", [(s[0], s[-1], sc.apex.get(k)) for k, s in enumerate(sc.segments)])
dd = [d for d in doms if d["id"] in per]
lab, cl = label_all(dd, per, sc)
s = score(dd, lab, cl); ins = insertion_rate(dd, lab)
imgt = {d["id"]: d["imgt"] for d in dd}; si = score(dd, imgt); insi = insertion_rate(dd, imgt)
V = [d for d in dd if pan.type_of(d["cls"]) == "V"]
agree = np.mean([l == i for d in V for l, i in zip(lab[d["id"]], d["imgt"])])
print(f"global pairwise: pan P {s['all']['precision']:.3f} R {s['all']['recall']:.3f} claimed {s['all']['claimed']:.3f} | IMGT P {si['all']['precision']:.3f} R {si['all']['recall']:.3f}")
print(f"  within-type: pan P {s['within-type']['precision']:.3f} R {s['within-type']['recall']:.3f} | IMGT P {si['within-type']['precision']:.3f} R {si['within-type']['recall']:.3f}")
print(f"  cross-type : pan P {s['cross-type']['precision']:.3f} R {s['cross-type']['recall']:.3f} | IMGT P {si['cross-type']['precision']:.3f} R {si['cross-type']['recall']:.3f}")
st = local_precision(dd, cache, lab, cl)
print("local test, claimed anchors (within, cross):", summarize(st))
allc = {d["id"]: set(range(len(lab[d["id"]]))) for d in dd}
st_all = local_precision(dd, cache, lab, allc)
print("local test, every residue incl. positional (within, cross):", summarize(st_all))
print("insertion rates pan:", {k: round(v, 3) for k, v in ins.items()}, " IMGT:", {k: round(v, 3) for k, v in insi.items()})
print(f"V domains: labels identical to IMGT {agree:.3f}")
pickle.dump(dict(anchors=A, labels=lab, claims={k: sorted(v) for k, v in cl.items()}), open("final.pkl", "wb"))
json.dump(dict(anchors=A, failing=sorted(fail)), open("final_anchors.json", "w"))
