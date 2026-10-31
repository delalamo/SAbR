import pickle, sys
sys.path.insert(0, "/tmp/pan")
import numpy as np, pan, ulib
from collections import defaultdict
from score import label_all
from number2 import Scheme, CDR_INTERIOR
from local_test import local_precision, summarize
z = pickle.load(open("a12.pkl", "rb")); s2 = pickle.load(open("sweep2.pkl", "rb")); f2 = pickle.load(open("final2.pkl", "rb"))
doms, cache, _ = ulib.load_dataset(); dd = [d for d in doms if d["id"] in z["per_dom"]]
frame = pan.Frame(*z["frame"]); cfr = {c: pan.Frame(*v) for c, v in s2["cfr"].items()}
stats = z["stats"]
GROUPS = {"TRAC": ["C-ALPHA"], "FN3": ["F:FN/TNC", "F:other"], "C2/I-set": ["C-LIKE:IgSF"], "V-like": ["V-LIKE"], "gamma-delta V": ["V-GAMMA", "V-DELTA"]}

def rows_without(excl):
    rows = {}
    for s, row in stats.items():
        occ = defaultdict(list); spreads = []
        for c, v in row.items():
            if c in excl: continue
            occ[pan.type_of(c)].append(v["occ"])
            if v["occ"] >= 0.5: spreads.append(v["spread"])
        rows[s] = dict(occ={t: (np.mean(occ[t]) if occ[t] else 0.0) for t in ("V", "C1", "C2/I", "FN3", "V-like")}, spread=max(spreads, default=0))
    return rows

for g, excl in GROUPS.items():
    train = [d for d in dd if d["cls"] not in excl]; test = [d for d in dd if d["cls"] in excl]
    rows = rows_without(set(excl))
    A0 = {int(s) for s, r in rows.items() if int(s) not in CDR_INTERIOR and max(r["occ"][t] for t in ("V", "C1", "C2/I")) >= 0.5 and r["spread"] <= 1.9} | {26}
    per_tr = {d["id"]: s2["variants"][("class", 3.5)][d["id"]] for d in train if d["id"] in s2["variants"][("class", 3.5)]}
    sc0 = Scheme(A0); lab0, cl0 = label_all([d for d in train if d["id"] in per_tr], per_tr, sc0)
    st = local_precision([d for d in train if d["id"] in per_tr], cache, lab0, cl0)
    fail = {int(k) for k, v in st.items() if v[1] / max(v[0], 1) < 0.9 or (v[2] >= 200 and v[3] / v[2] < 0.8)}
    A = sorted(A0 - fail - {1, 2} if False else A0 - fail)
    sc = Scheme(A)
    per_te = pan.site_matches(test, cache, frame, None, 3.5)       # pooled model only: the class is unseen
    lab, cl = label_all(test, per_te, sc)
    agree = np.mean([a == b for d in test for a, b in zip(lab[d["id"]], f2["labels"][d["id"]])])
    # local precision of the unseen class against all other domains
    allab = dict(f2["labels"]); allab.update(lab)
    allcl = {k: set(v) for k, v in f2["claims"].items()}; allcl.update({k: set(v) for k, v in cl.items()})
    stx = local_precision(dd, cache, allab, allcl)
    tl = {l for d in test for l in lab[d["id"]]}
    st_full = local_precision(dd, cache, f2["labels"], {k: set(v) for k, v in f2["claims"].items()})
    print(f"{g:14s} n={len(test):3d} anchors w/o class {len(A)} (full 82, shared {len(set(A) & set(f2['anchors']))}) | labels = full-scheme labels {agree:.3f} | "
          f"overall local precision with unseen-class labels {summarize(stx)[0]:.3f}/{summarize(stx)[1]:.3f} vs full {summarize(st_full)[0]:.3f}/{summarize(st_full)[1]:.3f}", flush=True)
