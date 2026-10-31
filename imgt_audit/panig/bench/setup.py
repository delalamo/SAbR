"""Train-only consensus frames, templates and oracle labels for the algorithm benchmark."""
import pickle, sys, random, time
sys.path.insert(0, "/tmp/pan"); sys.path.insert(0, "/tmp/sabr_unified/tools")
import numpy as np, pan, ulib, tmtools
from collections import defaultdict, Counter
t0 = time.time()
doms, cache, _ = ulib.load_dataset()
f2 = pickle.load(open("/tmp/pan/final2.pkl", "rb"))
A = sorted(f2["anchors"]); AS = set(A)
dd = [d for d in doms if d["id"] in f2["labels"]]
oracle = {d["id"]: {i: int(f2["labels"][d["id"]][i]) for i in f2["claims"][d["id"]]} for d in dd}
train = [d for d in dd if d["split"] == "train"]; test = [d for d in dd if d["split"] == "test"]
print("train", Counter(pan.type_of(d["cls"]) for d in train), "test", Counter(pan.type_of(d["cls"]) for d in test), flush=True)
frame = pan.build_v_frame(train, cache)
per_dom, _ = pan.site_table(train, cache, frame)
cfr = pan.class_frames(train, cache, frame, per_dom)
print("frames", len(frame.names), len(cfr), f"{time.time()-t0:.0f}s", flush=True)
# templates: per class, up to 5 train domains with the most oracle anchors, diverse PDB ids
by = defaultdict(list)
for d in train: by[d["cls"]].append(d)
templates = []
for c, L in by.items():
    L = sorted(L, key=lambda d: -len(oracle[d["id"]])); seen = set()
    for d in L:
        if d["pdb"] in seen: continue
        seen.add(d["pdb"]); templates.append(d["id"])
        if sum(1 for t in templates if t in {x["id"] for x in L}) >= 5: break
print("templates", len(templates), "classes", len(by))
pickle.dump(dict(A=A, oracle=oracle, train=[d["id"] for d in train], test=[d["id"] for d in test],
                 frame=(frame.names, frame.xyz, frame.cbu), cfr={c: (f.names, f.xyz, f.cbu) for c, f in cfr.items()},
                 templates=templates), open("/tmp/pan/bench/setup.pkl", "wb"))
print("done", f"{time.time()-t0:.0f}s")
