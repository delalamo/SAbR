import json, pickle, sys, time
sys.path.insert(0, "/tmp/pan")
import numpy as np, pan, ulib
from collections import Counter
t0 = time.time()
doms, cache, _ = ulib.load_dataset()
print("domains", len(doms), Counter(pan.type_of(d["cls"]) for d in doms))
frame = pan.build_v_frame(doms, cache)
print("V frame sites", len(frame.names), f"{time.time()-t0:.0f}s")
per_dom, stats = pan.site_table(doms, cache, frame)
print("matched domains", len(per_dom), f"{time.time()-t0:.0f}s")
pickle.dump(dict(frame=(frame.names, frame.xyz, frame.cbu), per_dom=per_dom, stats=stats), open("/tmp/pan/a12.pkl", "wb"))
