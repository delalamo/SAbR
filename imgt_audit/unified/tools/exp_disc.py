import sys, json, time
sys.path.insert(0, "/tmp/sabr_unified/tools")
import numpy as np, ulib
from exp_main import evaluate
t0 = time.time()
doms, cache, _ = ulib.load_dataset()
train = [d for d in doms if d["split"] == "train"]; test = [d for d in doms if d["split"] == "test"]
anchors = ulib.anchor_list()
for l2 in (1e-4, 1e-3):
    names, ref, info = ulib.train_profile(train, cache, anchors, l2=l2)
    print("l2", l2, info, f"{time.time()-t0:.0f}s", flush=True)
    for scale in (0.5, 1.0, 2.0):
        evaluate(test, cache, names, ref * scale, f"disc l2={l2} scale={scale}")
