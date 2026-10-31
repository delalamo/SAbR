import sys, json, time
sys.path.insert(0, "/tmp/sabr_unified/tools")
import numpy as np, ulib
from exp_main import evaluate
t0 = time.time()
doms, cache, _ = ulib.load_dataset()
train = [d for d in doms if d["split"] == "train"]; test = [d for d in doms if d["split"] == "test"]
anchors = ulib.anchor_list()
for l2 in (1e-5, 1e-6):
    names, ref, info = ulib.train_profile(train, cache, anchors, l2=l2, maxiter=1500)
    np.savez(f"/tmp/sabr_unified/ref_l2_{l2}.npz", names=np.array(names), ref=ref)
    print("l2", l2, info, f"{time.time()-t0:.0f}s", flush=True)
    for scale in (1.0, 2.0, 4.0, 8.0):
        evaluate(test, cache, names, ref * scale, f"disc l2={l2} scale={scale}")
