import sys, glob, numpy as np
sys.path.insert(0, "/tmp/sabr_unified/tools"); sys.path.insert(0, "/home/user/SAbR/src")
import ulib
from exp_main import evaluate
from sabr import unified
doms, cache, _ = ulib.load_dataset()
test = [d for d in doms if d["split"] == "test"]
z = np.load("ref_train_neg.npz"); names = [str(x) for x in z["names"]]; base = z["ref"] / 3.0
test_fams = set(str(x) for x in z["test_fams"])
neg = [np.load(f)["emb"] for f in sorted(glob.glob("neg/*.npz")) if f.split("/")[-1].split("_")[0] in test_fams]
for scale in (4.0, 6.0, 9.0):
    ref = base * scale
    rows, _ = evaluate(test, cache, names, ref, f"scale {scale}")
    unified.load_reference = lambda r=ref: (r, tuple(names))
    ps = np.array([unified.assign_anchors(ulib.dom_emb(d, cache))[1] for d in test])
    ns = np.array([unified.assign_anchors(E)[1] for E in neg])
    for thr in (300, 400, 600, 800):
        print(f"   thr {thr * scale / 3:.0f} (=~{thr} at scale 3): Ig kept {np.mean(ps >= thr * scale / 3):.3f} non-Ig passing {np.mean(ns >= thr * scale / 3):.3f}")
