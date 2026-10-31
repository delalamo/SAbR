"""Specificity: reference trained with non-Ig negatives from half the families, tested on the other half."""
import glob, json, sys, time
from collections import defaultdict
import numpy as np
sys.path.insert(0, "/tmp/sabr_unified/tools"); sys.path.insert(0, "/home/user/SAbR/src")
import ulib
from exp_main import evaluate
from sabr import unified

if __name__ == "__main__":
    t0 = time.time()
    doms, cache, _ = ulib.load_dataset()
    files = sorted(glob.glob("/tmp/sabr_unified/neg/*.npz"))
    fams = sorted({f.split("/")[-1].split("_")[0] for f in files})
    test_fams = set(fams[1::2])
    neg_train = [np.load(f)["emb"] for f in files if f.split("/")[-1].split("_")[0] not in test_fams]
    neg_test = [(f.split("/")[-1], np.load(f)["emb"]) for f in files if f.split("/")[-1].split("_")[0] in test_fams]
    print("held-out negative families:", sorted(test_fams), flush=True)
    train = [d for d in doms if d["split"] == "train"]; test = [d for d in doms if d["split"] == "test"]
    names, ref, info = ulib.train_profile(train, cache, ulib.anchor_list(), l2=1e-6, maxiter=2000, negatives=neg_train, neg_weight=1.0)
    ref = ref * 3.0
    np.savez("/tmp/sabr_unified/ref_train_neg1.npz", names=np.array(names), ref=ref, test_fams=np.array(sorted(test_fams)))
    print("trained", info, f"{time.time() - t0:.0f}s", flush=True)
    evaluate(test, cache, names, ref, "held-out Ig entries (with negatives)")
    unified.load_reference = lambda: (ref, tuple(names))
    pos = []
    for d in test:
        anc, score = unified.assign_anchors(ulib.dom_emb(d, cache))
        pos.append((score, len(anc)))
    neg = []
    for name, E in neg_test:
        anc, score = unified.assign_anchors(E)
        neg.append((name, score, len(anc)))
    ps = np.array([p[0] for p in pos]); ns = np.array([n[1] for n in neg])
    print("Ig domain scores pct 0.5/1/5/50:", np.percentile(ps, [0.5, 1, 5, 50]).round(1))
    print("non-Ig chain scores pct 50/95/99/max:", np.percentile(ns, [50, 95, 99, 100]).round(1))
    for thr in (100, 200, 300, 400, 500, 600):
        print(f"  threshold {thr}: Ig domains kept {np.mean(ps >= thr):.3f}; non-Ig chains passing {np.mean(ns >= thr):.3f}")
    print("worst non-Ig:", sorted(neg, key=lambda x: -x[1])[:6])
    json.dump(dict(pos=ps.tolist(), neg=[(n, float(s), a) for n, s, a in neg]), open("/tmp/sabr_unified/exp_neg1.json", "w"))
    print(f"{time.time() - t0:.0f}s")
