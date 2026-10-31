"""SAbR-style embedding profile trained on pan-Ig anchor labels (train split only)."""
import pickle, sys
sys.path.insert(0, "/tmp/pan"); sys.path.insert(0, "/tmp/sabr_unified/tools")
import numpy as np, ulib
S = pickle.load(open("/tmp/pan/bench/setup.pkl", "rb"))
doms, cache, _ = ulib.load_dataset()
tr = set(S["train"])
train = [dict(d, gt_sites={i: str(a) for i, a in S["oracle"][d["id"]].items()}) for d in doms if d["id"] in tr]
names, ref, info = ulib.train_profile(train, cache, [str(a) for a in S["A"]], l2=1e-6, maxiter=2000)
print(info, len(names))
np.savez("/tmp/pan/bench/emb_ref.npz", names=np.array(names), ref=ref)
