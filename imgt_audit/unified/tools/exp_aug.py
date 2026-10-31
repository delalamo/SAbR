"""Noise-augmented reference: clean held-out accuracy and accuracy/self-consistency under coordinate noise."""
import glob, json, os, sys, time
from collections import Counter, defaultdict
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, "/tmp/sabr_unified/tools"); sys.path.insert(0, "/home/user/SAbR/src")
import ulib
from exp_main import evaluate
from exp_noise_baseline import noisy
from sabr.model import encode
from sabr.alignment import align

G = {}
REFS = {"clean-trained": "/tmp/sabr_unified/ref_train_neg1.npz", "noise-augmented": "/tmp/sabr_unified/ref_train_aug.npz"}


def run(i):
    d = G["doms"][i]
    c = G["cache"][(d["pdb"], d["chain"])]
    a, b = d["rows"]
    refs = {k: np.load(v) for k, v in REFS.items()}
    out = defaultdict(dict)
    clean = {}
    for sigma in (0.0, 0.25, 0.5):
        E = (c["emb"] if sigma == 0 else encode(noisy(c, sigma, 7 + i)))[a:b]
        if d["domtype"] == "V":
            aln, _, _ = align(E, frozenset(), "auto", 0.0)
            imgt = [int(np.argmax(r)) if r.any() else -1 for r in aln]
            clean.setdefault("imgt", imgt)
            out["IMGT (SAbR)"][sigma] = (None, np.mean([x == y for x, y in zip(imgt, clean["imgt"])]))
        for k, z in refs.items():
            lab, sites, _ = ulib.number(E, [str(x) for x in z["names"]], z["ref"])
            clean.setdefault(k, lab)
            acc = np.mean([p == t for p, t in zip(lab, d["target"])])
            out[k][sigma] = (acc, np.mean([x == y for x, y in zip(lab, clean[k])]))
    return d["domtype"], dict(out)


if __name__ == "__main__":
    t0 = time.time()
    doms, cache, _ = ulib.load_dataset()
    train = [d for d in doms if d["split"] == "train"]; test = [d for d in doms if d["split"] == "test"]
    tf = set(str(x) for x in np.load(REFS["clean-trained"])["test_fams"])
    negs = [np.load(f)["emb"] for f in sorted(glob.glob("/tmp/sabr_unified/neg/*.npz")) if f.split("/")[-1].split("_")[0] not in tf]
    if not os.path.exists(REFS["noise-augmented"]):
        names, ref, info = ulib.train_profile(train, cache, ulib.anchor_list(), l2=1e-6, maxiter=2000,
                                              negatives=negs, neg_weight=1.0, augment=True)
        np.savez(REFS["noise-augmented"], names=np.array(names), ref=ref * 3.0, test_fams=np.array(sorted(tf)))
        print("trained", info, f"{time.time() - t0:.0f}s", flush=True)
    z = np.load(REFS["noise-augmented"])
    evaluate(test, cache, [str(x) for x in z["names"]], z["ref"], "noise-augmented reference, clean held-out")
    rng = np.random.default_rng(0)
    by = defaultdict(list)
    for d in test:
        by[d["cls"]].append(d)
    subset = [d for c in sorted(by) for d in by[c][:8]][:150]
    G.update(doms=subset, cache=cache)
    with Pool(4) as p:
        res = p.map(run, range(len(subset)), chunksize=2)
    print(f"== noise robustness on {len(subset)} held-out domains (all classes; IMGT on the V subset)")
    summary = {}
    for k in ("IMGT (SAbR)", "clean-trained", "noise-augmented"):
        for sigma in (0.0, 0.25, 0.5):
            vals = [r[k][sigma] for t, r in res if k in r]
            acc = [v[0] for v in vals if v[0] is not None]
            self_ = [v[1] for v in vals]
            summary[f"{k}|{sigma}"] = dict(accuracy=float(np.mean(acc)) if acc else None, self_consistency=float(np.mean(self_)))
            print(f"  {k:16s} sigma {sigma:4.2f}: labels exact vs target {np.mean(acc) if acc else float('nan'):.3f} | "
                  f"same as clean-structure numbering {np.mean(self_):.3f} (n={len(vals)})")
    json.dump(summary, open("/tmp/sabr_unified/exp_aug.json", "w"), indent=1)
    print(f"{time.time() - t0:.0f}s")
