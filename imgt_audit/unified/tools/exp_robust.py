"""Robustness of the unified numbering: unseen classes, coordinate noise, missing residues."""
import json, sys, time
from collections import Counter, defaultdict
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, "/tmp/sabr_unified/tools")
sys.path.insert(0, "/home/user/SAbR/src")
import ulib
from sabr.model import encode

G = {}


def _noise(args):
    i, sigma, seed = args
    d = G["doms"][i]
    c = G["cache"][(d["pdb"], d["chain"])]
    rng = np.random.default_rng(seed)
    X = c["coords"].astype(np.float64)
    # perturb N, CA, C and rebuild the synthetic CB as SAbR does
    bb = X[:, :3] + rng.normal(0, sigma, X[:, :3].shape)
    n, ca, cc = bb[:, 0], bb[:, 1], bb[:, 2]
    b, e = ca - n, cc - ca
    cb = -0.58273431 * np.cross(b, e) + 0.56802827 * b - 0.54067466 * e + ca
    E = encode(np.concatenate([bb, cb[:, None]], 1).astype(np.float32))
    a, z = d["rows"]
    labels, sites, _ = ulib.number(E[a:z], G["names"], G["prof"])
    return i, sigma, ulib.compare(d, sites, labels)


def _trunc(args):
    i, mode = args
    d = G["doms"][i]
    E = ulib.dom_emb(d, G["cache"])
    n = len(E)
    if mode == "termini5":
        keep = list(range(5, n - 5))
    elif mode == "loopgap":
        # remove 3 residues from the middle of every target loop longer than 4 (simulated missing density)
        sites = sorted(d["gt_sites"])
        drop = set()
        for a, b in zip(sites, sites[1:]):
            if b - a - 1 >= 5:
                mid = (a + b) // 2
                drop |= {mid - 1, mid, mid + 1}
        keep = [k for k in range(n) if k not in drop]
    labels, sites, _ = ulib.number(E[keep], G["names"], G["prof"])
    # compare anchors on retained residues (loop labels legitimately change when residues vanish)
    gt = {keep.index(k): a for k, a in d["gt_sites"].items() if k in keep}
    tp = sum(1 for k, a in sites.items() if gt.get(k) == a)
    return i, mode, dict(gt_anchors=len(gt), pred_anchors=len(sites), anchor_tp=tp)


def summarize(res, key):
    agg = defaultdict(Counter)
    for i, k, m in res:
        agg[k].update(m)
    for k, a in sorted(agg.items(), key=lambda kv: str(kv[0])):
        p = a["anchor_tp"] / max(a["pred_anchors"], 1); r = a["anchor_tp"] / max(a["gt_anchors"], 1)
        le = f" | labels exact {a['label_eq'] / a['n']:.3f}" if a.get("n") else ""
        print(f"  {key}={k}: anchor P {p:.3f} R {r:.3f}{le}")
    return {str(k): dict(a) for k, a in agg.items()}


if __name__ == "__main__":
    t0 = time.time()
    doms, cache, _ = ulib.load_dataset()
    anchors = ulib.anchor_list()
    out = {}
    from exp_main import evaluate
    # 2) coordinate noise and missing residues on the held-out split
    train = [d for d in doms if d["split"] == "train"]
    test = [d for d in doms if d["split"] == "test"]
    import os
    z = np.load(os.environ.get("REF", "/tmp/sabr_unified/ref_train.npz"))
    names, prof = [str(x) for x in z["names"]], z["ref"]
    G.update(doms=test, cache=cache, names=names, prof=prof)
    jobs = [(i, s, 1000 + i) for i in range(len(test)) for s in (0.0, 0.1, 0.25, 0.5, 1.0)]
    with Pool(4) as p:
        res = p.map(_noise, jobs, chunksize=8)
    print("== coordinate noise (re-encoded)")
    out["noise"] = summarize(res, "sigma")
    jobs = [(i, m) for i in range(len(test)) for m in ("termini5", "loopgap")]
    with Pool(4) as p:
        res = p.map(_trunc, jobs, chunksize=8)
    print("== missing residues")
    out["missing"] = summarize(res, "mode")
    json.dump(out, open(os.environ.get("OUT", "/tmp/sabr_unified/exp_robust.json"), "w"), indent=1)
    print(f"{time.time() - t0:.0f}s")
