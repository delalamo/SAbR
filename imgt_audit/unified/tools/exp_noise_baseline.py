"""Self-consistency under coordinate noise: SAbR IMGT (H/K/L references) versus unified numbering."""
import sys, numpy as np
from collections import Counter
from multiprocessing import Pool
sys.path.insert(0, "/tmp/sabr_unified/tools"); sys.path.insert(0, "/home/user/SAbR/src")
import ulib
from sabr.model import encode
from sabr.alignment import align
from sabr import unified

G = {}


def noisy(c, sigma, seed):
    rng = np.random.default_rng(seed)
    bb = c["coords"][:, :3].astype(np.float64) + rng.normal(0, sigma, c["coords"][:, :3].shape)
    n, ca, cc = bb[:, 0], bb[:, 1], bb[:, 2]
    b, e = ca - n, cc - ca
    cb = -0.58273431 * np.cross(b, e) + 0.56802827 * b - 0.54067466 * e + ca
    return np.concatenate([bb, cb[:, None]], 1).astype(np.float32)


def run(i):
    d = G["doms"][i]
    c = G["cache"][(d["pdb"], d["chain"])]
    a, b = d["rows"]
    z = np.load("/tmp/sabr_unified/ref_train_neg1.npz")
    names, ref = tuple(str(x) for x in z["names"]), z["ref"]
    out = {}
    clean_imgt = clean_uni = None
    for sigma in (0.0, 0.25, 0.5):
        E = c["emb"] if sigma == 0 else encode(noisy(c, sigma, 7 + i))
        Ed = E[a:b]
        aln, sel, _ = align(Ed, frozenset(), "auto", 0.0)
        imgt = [int(np.argmax(r)) if r.any() else -1 for r in aln]
        lab, sites, _ = ulib.number(Ed, list(names), ref)
        if sigma == 0:
            clean_imgt, clean_uni = imgt, lab
        out[sigma] = (np.mean([x == y for x, y in zip(imgt, clean_imgt)]), np.mean([x == y for x, y in zip(lab, clean_uni)]))
    return out


if __name__ == "__main__":
    doms, cache, _ = ulib.load_dataset()
    test = [d for d in doms if d["split"] == "test" and d["cls"] in ("VH", "V-KAPPA", "V-LAMBDA")][:90]
    G.update(doms=test, cache=cache)
    with Pool(2) as p:
        res = p.map(run, range(len(test)))
    for sigma in (0.25, 0.5):
        print(f"sigma {sigma}: residues numbered as on the clean structure - SAbR IMGT {np.mean([r[sigma][0] for r in res]):.3f}, "
              f"unified {np.mean([r[sigma][1] for r in res]):.3f}  (n={len(res)} V domains)")
