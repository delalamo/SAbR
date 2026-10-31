"""Leave-class-out: the reference is trained without the held-out class, then numbers it."""
import json, sys, time
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, "/tmp/sabr_unified/tools")
import glob, os
import ulib
NEGW = float(os.environ.get("NEGW", "0"))
REF = os.environ.get("REF", "/tmp/sabr_unified/ref_train.npz")

GROUPS = {"C2/I-set IgSF": ["C-LIKE:IgSF"], "V-like": ["V-LIKE"], "FN3": ["F:FN/TNC", "F:other"],
          "TRAC": ["C-ALPHA"], "gamma-delta V": ["V-GAMMA", "V-DELTA"], "MHC-II a2/b2": ["C-LIKE:MHC-II"]}


def run(item):
    g, cls = item
    doms, cache, _ = ulib.load_dataset()
    train = [d for d in doms if d["split"] == "train" and d["cls"] not in cls]
    test = [d for d in doms if d["cls"] in cls]
    negs = None
    if NEGW > 0:
        test_fams = set(str(x) for x in np.load(REF)["test_fams"])
        negs = [np.load(f)["emb"] for f in sorted(glob.glob("/tmp/sabr_unified/neg/*.npz"))
                if f.split("/")[-1].split("_")[0] not in test_fams]
    names, ref, info = ulib.train_profile(train, cache, ulib.anchor_list(), l2=1e-6, maxiter=800,
                                          negatives=negs, neg_weight=NEGW or 4.0)
    ref = ref * 3.0
    z = np.load(REF)
    full = ([str(x) for x in z["names"]], z["ref"])
    out = {}
    for tag, (nm, rf), subset in (("held_out", (names, ref), test),
                                  ("included", full, [d for d in test if d["split"] == "test"])):
        agg = dict(n=0, eq=0, gt=0, pred=0, tp=0, domains=len(subset))
        for d in subset:
            labels, sites, _ = ulib.number(ulib.dom_emb(d, cache), nm, rf)
            m = ulib.compare(d, sites, labels)
            agg["n"] += m["n"]; agg["eq"] += m["label_eq"]; agg["gt"] += m["gt_anchors"]
            agg["pred"] += m["pred_anchors"]; agg["tp"] += m["anchor_tp"]
        out[tag] = dict(domains=agg["domains"], anchor_precision=agg["tp"] / max(agg["pred"], 1),
                        anchor_recall=agg["tp"] / max(agg["gt"], 1), label_exact=agg["eq"] / max(agg["n"], 1))
    return g, out


if __name__ == "__main__":
    t0 = time.time()
    with Pool(3) as p:
        res = dict(p.map(run, GROUPS.items(), chunksize=1))
    for g, r in res.items():
        for tag in ("held_out", "included"):
            x = r[tag]
            print(f"{g:15s} {tag:9s} n={x['domains']:3d} anchor P {x['anchor_precision']:.3f} R {x['anchor_recall']:.3f} labels exact {x['label_exact']:.3f}")
    json.dump(res, open("/tmp/sabr_unified/exp_loco.json", "w"), indent=1)
    print(f"{time.time() - t0:.0f}s")
