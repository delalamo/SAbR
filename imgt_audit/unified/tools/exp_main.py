"""Held-out recovery of the structure-consistent numbering by a single unified embedding profile."""
import json, sys, time
from collections import Counter, defaultdict
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, "/tmp/sabr_unified/tools")
import ulib

G = {}


def _run(i):
    d = G["doms"][i]
    E = ulib.dom_emb(d, G["cache"])
    labels, sites, score = ulib.number(E, G["names"], G["prof"])
    return i, ulib.compare(d, sites, labels), labels, score


def evaluate(doms, cache, names, prof, tag, procs=4):
    G.update(doms=doms, cache=cache, names=names, prof=prof)
    with Pool(procs) as p:
        res = p.map(_run, range(len(doms)), chunksize=4)
    agg = defaultdict(Counter)
    for i, m, labels, score in res:
        for k in ("ALL", doms[i]["cls"]):
            agg[k].update(m); agg[k]["domains"] += 1
    rows = {}
    for k, a in agg.items():
        rows[k] = dict(domains=a["domains"], anchor_precision=a["anchor_tp"] / max(a["pred_anchors"], 1),
                       anchor_recall=a["anchor_tp"] / max(a["gt_anchors"], 1), label_exact=a["label_eq"] / max(a["n"], 1))
    print(f"== {tag}")
    for k in ["ALL"] + sorted(x for x in rows if x != "ALL"):
        r = rows[k]
        print(f"  {k:18s} n={r['domains']:4d} anchor P {r['anchor_precision']:.3f} R {r['anchor_recall']:.3f} | labels exact {r['label_exact']:.3f}")
    return rows, res


if __name__ == "__main__":
    t0 = time.time()
    doms, cache, miss = ulib.load_dataset()
    print("domains", len(doms), "missing", dict(miss), Counter(d["split"] for d in doms))
    anchors = ulib.anchor_list()
    train = [d for d in doms if d["split"] == "train"]
    test = [d for d in doms if d["split"] == "test"]
    variants = {"balanced+rescaled": dict(balance=True, rescale=True),
                "balanced": dict(balance=True, rescale=False),
                "pooled+rescaled": dict(balance=False, rescale=True)}
    out = {}
    for tag, kw in variants.items():
        names, prof = ulib.build_profile(train, cache, anchors, **kw)
        print(tag, "profile columns", len(names))
        rows, _ = evaluate(test, cache, names, prof, tag)
        out[tag] = rows
    json.dump(out, open("/tmp/sabr_unified/exp_main.json", "w"), indent=1)
    print(f"{time.time() - t0:.0f}s")
