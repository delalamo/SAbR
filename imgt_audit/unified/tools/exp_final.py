"""Final held-out evaluation of the unified reference (fixed hyperparameters: l2=1e-6, scale=3)."""
import json, random, sys, time
from collections import Counter, defaultdict
from multiprocessing import Pool
import numpy as np
import tmtools
sys.path.insert(0, "/tmp/sabr_unified/tools")
sys.path.insert(0, "/home/user/SAbR/src")
import ulib
from exp_main import evaluate

L2, SCALE = 1e-6, 3.0
G = {}


def region(label):
    n = ulib.lab_key(label)[0]
    for name, lo, hi in (("FR-A/A'", 1, 15), ("B", 16, 26), ("BC/CDR1", 27, 38), ("C/C'", 39, 55), ("C'C''/CDR2", 56, 65),
                         ("C''/D", 66, 84), ("E/F", 85, 104), ("FG/CDR3", 105, 117), ("G", 118, 128)):
        if lo <= n <= hi:
            return name
    return "termini"


def _chain(key):
    """Whole-chain domain detection versus the domains annotated in this chain."""
    import importlib
    from sabr import unified
    c = G["cache"][key]
    E = c["emb"]
    gaps = frozenset(int(x) for x in c["gaps"])
    domains = G["predict"](E, gaps)
    out = []
    for d in G["by_chain"][key]:
        a, b = d["rows"]
        best = max(domains, key=lambda m: len([r for r in m if a <= r < b]), default=None)
        hit = best is not None and len([r for r in best if a <= r < b]) >= 0.5 * len(d["gt_sites"])
        if not hit:
            out.append((d["id"], d["cls"], False, 0, 0, len(d["target"]))); continue
        local = {r - a: l for r, l in best.items() if a <= r < b}
        labels = ulib.fill_loops(b - a, local)
        eq = sum(labels[j] == t for j, t in enumerate(d["target"]))
        tp = sum(1 for j, l in local.items() if d["gt_sites"].get(j) == l)
        out.append((d["id"], d["cls"], True, eq, tp, len(d["target"])))
    return out, len(domains), len(G["by_chain"][key])


def _pair(args):
    a, b = args
    r = tmtools.tm_align(a["ca"], b["ca"], a["seq"], b["seq"])
    if max(r.tm_norm_chain1, r.tm_norm_chain2) < 0.5:
        return None
    Xa = a["ca"] @ r.u.T + r.t
    eq, i, j = {}, 0, 0
    for x, y in zip(r.seqxA, r.seqyA):
        if x != "-" and y != "-" and np.linalg.norm(Xa[i] - b["ca"][j]) <= 3.0:
            eq[i] = j
        i += x != "-"; j += y != "-"
    res = {}
    for key in ("imgt", "target", "pred"):
        la, lb = a[key], b[key]
        pos = {l: k for k, l in enumerate(lb)}
        same = [(i, pos[l]) for i, l in enumerate(la) if l in pos]
        res[key] = (len(eq), sum(la[i] == lb[j] for i, j in eq.items()), len(same),
                    sum(np.linalg.norm(Xa[i] - b["ca"][j]) <= 3.0 for i, j in same))
    return res


if __name__ == "__main__":
    t0 = time.time()
    doms, cache, miss = ulib.load_dataset()
    print("domains", len(doms), dict(miss), Counter(d["split"] for d in doms), flush=True)
    anchors = ulib.anchor_list()
    train = [d for d in doms if d["split"] == "train"]
    test = [d for d in doms if d["split"] == "test"]
    names, ref, info = ulib.train_profile(train, cache, anchors, l2=L2, maxiter=2000)
    ref = ref * SCALE
    np.savez("/tmp/sabr_unified/ref_train.npz", names=np.array(names), ref=ref)
    print("trained", info, len(names), f"{time.time() - t0:.0f}s", flush=True)
    rows, res = evaluate(test, cache, names, ref, "held-out entries")
    # error analysis by region of the target label
    reg = defaultdict(Counter)
    preds = {}
    for i, m, labels, score in res:
        d = test[i]; preds[d["id"]] = labels
        for t, p in zip(d["target"], labels):
            reg[region(t)]["n"] += 1; reg[region(t)]["eq"] += (t == p)
    print("== exact-label agreement by region (held-out)")
    for k, v in sorted(reg.items()):
        print(f"  {k:12s} {v['eq'] / v['n']:.3f} (n={v['n']})")
    # whole-chain multi-domain detection with the shipped algorithm on the held-out chains
    from sabr import unified
    unified.load_reference.cache_clear()
    def predict(E, gaps, _names=names, _ref=ref):
        unified.load_reference = lambda: (_ref, tuple(_names))
        return unified.find_domains(E, gaps)
    by_chain = defaultdict(list)
    for d in test:
        by_chain[(d["pdb"], d["chain"])].append(d)
    G.update(cache=cache, by_chain=by_chain, predict=predict)
    with Pool(4) as p:
        ch = p.map(_chain, list(by_chain), chunksize=2)
    found = sum(x[2] for c in ch for x in c[0]); total = sum(len(c[0]) for c in ch)
    eq = sum(x[3] for c in ch for x in c[0] if x[2]); n = sum(x[5] for c in ch for x in c[0] if x[2])
    extra = sum(max(0, c[1] - c[2]) for c in ch)
    print(f"== whole-chain detection: {found}/{total} annotated domains found; {extra} extra domains in {len(ch)} chains; "
          f"labels exact on found domains {eq / max(n, 1):.3f}")
    missed = Counter(x[1] for c in ch for x in c[0] if not x[2])
    print("   missed by class:", dict(missed))
    # structural consistency of the predicted numbering (pairwise TM-align, 3 A)
    rnd = random.Random(0)
    slim = []
    for d in test:
        c = cache[(d["pdb"], d["chain"])]; a, b = d["rows"]
        slim.append(dict(ca=c["coords"][a:b, 1].astype(np.float64), seq="".join(c["seq"][a:b]), imgt=d["imgt"],
                         target=d["target"], pred=preds[d["id"]], cls=d["cls"]))
    jobs = [(x, y) for x in slim for y in rnd.sample(slim, 15) if y is not x]
    with Pool(4) as p:
        pr = [r for r in p.map(_pair, jobs, chunksize=10) if r]
    print("== structural consistency (pairs with TM>=0.5):", len(pr))
    cons = {}
    for key in ("imgt", "target", "pred"):
        v = np.sum([r[key] for r in pr], axis=0)
        cons[key] = dict(recall=v[1] / v[0], precision=v[3] / v[2])
        print(f"  {key:7s} recall {v[1] / v[0]:.3f}  precision {v[3] / v[2]:.3f}")
    json.dump(dict(info=info, heldout=rows, region={k: v['eq'] / v['n'] for k, v in reg.items()},
                   chain=dict(found=found, total=total, extra=extra, exact=eq / max(n, 1), missed=dict(missed)),
                   consistency=cons), open("/tmp/sabr_unified/exp_final.json", "w"), indent=1)
    print(f"{time.time() - t0:.0f}s")
