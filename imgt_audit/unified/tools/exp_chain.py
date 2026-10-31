"""Whole-chain domain detection on held-out chains with the shipped find_domains."""
import sys, json
sys.path.insert(0, "/tmp/sabr_unified/tools"); sys.path.insert(0, "/home/user/SAbR/src")
from collections import Counter, defaultdict
from multiprocessing import Pool
import numpy as np, ulib
import exp_final
from sabr import unified

if __name__ == "__main__":
    doms, cache, _ = ulib.load_dataset()
    import os
    z = np.load(os.environ.get("REF", "ref_train.npz")); names = [str(x) for x in z["names"]]; ref = z["ref"]
    def predict(E, gaps):
        unified.load_reference = lambda: (ref, tuple(names))
        return unified.find_domains(E, gaps)
    test = [d for d in doms if d["split"] == "test"]
    by = defaultdict(list)
    for d in test: by[(d["pdb"], d["chain"])].append(d)
    exp_final.G.update(cache=cache, by_chain=by, predict=predict)
    with Pool(4) as p:
        ch = p.map(exp_final._chain, list(by), chunksize=2)
    found = sum(x[2] for c in ch for x in c[0]); total = sum(len(c[0]) for c in ch)
    eq = sum(x[3] for c in ch for x in c[0] if x[2]); n = sum(x[5] for c in ch for x in c[0] if x[2])
    print(f"whole-chain: {found}/{total} annotated domains found; labels exact on found {eq / max(n, 1):.3f}; "
          f"domains reported per chain {np.mean([c[1] for c in ch]):.2f} vs annotated {np.mean([c[2] for c in ch]):.2f}")
    print("missed:", dict(Counter(x[1] for c in ch for x in c[0] if not x[2])))
    json.dump(dict(found=found, total=total, exact=eq / max(n, 1)), open("exp_chain.json", "w"))
