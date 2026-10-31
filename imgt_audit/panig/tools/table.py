import pickle, sys
sys.path.insert(0, "/tmp/pan")
import numpy as np, pan
from collections import defaultdict
z = pickle.load(open("/tmp/pan/a12.pkl", "rb")); stats = z["stats"]
TYPES = ["V", "C1", "C2/I", "FN3", "V-like"]
REG = [("A", 1, 15), ("B", 16, 26), ("CDR1", 27, 38), ("C", 39, 46), ("C'", 47, 55), ("CDR2", 56, 65), ("C''", 66, 74), ("D", 75, 84), ("E", 85, 96), ("F", 97, 104), ("CDR3", 105, 117), ("G", 118, 128)]
reg = lambda n: next(r for r, a, b in REG if a <= n <= b)
rows = {}
for s, row in stats.items():
    occ = defaultdict(list); means = []; spreads = []
    for c, v in row.items():
        occ[pan.type_of(c)].append(v["occ"])
        if v["occ"] >= 0.5:
            means.append(np.array(v["mean"])); spreads.append(v["spread"])
    o = {t: (np.mean(occ[t]) if occ[t] else 0.0) for t in TYPES}
    dev = max((np.linalg.norm(a - b) for i, a in enumerate(means) for b in means[i + 1:]), default=0.0)
    rows[s] = dict(occ=o, spread=max(spreads, default=0), classdev=dev)
pickle.dump(rows, open("/tmp/pan/rows.pkl", "wb"))
print("site reg   " + " ".join(f"{t:>6}" for t in TYPES) + "  spread  classdev")
for s in sorted(rows, key=int):
    r = rows[s]
    print(f"{s:>4} {reg(int(s)):5s} " + " ".join(f"{r['occ'][t]:6.2f}" for t in TYPES) + f"  {r['spread']:6.2f}  {r['classdev']:6.2f}")
