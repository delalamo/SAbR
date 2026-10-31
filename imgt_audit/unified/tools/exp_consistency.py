"""Reproducibility: identical domain sequences in independent structures should get identical labels."""
import glob, json, os, sys
from collections import Counter, defaultdict
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, "/home/user/SAbR/src")
from sabr import unified

REF = sys.argv[1] if len(sys.argv) > 1 else "/tmp/sabr_unified/ref_train.npz"


def run(path):
    z = np.load(REF)
    names, ref = tuple(str(x) for x in z["names"] if True), z["ref"]
    unified.load_reference = lambda: (ref, names)
    c = np.load(path)
    E, seq = c["emb"], "".join(c["seq"])
    gaps = frozenset(int(x) for x in c["gaps"])
    out = []
    try:
        domains = unified.find_domains(E, gaps)
    except Exception:
        return out
    for anchors in domains:
        a, b = min(anchors), max(anchors) + 1
        local = {r - a: l for r, l in anchors.items()}
        labels = unified.fill_loops(b - a, local)
        out.append((os.path.basename(path), seq[a:b], labels))
    return out


if __name__ == "__main__":
    files = sorted(glob.glob("/tmp/sabr_unified/emb/*.npz"))
    with Pool(4) as p:
        res = [x for r in p.map(run, files, chunksize=8) for x in r]
    groups = defaultdict(list)
    for name, s, labels in res:
        groups[s].append((name, labels))
    multi = {s: v for s, v in groups.items() if len({n.split('_')[0] for n, _ in v}) >= 2}
    tot = same = 0
    whole = Counter()
    for s, v in multi.items():
        # compare every copy with the most common labelling of that sequence
        mode = Counter(tuple(l) for _, l in v).most_common(1)[0][0]
        for _, l in v:
            tot += len(l); same += sum(x == y for x, y in zip(l, mode))
            whole[tuple(l) == mode] += 1
    copies = sum(len(v) for v in multi.values())
    print(f"domains numbered {len(res)}; sequences seen in >=2 entries: {len(multi)} ({copies} copies)")
    print(f"residue labels identical to the consensus labelling: {same / max(tot, 1):.4f}")
    print(f"copies numbered identically end to end: {whole[True]}/{copies} ({whole[True] / max(copies, 1):.3f})")
    big = sorted(multi.items(), key=lambda kv: -len(kv[1]))[:5]
    for s, v in big:
        mode = Counter(tuple(l) for _, l in v).most_common(1)[0][0]
        print(f"  {len(v):3d} copies, len {len(s)}: identical {sum(tuple(l) == mode for _, l in v)}  starts {s[:12]}")
    json.dump(dict(sequences=len(multi), copies=copies, residue_identity=same / max(tot, 1),
                   identical_copies=whole[True] / max(copies, 1)), open("/tmp/sabr_unified/exp_consistency.json", "w"))
