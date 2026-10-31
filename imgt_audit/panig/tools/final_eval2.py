import json, pickle, sys, glob, os
sys.path.insert(0, "/tmp/pan")
import numpy as np, pan, ulib
from collections import Counter, defaultdict
from score import anchor_set, label_all, score, insertion_rate
from number2 import Scheme
from local_test import local_precision, summarize
z = pickle.load(open("a12.pkl", "rb")); rows = pickle.load(open("rows.pkl", "rb")); s2 = pickle.load(open("sweep2.pkl", "rb"))
f = pickle.load(open("final.pkl", "rb"))
A = sorted(set(f["anchors"]) | {26}); sc = Scheme(A)
doms, cache, _ = ulib.load_dataset(); per = s2["variants"][("class", 3.5)]; dd = [d for d in doms if d["id"] in per]
frame = pan.Frame(*z["frame"]); cfr = {c: pan.Frame(*v) for c, v in s2["cfr"].items()}
lab, cl = label_all(dd, per, sc)
out = {"anchors": A}
# 1) metrics by type
s = score(dd, lab, cl); imgt = {d["id"]: d["imgt"] for d in dd}; si = score(dd, imgt)
ins = insertion_rate(dd, lab); insi = insertion_rate(dd, imgt)
st = local_precision(dd, cache, lab, cl); st_all = local_precision(dd, cache, lab, {d["id"]: set(range(len(lab[d["id"]]))) for d in dd})
sti = local_precision(dd, cache, imgt, {d["id"]: set(range(len(d["imgt"]))) for d in dd})
print(f"anchors {len(A)}")
print(f"global: pan P {s['all']['precision']:.3f} R {s['all']['recall']:.3f} | IMGT P {si['all']['precision']:.3f} R {si['all']['recall']:.3f}")
print("local anchors:", [round(x, 3) for x in summarize(st)[:2]], " local all residues:", [round(x, 3) for x in summarize(st_all)[:2]], " IMGT local:", [round(x, 3) for x in summarize(sti)[:2]])
print("type-pair global precision pan vs IMGT:")
for k in sorted(k for k in s if "~" in k):
    if k in si: print(f"   {k:14s} pan {s[k]['precision']:.3f}/{s[k]['recall']:.3f}  IMGT {si[k]['precision']:.3f}/{si[k]['recall']:.3f}")
print("insertions pan:", {k: round(v, 3) for k, v in ins.items()}, "IMGT:", {k: round(v, 3) for k, v in insi.items()})
V = [d for d in dd if pan.type_of(d["cls"]) == "V"]
print("V labels identical to IMGT:", round(np.mean([l == i for d in V for l, i in zip(lab[d["id"]], d["imgt"])]), 3))
out.update(global_pan=s["all"], global_imgt=si["all"], local_anchor=summarize(st)[:2], local_all=summarize(st_all)[:2], local_imgt=summarize(sti)[:2],
           ins_pan=ins, ins_imgt=insi)
# 2) reproducibility: identical domain sequences in other cached chains
seqs = {}
for d in dd:
    c = cache[(d["pdb"], d["chain"])]; a, b = d["rows"]; seqs[d["id"]] = "".join(c["seq"][a:b])
chains = {}
for p in glob.glob("/tmp/sabr_unified/emb/*.npz"):
    k = tuple(os.path.basename(p)[:-4].split("_", 1)); chains[k] = None
copies = defaultdict(list)
for d in dd:
    if pan.type_of(d["cls"]) == "V": continue
    s0 = seqs[d["id"]]
    for k in chains:
        c = cache.get(k)
        if c is None:
            c = cache[k] = dict(np.load(f"/tmp/sabr_unified/emb/{k[0]}_{k[1]}.npz"))
        full = "".join(c["seq"]); i = full.find(s0)
        if i >= 0:
            copies[d["id"]].append((k, i))
rep_tot = rep_same = ident = ncopy = 0
for did, hits in copies.items():
    if len({h[0][0] for h in hits}) < 2: continue
    d = next(x for x in dd if x["id"] == did)
    labs = []
    for k, i in hits:
        dc = dict(d, pdb=k[0], chain=k[1], rows=(i, i + len(d["target"])), id=f"{k[0]}_{k[1]}_{i}")
        ca, cbu = pan.geometry(dc, cache)
        fr = cfr.get(d["cls"], frame)
        Rt = pan.superpose_on_core(dc, ca, fr) or pan.superpose_on_core(dc, ca, frame)
        m, R, t = pan.match(ca, cbu, fr, *Rt, refine_on=set(pan.CORE), dmax=3.5)
        site_of = {i2: int(sn) for i2, (sn, dist) in m.items() if int(sn) in set(A)}
        labs.append(tuple(sc.number(len(d["target"]), site_of)))
    mode = Counter(labs).most_common(1)[0][0]
    for l in labs:
        ncopy += 1; ident += l == mode
        rep_tot += len(l); rep_same += sum(x == y for x, y in zip(l, mode))
print(f"reproducibility: {sum(1 for h in copies.values() if len({x[0][0] for x in h}) >= 2)} sequences, {ncopy} copies: residue labels identical {rep_same / rep_tot:.4f}; copies identical end to end {ident / ncopy:.3f}")
out.update(repro=dict(copies=ncopy, residue=rep_same / rep_tot, identical=ident / ncopy))
json.dump(out, open("final_eval2.json", "w"), indent=1, default=float)
pickle.dump(dict(anchors=A, labels=lab, claims={k: sorted(v) for k, v in cl.items()}), open("final2.pkl", "wb"))
