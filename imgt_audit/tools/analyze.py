"""Pairwise TM-align of IMGT-numbered domains; compare structural equivalences with IMGT labels."""
import itertools, json, os, re, sys
from collections import Counter, defaultdict
import numpy as np
import tmtools

ROOT = os.path.join(os.path.dirname(__file__), "..")
CUTOFF = 3.0      # Å, CA-CA after TM superposition
MIN_TM = 0.5


MHC1 = {"HLA-A", "HLA-B", "MH1-K1b", "MH1-A", "H2-T22", "CD1D", "CD1B", "MR1", "FCGRT", "HFE", "MICA", "AZGP1"}


def subclass(d):
    if d["label"] == "F-DOMAIN":
        return "F:FN/TNC" if d["gene"].split("*")[0] in ("FN1", "TNC") else "F:cytokine-R"
    if d["label"] != "C-LIKE":
        return d["label"]
    g = d["gene"].split("*")[0]
    if g == "B2M": return "C-LIKE:B2M"
    if g in MHC1: return "C-LIKE:MHC-I-a3"
    if g in ("HLA-DRA", "HLA-DMA"): return "C-LIKE:MHC-II-a2"
    if g in ("HLA-DRB1", "HLA-DMB"): return "C-LIKE:MHC-II-b2"
    return "C-LIKE:IgSF-C2"


def load(min_frac=0.85):
    doms = []
    for d in json.load(open(os.path.join(ROOT, "domains.json"))):
        if len(d["residues"]) < min_frac * d["n_seq"]:
            continue
        d["id"] = f'{d["pdb"]}_{d["chain"]}_{d["label"]}'
        d["ca"] = np.array([r["atoms"]["CA"] for r in d["residues"]], float)
        d["seq"] = "".join(r["aa"] if r["aa"] != "X" else "G" for r in d["residues"])
        d["lab"] = [r["imgt"] for r in d["residues"]]
        d["cls"] = subclass(d)
        doms.append(d)
    # one domain per (pdb, label, sequence): drop identical copies in the same entry
    seen, out, ids = set(), [], Counter()
    for d in doms:
        k = (d["pdb"], d["label"], d["seq"])
        if k not in seen:
            seen.add(k); ids[d["id"]] += 1
            if ids[d["id"]] > 1:
                d["id"] += f"#{ids[d['id']]}"
            out.append(d)
    return out


def align(a, b):
    r = tmtools.tm_align(a["ca"], b["ca"], a["seq"], b["seq"])
    xa = a["ca"] @ r.u.T + r.t
    pairs, i, j = [], 0, 0
    for ca, cb in zip(r.seqxA, r.seqyA):
        if ca != "-" and cb != "-":
            dist = float(np.linalg.norm(xa[i] - b["ca"][j]))
            pairs.append((i, j, dist))
        i += ca != "-"; j += cb != "-"
    return max(r.tm_norm_chain1, r.tm_norm_chain2), r.rmsd, pairs


def lab_key(l):
    m = re.match(r"(\d+)(?:\.(\d+)|([A-Z]))?$", l)
    return (int(m.group(1)), int(m.group(2) or 0), m.group(3) or "")


def run(doms, tag):
    """Return per-pair records of label agreement."""
    recs = []
    for a, b in itertools.combinations(doms, 2):
        tm, rmsd, pairs = align(a, b)
        if tm < MIN_TM:
            recs.append(dict(a=a["id"], b=b["id"], tm=tm, rmsd=rmsd, pairs=None)); continue
        P = [(a["lab"][i], b["lab"][j], round(dd, 2)) for i, j, dd in pairs]
        recs.append(dict(a=a["id"], b=b["id"], tm=tm, rmsd=rmsd, pairs=P))
    json.dump(recs, open(os.path.join(ROOT, f"pairs_{tag}.json"), "w"))
    return recs


if __name__ == "__main__":
    doms = load()
    by = defaultdict(list)
    for d in doms:
        by[d["domtype"]].append(d)
    for t, ds in by.items():
        print(t, len(ds), Counter(d["label"] for d in ds).most_common())
    for t in sys.argv[1:]:
        if "+" in t:
            ds = sum((by[x] for x in t.split("+")), [])
        else:
            ds = by[t]
        recs = run(ds, t)
        print(t, "pairs", len(recs), "same-fold", sum(r["pairs"] is not None for r in recs))
