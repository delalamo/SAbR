"""Tables describing the derived scheme: anchor sites, strand membership, IMGT->new changes per class."""
import json, os, re, sys
from collections import Counter, defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from scheme import lab_key, is_int

D = os.environ.get("SCHEME_DIR", os.path.join(os.path.dirname(__file__), "..", "big", "scheme"))
IG_STRANDS = [("A", 1, 15), ("B", 16, 26), ("BC", 27, 38), ("C", 39, 46), ("C'", 47, 55), ("C'C''", 56, 65),
              ("C''", 66, 74), ("D", 75, 84), ("E", 85, 96), ("F", 97, 104), ("FG", 105, 117), ("G", 118, 128)]
G_STRANDS = [("A", 1, 9), ("AB", 10, 10), ("B", 11, 17), ("BC", 18, 19), ("C", 20, 27), ("CD", 28, 29), ("D", 30, 37),
             ("D-helix", 38, 49), ("helix", 50, 92)]


def region(lab, table):
    n = lab_key(lab)[0]
    for nm, lo, hi in table:
        if lo <= n <= hi:
            return nm
    return "N-ext" if n < table[0][1] else "C-ext"


def group(cls):
    if cls.startswith("V"): return "V"
    if cls.startswith("F"): return "F"
    if cls.startswith("G"): return "G"
    if cls.startswith("S"): return "S"
    if cls.startswith("A-"): return "A"
    return "C"


def site_table(recs, fam, table):
    """Per anchor site: SS, sheet partner strand, occupancy by group, IMGT names by class."""
    cons = json.load(open(os.path.join(D, f"consensus_{fam}.json")))
    names = cons["names"]
    ss = defaultdict(Counter); partner = defaultdict(Counter); occ = defaultdict(Counter); imgt = defaultdict(lambda: defaultdict(Counter))
    ngroup = Counter(group(r["cls"]) for r in recs)
    for r in recs:
        g = group(r["cls"])
        part = r.get("partner") or [-1] * len(r["new"])
        for i, (nl, il, s, st) in enumerate(zip(r["new"], r["imgt"], r["site"], r["ss"])):
            if not s:
                continue
            ss[nl][st] += 1; occ[nl][g] += 1; imgt[nl][r["cls"]][il] += 1
            if part[i] >= 0:
                partner[nl][region(r["new"][part[i]], table)] += 1
    rows = []
    for n in names:
        tot = sum(ss[n].values()) or 1
        rows.append(dict(site=n, region=region(n, table), E=round(ss[n]["E"] / tot, 2), H=round(ss[n]["H"] / tot, 2),
                         partner=partner[n].most_common(1)[0][0] if partner[n] else "",
                         occ={g: round(occ[n][g] / ngroup[g], 2) for g in ngroup if occ[n][g]},
                         imgt={c: cnt.most_common(1)[0] for c, cnt in imgt[n].items()}))
    return rows


def changes(recs, min_frac=0.5, min_n=5):
    """For each class and IMGT label: modal new label when it differs."""
    t = defaultdict(lambda: defaultdict(Counter))
    for r in recs:
        for il, nl in zip(r["imgt"], r["new"]):
            t[r["cls"]][il][nl] += 1
    out = defaultdict(list)
    for c, d in t.items():
        for il in sorted(d, key=lab_key):
            n = sum(d[il].values()); nl, k = d[il].most_common(1)[0]
            if n >= min_n and k / n >= min_frac and nl != il:
                out[c].append((il, nl, round(k / n, 2), n))
    return out


def compress(chg):
    """Group consecutive integer changes with the same offset into ranges."""
    runs, cur = [], []
    for il, nl, f, n in chg:
        if is_int(il) and is_int(nl):
            off = int(nl) - int(il)
            if cur and is_int(cur[-1][0]) and is_int(cur[-1][1]) and int(il) == int(cur[-1][0]) + 1 and int(cur[-1][1]) - int(cur[-1][0]) == off:
                cur.append((il, nl, f, n)); continue
        if cur: runs.append(cur)
        cur = [(il, nl, f, n)]
    if cur: runs.append(cur)
    txt = []
    for r in runs:
        if len(r) == 1:
            txt.append(f"{r[0][0]}→{r[0][1]} ({r[0][2]:.0%})")
        else:
            off = int(r[0][1]) - int(r[0][0])
            txt.append(f"{r[0][0]}–{r[-1][0]}→{r[0][1]}–{r[-1][1]} ({off:+d}; {min(x[2] for x in r):.0%}+)")
    return txt


if __name__ == "__main__":
    recs = json.load(open(os.path.join(D, "renumbered.json")))
    by = defaultdict(list)
    for r in recs:
        by["Ig" if r["domtype"] in "VCF" else r["domtype"]].append(r)
    out = {}
    for fam, tab in (("Ig", IG_STRANDS), ("G", G_STRANDS), ("S", IG_STRANDS), ("A", IG_STRANDS)):
        if fam not in by or not os.path.exists(os.path.join(D, f"consensus_{fam}.json")):
            continue
        out[f"sites_{fam}"] = site_table(by[fam], fam, tab)
        ch = changes(by[fam])
        out[f"changes_{fam}"] = {c: v for c, v in ch.items()}
        out[f"changes_{fam}_compressed"] = {c: compress(v) for c, v in ch.items()}
    json.dump(out, open(os.path.join(D, "report_tables.json"), "w"), indent=1)
    for k, v in out.items():
        if k.endswith("compressed"):
            print("==", k)
            for c, txt in sorted(v.items()):
                print(f"  {c:18s} {'; '.join(txt)}")
