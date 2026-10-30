"""Cross-type alignment (V vs C vs F): do shared IMGT labels superpose?"""
import itertools, json, os, sys, random
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(__file__))
from analyze import load, align, CUTOFF, lab_key

random.seed(0)
doms = load()
pick = {"VH": 4, "V-KAPPA": 4, "V-LAMBDA": 3, "V-ALPHA": 3, "V-BETA": 3,
        "CH1": 3, "C-KAPPA": 3, "C-LAMBDA": 2, "CH3": 3, "C-LIKE:B2M": 3, "C-LIKE:MHC-I-a3": 3, "C-BETA-2": 3,
        "F:FN/TNC": 3, "F:cytokine-R": 3}
bycls = defaultdict(list)
for d in doms:
    bycls[d["cls"]].append(d)
sel = {c: random.sample(bycls[c], min(k, len(bycls[c]))) for c, k in pick.items()}
grp = lambda c: "V" if c.startswith("V") else ("F" if c.startswith("F") else "C")
stat = defaultdict(lambda: defaultdict(Counter))
tms = defaultdict(list)
all_sel = [d for v in sel.values() for d in v]
for a, b in itertools.combinations(all_sel, 2):
    ga, gb = grp(a["cls"]), grp(b["cls"])
    if ga == gb:
        continue
    tm, rmsd, pairs = align(a, b)
    key = tuple(sorted((ga, gb)))
    tms[key].append(tm)
    m_ab = {a["lab"][i]: b["lab"][j] for i, j, dd in pairs if dd <= CUTOFF}
    m_ba = {v: k for k, v in m_ab.items()}
    for x, mp, gx, gy in ((a, m_ab, ga, gb), (b, m_ba, gb, ga)):
        for p in x["lab"]:
            stat[(gx, gy)][p][mp.get(p, "-")] += 1
for k, v in tms.items():
    print(k, "n", len(v), "TM mean %.2f" % (sum(v) / len(v)))
key_pos = ["16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "39", "40", "41", "42", "43", "44", "45",
           "77", "78", "79", "80", "81", "82", "83", "84", "85", "86", "87", "88", "89", "90", "91", "92", "97", "98",
           "99", "100", "101", "102", "103", "104", "118", "119", "120", "121", "122", "123"]
print("pos   " + "  ".join(f"{a+'>'+b:>14}" for a, b in [("V", "C"), ("C", "V"), ("C", "F"), ("V", "F")]))
for p in key_pos:
    cells = []
    for k in [("V", "C"), ("C", "V"), ("C", "F"), ("V", "F")]:
        c = stat[k].get(p)
        if not c: cells.append(" " * 14); continue
        q, n = c.most_common(1)[0]; tot = sum(c.values())
        cells.append(f"{q + ('*' if q not in (p, '-') else ''):>8} {n/tot:4.2f}")
    print(f"{p:>4}  " + "  ".join(cells))
