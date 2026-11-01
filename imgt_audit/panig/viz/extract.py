"""Data for the IMGT vs pan-Ig comparison page: representative domains, labels, observed strands/sheets."""
import sys, pickle, json, re
sys.path[:0] = ['/tmp/pan', '/tmp/sabr_unified/tools']
import numpy as np, ulib, pan
from collections import Counter, defaultdict
from Bio.PDB import MMCIFParser
from number2 import Scheme
f2 = pickle.load(open('/tmp/pan/final2.pkl', 'rb')); doms, cache, _ = ulib.load_dataset()
dd = [d for d in doms if d['id'] in f2['labels']]
SC = Scheme(f2['anchors'])
STR = [("A", 1, 8), ("A'", 9, 16), ("B", 17, 26), ("C", 39, 46), ("C'", 47, 55), ("C''", 66, 74), ("D", 75, 84),
       ("E", 85, 96), ("F", 97, 104), ("G", 118, 128)]
LOOPS = [("BC (CDR1)", 27, 38), ("C'C'' (CDR2)", 56, 65), ("FG (CDR3)", 105, 117)]
def num(l):
    m = re.match(r"(\d+)", str(l)); return int(m.group(1)) if m else None
def strand_of(l):
    n = num(l)
    if n is None: return None
    for s, a, b in STR:
        if a <= n <= b: return s
    return None

def dssp(res):
    """Minimal Kabsch-Sander: backbone H-bonds -> beta bridges -> E residues and sheet ids."""
    n = len(res)
    N = np.array([r['N'].coord for r in res]); C = np.array([r['C'].coord for r in res]); O = np.array([r['O'].coord for r in res])
    H = N.copy()
    for i in range(1, n):
        v = C[i - 1] - O[i - 1]; H[i] = N[i] + v / np.linalg.norm(v)
    def d(a, b): return np.linalg.norm(a[:, None] - b[None], axis=2) + 1e-6
    # E[i, j]: C=O of i accepts from N-H of j
    E = 0.084 * 332 * (1 / d(O, N) + 1 / d(C, H) - 1 / d(O, H) - 1 / d(C, N))
    hb = E < -0.5
    np.fill_diagonal(hb, False); hb[:, 0] = False
    for i in range(n - 1): hb[i, i + 1] = False
    def HB(a, b): return 0 <= a < n and 0 <= b < n and hb[a, b]
    bridges = []
    for i in range(1, n - 1):
        for j in range(1, n - 1):
            if abs(i - j) < 3: continue
            par = (HB(i - 1, j) and HB(j, i + 1)) or (HB(j - 1, i) and HB(i, j + 1))
            anti = (HB(i, j) and HB(j, i)) or (HB(i - 1, j + 1) and HB(j - 1, i + 1))
            if par or anti: bridges.append((i, j))
    partners = defaultdict(set)
    for i, j in bridges: partners[i].add(j); partners[j].add(i)
    ss = ['-'] * n
    for i in partners:
        ss[i] = 'E' if any(abs(i - k) == 1 and k in partners for k in (i - 1, i + 1)) else 'B'
    # residue graph: bridges + backbone steps that continue a ladder (partners stay within +-2)
    Eres = [i for i in range(n) if ss[i] in 'EB']
    parent = {i: i for i in Eres}
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b): parent[find(a)] = find(b)
    for i, j in bridges: union(i, j)
    step = {}
    for i in Eres:
        for k in (i + 1, i + 2):
            if k in parent and any(abs(p - q) <= 2 for p in partners[i] for q in partners[k]):
                union(i, k); step[i] = k; break
    sheet = {i: find(i) for i in Eres}
    # strands: maximal runs linked by ladder-continuing steps
    runs, seen = [], set()
    for i in Eres:
        if i in seen: continue
        run = [i]; seen.add(i); k = i
        while k in step and step[k] not in seen:
            k = step[k]; run.append(k); seen.add(k)
        if len(run) >= 2: runs.append(run)
    sid = {}
    for k, r in enumerate(runs):
        for i in range(r[0], r[-1] + 1): sid[i] = k
    sheet = {i: sheet.get(i, sheet[runs[sid[i]][0]]) for i in sid}
    return ss, sid, sheet

parser = MMCIFParser(QUIET=True)
def residues_for(d):
    c = cache[(d['pdb'], d['chain'])]; a, b = d['rows']
    s = parser.get_structure(d['pdb'], f"/tmp/sabr_unified/cif/{d['pdb']}.cif")
    model = next(iter(s)); ch = model[d['chain']]
    byid = {}
    for r in ch:
        byid[f"{r.id[1]}{r.id[2].strip()}"] = r
    ids = [str(x) for x in c['ids'][a:b]]
    return [byid.get(i) for i in ids], ids

CLASSES = [("VH", "V", "Antibody heavy V"), ("V-KAPPA", "V", "Antibody κ V"), ("V-LAMBDA", "V", "Antibody λ V"),
           ("V-ALPHA", "V", "TCR α V"), ("V-BETA", "V", "TCR β V"), ("V-GAMMA", "V", "TCR γ V"), ("V-DELTA", "V", "TCR δ V"),
           ("V-LIKE", "V-like", "IgSF V-like"),
           ("CH1", "C1", "Antibody CH1"), ("C-KAPPA", "C1", "Antibody Cκ"), ("C-LAMBDA", "C1", "Antibody Cλ"),
           ("CH2", "C1", "Antibody CH2"), ("CH3", "C1", "Antibody CH3"), ("C-BETA", "C1", "TCR Cβ"),
           ("C-LIKE:MHC-I-a3", "C1", "MHC I α3"), ("C-LIKE:MHC-II", "C1", "MHC II α2/β2"), ("C-LIKE:B2M", "C1", "β2-microglobulin"),
           ("C-ALPHA", "C1", "TCR Cα (TRAC)"), ("C-LIKE:IgSF", "C2/I", "IgSF C2/I-set"),
           ("F:FN/TNC", "FN3", "FN3 (fibronectin/tenascin)"), ("F:other", "FN3", "FN3 (other)")]
out = []
for cls, typ, name in CLASSES:
    D = [d for d in dd if d['cls'] == cls]
    if not D: continue
    ident = [np.mean([a == b for a, b in zip(f2['labels'][d['id']], d['imgt'])]) for d in D]
    med = float(np.median(ident))
    order = sorted(range(len(D)), key=lambda k: abs(ident[k] - med))
    rep = None
    for k in order[:15]:
        d = D[k]; R, ids = residues_for(d)
        if all(r is not None and all(x in r for x in ('N', 'CA', 'C', 'O')) for r in R):
            rep = (d, R, ids); break
    if rep is None: print("skip", cls); continue
    d, R, ids = rep
    ss, sid, sheet = dssp(R)
    P = f2['labels'][d['id']]; I = d['imgt']; cl = set(f2['claims'][d['id']])
    c = cache[(d['pdb'], d['chain'])]; a, b = d['rows']; seq = c['seq'][a:b]
    # name sheets by the pan strands they contain
    sheet_str = defaultdict(Counter)
    for i in sheet:
        s = strand_of(P[i])
        if s: sheet_str[sheet[i]][s] += 1
    def sheet_name(k):
        cnt = sheet_str[k]
        if not cnt: return "?"
        front = sum(cnt[s] for s in ("A", "B", "E", "D")); back = sum(cnt[s] for s in ("A'", "G", "F", "C", "C'", "C''"))
        return "ABED" if front > back else "A'GFCC'C''"
    LOOPN = lambda l: next((nm.split()[0] for nm, a2, b2 in LOOPS if num(l) and a2 <= num(l) <= b2), "loop")
    res = []
    for i in range(len(P)):
        kind = "anchor" if i in cl else ("insertion" if "." in P[i] else "fill")
        res.append(dict(aa=seq[i], auth=ids[i], imgt=I[i], pan=P[i], kind=kind, ss=ss[i],
                        sheet=(sheet_name(sheet[i]) if i in sheet else None),
                        istr=strand_of(I[i]), pstr=strand_of(P[i])))
    # observed strands: per DSSP run, the pan and IMGT names (majority)
    strands = []
    for k in sorted(set(sid.values())):
        idx = [i for i in sid if sid[i] == k]
        pn = Counter(strand_of(P[i]) or LOOPN(P[i]) for i in idx).most_common(1)[0][0]
        im = Counter(strand_of(I[i]) or LOOPN(I[i]) for i in idx).most_common(1)[0][0]
        strands.append(dict(start=min(idx), end=max(idx), pan=pn, imgt=im, sheet=sheet_name(sheet[idx[0]]),
                            pan_range=f"{P[min(idx)]}–{P[max(idx)]}", imgt_range=f"{I[min(idx)]}–{I[max(idx)]}"))
    # class-level stats
    regdiff = Counter(); regtot = Counter(); ins_imgt = Counter(); ins_pan = Counter(); occ_imgt = Counter(); occ_pan = Counter()
    for dx in D:
        Px, Ix = f2['labels'][dx['id']], dx['imgt']
        for p, im in zip(Px, Ix):
            r = strand_of(im) or next((l for l, a2, b2 in LOOPS if num(im) and a2 <= num(im) <= b2), "other")
            regtot[r] += 1; regdiff[r] += p != im
            if "." in im: ins_imgt[im.split(".")[0]] += 1
            if "." in p: ins_pan[p.split(".")[0]] += 1
        for l in set(Ix):
            if num(l) and "." not in l: occ_imgt[num(l)] += 1
        for l in set(Px):
            if num(l) and "." not in l: occ_pan[num(l)] += 1
    out.append(dict(cls=cls, type=typ, name=name, n=len(D), identity=med, mean_identity=float(np.mean(ident)),
                    rep=d['id'], pdb=d['pdb'], chain=d['chain'], residues=res, strands=strands,
                    regdiff={k: round(regdiff[k] / regtot[k], 3) for k in regtot},
                    ins_imgt={k: round(v / len(D), 2) for k, v in ins_imgt.most_common(8)},
                    ins_pan={k: round(v / len(D), 2) for k, v in ins_pan.most_common(8)},
                    occ_imgt={k: round(v / len(D), 2) for k, v in occ_imgt.items()},
                    occ_pan={k: round(v / len(D), 2) for k, v in occ_pan.items()}))
    print(cls, d['id'], f"identity {ident[order[0]] if False else np.mean([a == b for a, b in zip(P, I)]):.2f}", "strands:",
          " ".join(f"{s['pan']}{'' if s['pan'] == s['imgt'] else '(IMGT ' + str(s['imgt']) + ')'}[{s['sheet'][:2]}:{s['pan_range']}]" for s in strands))
meta = dict(anchors=f2['anchors'], segments=[dict(start=s[0], end=s[-1], apex=SC.apex.get(k)) for k, s in enumerate(SC.segments)],
            strands=[dict(name=s, start=a, end=b) for s, a, b in STR], loops=[dict(name=s, start=a, end=b) for s, a, b in LOOPS])
json.dump(dict(meta=meta, domains=out), open('/tmp/pan/viz/data.json', 'w'))
