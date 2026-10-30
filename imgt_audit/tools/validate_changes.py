"""Independent pairwise validation of every proposed IMGT -> new relabelling.

For class X with change p -> q, pair X domains with reference-class domains in which label q is
unchanged by the new scheme. After a TM-align superposition of each pair, ask whether X's residue
(IMGT p) lies within 3 A of the reference residue at q (proposal) or at p (IMGT).
"""
import json, os, random, sys
from collections import Counter, defaultdict
from multiprocessing import Pool
import numpy as np
import tmtools
sys.path.insert(0, os.path.dirname(__file__))
from scheme import load_big, lab_key, is_int

D = os.environ.get("SCHEME_DIR", os.path.join(os.path.dirname(__file__), "..", "big", "scheme"))


def _pair(args):
    a, b, tests = args
    r = tmtools.tm_align(a["ca"], b["ca"], a["seq"], b["seq"])
    if max(r.tm_norm_chain1, r.tm_norm_chain2) < 0.5:
        return []
    Xa = a["ca"] @ r.u.T + r.t
    ia = {l: i for i, l in enumerate(a["imgt"])}
    ib = {l: i for i, l in enumerate(b["imgt"])}
    out = []
    for p, q in tests:
        if p not in ia or q not in ib:
            continue
        x = Xa[ia[p]]
        dq = float(np.linalg.norm(x - b["ca"][ib[q]]))
        dp = float(np.linalg.norm(x - b["ca"][ib[p]])) if p in ib else np.nan
        out.append((p, q, dq, dp))
    return out


def main(min_frac=0.6, max_pairs=60, seed=0):
    rnd = random.Random(seed)
    tab = json.load(open(os.path.join(D, "report_tables.json")))
    recs = {r["id"]: r for r in json.load(open(os.path.join(D, "renumbered.json")))}
    doms = {d["id"]: d for d in load_big() if d["id"] in recs}
    for i, r in recs.items():
        if i in doms:
            doms[i]["imgt"] = r["imgt"]; doms[i]["new"] = r["new"]
    bycls = defaultdict(list)
    for d in doms.values():
        bycls[d["cls"]].append(d)
    # labels that the new scheme leaves untouched, per class
    keep = defaultdict(set)
    for c, ds in bycls.items():
        cnt = defaultdict(Counter)
        for d in ds:
            for il, nl in zip(d["imgt"], d["new"]):
                cnt[il][nl == il] += 1
        keep[c] = {l for l, k in cnt.items() if k[True] / sum(k.values()) >= 0.9}
    jobs, meta = [], []
    results = defaultdict(list)
    for fam in ("Ig", "G", "S", "A"):
        for c, chg in tab.get(f"changes_{fam}", {}).items():
            tests = [(p, q) for p, q, f, n in chg if f >= min_frac and not q.startswith("-")]
            if not tests or c not in bycls:
                continue
            fam_of = lambda cls: fam
            refs = [rc for rc in bycls if rc != c and (fam != "Ig" or rc[0] == c[0] or True)]
            for p, q in tests:
                cand = [rc for rc in refs if q in keep[rc] and len(bycls[rc]) >= 3 and same_family(rc, c)]
                if not cand:
                    continue
                for _ in range(max_pairs):
                    rc = rnd.choice(cand)
                    jobs.append((slim(rnd.choice(bycls[c])), slim(rnd.choice(bycls[rc])), [(p, q)]))
                    meta.append((c, p, q, rc))
    with Pool(4) as pool:
        res = pool.map(_pair, jobs, chunksize=10)
    for (c, p, q, rc), rr in zip(meta, res):
        for _, _, dq, dp in rr:
            results[(c, p, q)].append((rc, dq, dp))
    rows = []
    for (c, p, q), v in sorted(results.items(), key=lambda kv: (kv[0][0], lab_key(kv[0][1]))):
        dq = np.array([x[1] for x in v]); dp = np.array([x[2] for x in v])
        okp = ~np.isnan(dp)
        rows.append(dict(cls=c, imgt=p, new=q, n=len(v), refs=sorted({x[0] for x in v}),
                         med_new=round(float(np.median(dq)), 2),
                         med_imgt=round(float(np.median(dp[okp])), 2) if okp.any() else None,
                         new_within3=round(float(np.mean(dq <= 3)), 2),
                         imgt_within3=round(float(np.mean(dp[okp] <= 3)), 2) if okp.any() else None,
                         new_closer=round(float(np.mean(dq[okp] < dp[okp])), 2) if okp.any() else None))
    json.dump(rows, open(os.path.join(D, "validation.json"), "w"), indent=1)
    for r in rows:
        print(f"{r['cls']:>16} {r['imgt']:>6}->{r['new']:<7} n={r['n']:3d} med new {r['med_new']:5.2f} A (<=3A {r['new_within3']:.2f}) | "
              f"IMGT {r['med_imgt'] if r['med_imgt'] is not None else '  -  '} A (<=3A {r['imgt_within3'] if r['imgt_within3'] is not None else ' - '}) | new closer {r['new_closer']}")


def same_family(a, b):
    """References come from the same domain type (V with V, C with C, ...)."""
    def g(c):
        for p, t in (("V", "V"), ("F:", "F"), ("G", "G"), ("S-", "S"), ("A-", "A")):
            if c.startswith(p):
                return t
        return "C"
    return g(a) == g(b)


def slim(d):
    return dict(ca=d["ca"], seq=d["seq"], imgt=d["imgt"])


if __name__ == "__main__":
    main()
