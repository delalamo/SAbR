#!/usr/bin/env python3
"""Recompute the PM0 headline numbers straight from the shared tables (independent of paper_meta code)."""
import pandas as pd
from scipy.stats import binomtest

D = "/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad/deeper"
OUT = "/home/user/SAbR/register_error_analysis/retest/deeper/verify/paper_meta_PM0_amyloid_artifact_integrity"
e = pd.read_csv(f"{D}/entries.csv")
c = pd.read_csv(f"{D}/chains.csv.gz")
r = pd.read_csv(f"{D}/runs_first_occurrence.csv")

# 1. independent identification: title keywords OR light-chain-only homo-oligomer (no H chain, >=3 identical V chains)
kw = e.title.str.contains(r"(?i)amyloid|fibril", na=False)
g = c.groupby("pdb").agg(nH=("ctype", lambda s: (s == "H").sum()), nV=("ctype", "size"), nseq=("vseq", "nunique"))
homo = set(g[(g.nH == 0) & (g.nV >= 3) & (g.nseq == 1)].index)
cand = e[kw | e.pdb.isin(homo)][["pdb", "method", "res", "year", "err_first", "err_any", "modes", "title"]]
cand.to_csv(f"{OUT}/candidate_fibril_entries.csv", index=False)
AMY = sorted(e[e.title.str.contains(r"(?i)amyloid fibril|AL amyloid|amyloidosis|amyloid AL", na=False)].pdb)
print("AL fibril entries in analysis set:", len(AMY), AMY)
a = e[e.pdb.isin(AMY)]
print("flagged err_first:", int(a.err_first.sum()), "/", len(a), "; methods", a.method.value_counts().to_dict())
print("modes per entry:", a.modes.value_counts().to_dict())

# 2. units and runs
em = c[c.method == "EM"]
u = em.groupby("unit").agg(err=("err_first", "any"), amy=("pdb", lambda s: s.isin(AMY).any()), ctype=("ctype", "first"))
print("EM error units:", int(u.err.sum()), "of", len(u), "; amyloid error units:", int((u.err & u.amy).sum()))
r["amy"] = r.pdb.isin(AMY)
print("first-occurrence runs:", len(r), "amyloid:", int(r.amy.sum()), f"({r.amy.mean():.1%}); EM runs", int((r.method == 'EM').sum()))
print("run length median amyloid vs other:", r[r.amy].length.median(), r[~r.amy].length.median())
print("|delta| median amyloid vs other:", r[r.amy].delta.abs().median(), r[~r.amy].delta.abs().median())
print("origin 'chain end/unbuilt loop' share of runs: with amyloid %.3f, without %.3f; amyloid runs in that class %d/%d" % (
    (r.cls == 3).mean(), (r[~r.amy].cls == 3).mean(), int((r.amy & (r.cls == 3)).sum()), int((r.cls == 3).sum())))
print("mode x amyloid (runs):\n", pd.crosstab(r["mode"], r.amy))

# 3. downstream distortions if NOT excluded
lam = u[u.ctype == "L"]
print("EM lambda error units: total", int(lam.err.sum()), "of which amyloid", int((lam.err & lam.amy).sum()),
      "; lambda unit rate with/without amyloid: %.3f / %.3f; kappa %.3f; heavy %.3f" % (
          lam.err.mean(), lam[~lam.amy].err.mean(), u[u.ctype == "K"].err.mean(), u[u.ctype == "H"].err.mean()))
eem = e[e.method == "EM"].assign(amy=lambda d: d.pdb.isin(AMY))
for y0, y1 in [(2024, 2025)]:
    s = eem[(eem.year >= y0) & (eem.year <= y1)]
    print(f"EM entry error rate {y0}-{y1}: with amyloid {s.err_first.mean():.3f} ({int(s.err_first.sum())}/{len(s)}), "
          f"without {s[~s.amy].err_first.mean():.3f}")
p0 = eem[~eem.amy].err_first.mean()
print("class-level enrichment: 15/15 vs EM base rate %.3f -> binomial p = %.2e" % (p0, binomtest(15, 15, p0, alternative='greater').pvalue))
