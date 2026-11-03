#!/usr/bin/env python3
"""Alternative mechanism: correction bias.  If errors in highly cited papers are noticed and the coordinates
replaced (PDB revision with 'Atomic model' / 'Polymer sequence' group, major_revision >= 2), those papers would
now scan clean, producing 'error papers are cited less' without low visibility causing errors.

Tests: (a) do coordinate-replaced papers have more citations? (b) is the citation OR unchanged when papers with any
coordinate replacement in an EM entry are dropped? (c) GEE/cluster versions.
"""
import json, warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
OUT = "/home/user/SAbR/register_error_analysis/retest/deeper/verify/paper_meta_PM2_lower_citations_confounding"
MOD = "/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad/retest/modelling.jsonl"
X = pd.read_csv(f"{OUT}/v01_analysis_frame.csv.gz")
rev = {}
for line in open(MOD):
    r = json.loads(line)
    groups = {g["group"] for g in (r.get("pdbx_audit_revision_group") or [])}
    majors = [h.get("major_revision") or 1 for h in (r.get("pdbx_audit_revision_history") or [])]
    rev[r["rcsb_id"].lower()] = dict(coord=bool(groups & {"Atomic model", "Polymer sequence", "Non-polymer description"}),
                                     major2=bool(majors and max(majors) >= 2), known=True)
cov, coord, major2 = [], [], []
for s in X.pdbs:
    ids = s.split(";")
    k = [rev.get(i.lower()) for i in ids]
    k = [z for z in k if z]
    cov.append(len(k) / len(ids))
    coord.append(float(any(z["coord"] for z in k)) if k else np.nan)
    major2.append(float(any(z["major2"] for z in k)) if k else np.nan)
X["rev_cov"] = cov
X["coord_rev"] = coord
X["major2"] = major2
print("revision coverage among papers:", np.mean(np.array(cov) > 0), "full", np.mean(np.array(cov) == 1))
print(X.groupby("err")[["coord_rev", "major2"]].mean())
XV = "l_cites_per_year"
BASE = "C(period) + l2_units + res_median"
out = []


def fit(f, d, name, x=XV):
    m = smf.logit(f, d).fit(disp=0, maxiter=300, cov_type="cluster", cov_kwds={"groups": pd.factorize(d.lab)[0]})
    ci = m.conf_int().loc[x]
    out.append(dict(test=name, x=x, OR=np.exp(m.params[x]), lo=np.exp(ci[0]), hi=np.exp(ci[1]), p=m.pvalues[x], n=len(d), n_err=int(d.err.sum())))
    print(f"{name:60s} {x:18s} OR={np.exp(m.params[x]):.3f} ({np.exp(ci[0]):.2f}-{np.exp(ci[1]):.2f}) p={m.pvalues[x]:.3g} n={len(d)} ne={int(d.err.sum())}")


d = X.dropna(subset=["coord_rev"])
# (a) are coordinate replacements more common in highly cited papers?
m = smf.logit(f"coord_rev ~ {XV} + {BASE}", d).fit(disp=0)
out.append(dict(test="coord replacement ~ cites (all papers)", x=XV, OR=np.exp(m.params[XV]), p=m.pvalues[XV], n=len(d)))
print("coord_rev ~ cites", np.exp(m.params[XV]), m.pvalues[XV], "rate", d.coord_rev.mean())
# (b) error ~ coord_rev
fit(f"err ~ coord_rev + {BASE}", d, "err ~ coordinate replacement", x="coord_rev")
fit(f"err ~ {XV} + coord_rev + {BASE}", d, "cites + coordinate replacement")
fit(f"err ~ {XV} + {BASE}", d[d.coord_rev == 0], "papers never coordinate-replaced")
fit(f"err ~ {XV} + {BASE}", d[d.major2 == 0], "papers never major revision >=2")
pd.DataFrame(out).to_csv(f"{OUT}/v04_correction_bias.csv", index=False)
X[["pkey", "rev_cov", "coord_rev", "major2"]].to_csv(f"{OUT}/v04_revision_flags.csv.gz", index=False)
