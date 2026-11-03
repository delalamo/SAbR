#!/usr/bin/env python3
"""Re-run paper_meta's per-author enrichment (network.py section 2) with the amyloid papers put back as 'error'
papers, to check the claim 'Fandrich/Schonland group = top co-author hit, 3/3 papers flagged, q=0.06'.
Same model (err ~ period + log2 units + res_median) and Poisson-binomial upper tail + BH as network.py."""
import re, unicodedata, warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings("ignore")
PM = "/home/user/SAbR/register_error_analysis/retest/deeper/paper_meta"
OUT = "/home/user/SAbR/register_error_analysis/retest/deeper/verify/paper_meta_PM0_amyloid_artifact_confounding"
D = pd.read_csv(f"{PM}/paper_analysis_table.csv")
A0 = pd.read_csv(f"{PM}/paper_authors.csv.gz")

def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z]", "", s)

def pb_upper(ps, k):
    dist = np.zeros(len(ps) + 1); dist[0] = 1
    for p in ps:
        dist[1:] = dist[1:] * (1 - p) + dist[:-1] * p; dist[0] *= (1 - p)
    return float(dist[k:].sum())

def run(include_amy):
    grp = ["error", "clean"] + (["amyloid_artifact"] if include_amy else [])
    X = D[(D.method == "EM") & D.group.isin(grp) & D.published & D.n_authors.notna()].dropna(subset=["res_median"]).copy()
    X["err"] = X.group.isin(["error", "amyloid_artifact"]).astype(int)
    X = X.reset_index(drop=True)
    m = smf.logit("err ~ C(period) + l2_units + res_median", X).fit(disp=0)
    X["p_hat"] = m.predict(X)
    A = A0[A0.doi.isin(X.doi)].copy()
    A["akey"] = np.where(A.orcid.notna(), "orcid:" + A.orcid.astype(str), "n:" + A["last"].map(norm) + "_" + A.initials.fillna("").map(norm))
    nm = A.assign(nk="n:" + A["last"].map(norm) + "_" + A.initials.fillna("").map(norm))
    one = nm[nm.orcid.notna()].groupby("nk").orcid.nunique(); one = one[one == 1].index
    o_map = nm[nm.orcid.notna() & nm.nk.isin(one)].groupby("nk").orcid.first()
    A["akey"] = np.where(A.orcid.isna() & nm.nk.isin(o_map.index), "orcid:" + nm.nk.map(o_map).astype(str), A.akey)
    A["name"] = A["last"].astype(str) + " " + A.initials.fillna("").astype(str)
    names = A.groupby("akey").name.agg(lambda s: s.value_counts().index[0])
    doi_idx = {d: i for i, d in enumerate(X.doi)}
    err = X.err.values.astype(bool)
    amy = X.group.eq("amyloid_artifact").values
    out = []
    for ent, g in A[["doi", "akey"]].drop_duplicates().groupby("akey"):
        idx = [doi_idx[d] for d in g.doi]
        if len(idx) < 3: continue
        o = int(err[idx].sum()); ps = X.p_hat.values[idx]
        out.append(dict(entity=ent, name=names[ent], n_papers=len(idx), n_err=o, n_amy=int(amy[idx].sum()),
                        expected=ps.sum(), p_upper=pb_upper(ps, o) if o else 1.0))
    R = pd.DataFrame(out)
    R["q"] = multipletests(R.p_upper, method="fdr_bh")[1]
    R["include_amyloid"] = include_amy
    return R.sort_values("p_upper"), X

R1, X1 = run(True)
R0, X0 = run(False)
print("papers (incl amyloid):", len(X1), "error:", int(X1.err.sum()), " amyloid published papers:", int((X1.group == 'amyloid_artifact').sum()))
pd.set_option("display.width", 250)
print("== WITH amyloid papers as error")
print(R1.head(12).to_string())
print("== WITHOUT (as in network.py)")
print(R0.head(6).to_string())
pd.concat([R1.head(40), R0.head(40)]).to_csv(f"{OUT}/author_enrichment_with_without_amyloid.csv", index=False)
