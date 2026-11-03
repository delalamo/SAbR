#!/usr/bin/env python3
"""Reproduce the paper_meta per-author enrichment (network.py, section 2) with and without the 5 published
AL-amyloid papers counted as error papers, to check the claim 'before exclusion the Faendrich/Schoenland group
was the top co-author hit (3/3 papers, q=0.06)'. Logic copied from paper_meta/network.py; only the case set changes."""
import re, unicodedata, warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")
PM = "/home/user/SAbR/register_error_analysis/retest/deeper/paper_meta"
OUT = "/home/user/SAbR/register_error_analysis/retest/deeper/verify/paper_meta_PM0_amyloid_artifact_integrity"
D0 = pd.read_csv(f"{PM}/paper_analysis_table.csv", low_memory=False)
A0 = pd.read_csv(f"{PM}/paper_authors.csv.gz")


def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z]", "", s)


def pb_upper(ps, k):
    dist = np.zeros(len(ps) + 1)
    dist[0] = 1
    for p in ps:
        dist[1:] = dist[1:] * (1 - p) + dist[:-1] * p
        dist[0] *= (1 - p)
    return float(dist[k:].sum())


def run(include_amyloid):
    D = D0.copy()
    groups = ["error", "clean"] + (["amyloid_artifact"] if include_amyloid else [])
    if include_amyloid:
        D.loc[D.group == "amyloid_artifact", "err"] = 1
    X = D[(D.method == "EM") & D.group.isin(groups) & D.published & D.n_authors.notna()].dropna(subset=["res_median"]).copy()
    X = X.reset_index(drop=True)
    m = smf.logit("err ~ C(period) + l2_units + res_median", X).fit(disp=0)
    X["p_hat"] = m.predict(X)
    A = A0[A0.doi.isin(X.doi)].copy()
    A["akey"] = np.where(A.orcid.notna(), "orcid:" + A.orcid.astype(str), "n:" + A["last"].map(norm) + "_" + A.initials.fillna("").map(norm))
    nm = A.assign(nk="n:" + A["last"].map(norm) + "_" + A.initials.fillna("").map(norm))
    one = nm[nm.orcid.notna()].groupby("nk").orcid.nunique()
    one = one[one == 1].index
    o_map = nm[nm.orcid.notna() & nm.nk.isin(one)].groupby("nk").orcid.first()
    A["akey"] = np.where(A.orcid.isna() & nm.nk.isin(o_map.index), "orcid:" + nm.nk.map(o_map).astype(str), A.akey)
    A["name"] = A["last"].astype(str) + " " + A.initials.fillna("").astype(str)
    names = A.groupby("akey").name.agg(lambda s: s.value_counts().index[0])
    doi_idx = {d: i for i, d in enumerate(X.doi)}
    err = X.err.values.astype(bool)
    lab = X.last_author.fillna("").map(norm).values
    memb = A[["doi", "akey"]].drop_duplicates()
    out = []
    for ent, g in memb.groupby("akey"):
        idx = [doi_idx[d] for d in g.doi if d in doi_idx]
        if len(idx) < 3:
            continue
        o = int(err[idx].sum())
        ps = X.p_hat.values[idx]
        out.append(dict(entity=ent, name=names.get(ent), n_papers=len(idx), n_err=o, expected=ps.sum(),
                        p_upper=pb_upper(ps, o) if o > 0 else 1.0, n_labs=len({lab[i] for i in idx}),
                        n_err_labs=len({lab[i] for i in idx if err[i]})))
    R = pd.DataFrame(out)
    R["q"] = multipletests(R.p_upper, method="fdr_bh")[1]
    R["include_amyloid"] = include_amyloid
    return R.sort_values("p_upper"), X


res = []
for inc in (False, True):
    R, X = run(inc)
    print(f"== include_amyloid={inc}: papers {len(X)}, error {int(X.err.sum())}, authors tested {len(R)}")
    print(R.head(12)[["name", "n_papers", "n_err", "expected", "p_upper", "q", "n_labs", "n_err_labs"]].to_string())
    res.append(R)
pd.concat(res).to_csv(f"{OUT}/author_enrichment_with_without_amyloid.csv", index=False)
