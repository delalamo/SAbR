#!/usr/bin/env python3
"""Paper-level lab clustering: do error PAPERS cluster by lab (PDB last author) once paper size is accounted for?
Statistic: sum over labs with >=2 papers of (O-E)^2 / sum p(1-p), E from period + log2 n_units + resolution.
Null: permute error labels within strata (period x n_units bin x resolution bin), 5000 permutations."""
import re, numpy as np, pandas as pd, statsmodels.formula.api as smf
OUT = "/home/user/SAbR/register_error_analysis/retest/deeper/paper_meta"
rng = np.random.default_rng(11)
D = pd.read_csv(f"{OUT}/paper_analysis_table.csv")
res = []
for name, X in [("EM all papers", D[(D.method == "EM") & D.group.isin(["error", "clean"])]),
                ("EM published", D[(D.method == "EM") & D.group.isin(["error", "clean"]) & D.published])]:
    X = X.dropna(subset=["res_median"]).copy().reset_index(drop=True)
    X["lab"] = X.last_author.fillna("").str.lower().str.replace(r"[^a-z]", "", regex=True)
    m = smf.logit("err ~ C(period) + l2_units + res_median", X).fit(disp=0)
    X["p"] = m.predict(X)
    X["stratum"] = X.period + "|" + pd.cut(X.n_units, [0, 1, 2, 4, 1000]).astype(str) + "|" + pd.cut(X.res_median, [0, 3.3, 3.7, 6]).astype(str)
    labs = X.groupby("lab").filter(lambda g: len(g) >= 2)
    idx = [g.index.values for _, g in labs.groupby("lab")]
    p, v = X.p.values, (X.p * (1 - X.p)).values
    def stat(e):
        return sum((e[i].sum() - p[i].sum()) ** 2 / v[i].sum() for i in idx) / len(idx)
    e = X.err.values.astype(float)
    obs = stat(e)
    groups = [np.where(X.stratum.values == s)[0] for s in X.stratum.unique()]
    null = []
    for _ in range(5000):
        e2 = e.copy()
        for g in groups:
            e2[g] = rng.permutation(e[g])
        null.append(stat(e2))
    null = np.array(null)
    # labs with >=2 error papers
    k = labs.groupby("lab").agg(n=("err", "size"), err=("err", "sum"), exp=("p", "sum"))
    res.append(dict(set=name, n_papers=len(X), n_err=int(e.sum()), n_labs_ge2=len(idx), dispersion_obs=obs,
                    dispersion_null=null.mean(), ratio=obs / null.mean(), p_perm=(1 + (null >= obs).sum()) / 5001,
                    labs_with_ge2_err=int((k.err >= 2).sum()), labs_with_ge2_err_expected=np.nan))
    print(k[k.err >= 2].sort_values("err", ascending=False).head(15))
R = pd.DataFrame(res)
R.to_csv(f"{OUT}/lab_dispersion_paper_level.csv", index=False)
print(R.to_string())
