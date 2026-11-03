#!/usr/bin/env python3
"""Heterogeneity and measurement checks: size x citation interaction, size subgroups, citation source,
percentile metric with lab clustering, period-specific estimates (lab-clustered), X-ray replication and
EM+X-ray pooled model.  Then BH across all verification tests (v01, v03, v04, v05, v06)."""
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")
PM = "/home/user/SAbR/register_error_analysis/retest/deeper/paper_meta"
OUT = "/home/user/SAbR/register_error_analysis/retest/deeper/verify/paper_meta_PM2_lower_citations_confounding"
X = pd.read_csv(f"{OUT}/v01_analysis_frame.csv.gz", parse_dates=["pub_date"])
XV = "l_cites_per_year"
BASE = "C(period) + l2_units + res_median"
out = []


def fit(f, d, name, x=XV, cluster="lab"):
    if cluster:
        m = smf.logit(f, d).fit(disp=0, maxiter=300, cov_type="cluster", cov_kwds={"groups": pd.factorize(d[cluster])[0]})
    else:
        m = smf.logit(f, d).fit(disp=0, maxiter=300)
    ci = m.conf_int().loc[x]
    out.append(dict(test=name, x=x, OR=np.exp(m.params[x]), lo=np.exp(ci[0]), hi=np.exp(ci[1]), p=m.pvalues[x], n=len(d), n_err=int(d.err.sum())))
    print(f"{name:60s} {x:28s} OR={np.exp(m.params[x]):.3f} ({np.exp(ci[0]):.2f}-{np.exp(ci[1]):.2f}) p={m.pvalues[x]:.3g} n={len(d)} ne={int(d.err.sum())}")
    return m


fit(f"err ~ {XV} * l2_units + C(period) + res_median", X, "interaction cites x log2 units", x=f"{XV}:l2_units")
fit(f"err ~ {XV} + {BASE}", X[X.n_units <= 2], "papers with <=2 antibody V regions (single Fab/VHH)")
fit(f"err ~ {XV} + {BASE}", X[X.n_units > 2], "papers with >2 antibody V regions")
# citation source
for src in ["cites_epmc", "cites_cr"]:
    d = X.dropna(subset=[src]).copy()
    d["lc"] = np.log1p(d[src] / d.age_y.clip(lower=0.25))
    fit(f"err ~ lc + {BASE}", d, f"citation source {src} only", x="lc")
fit(f"err ~ cite_pct + {BASE}", X, "within-pub-year percentile (0->1), lab-clustered", x="cite_pct")
X["period3"] = X.period.map({"2018-19": "2018-20", "2020": "2018-20", "2021": "2021", "2022-23": "2022-26", "2024-26": "2022-26"})
for p_, g in X.groupby("period3"):
    fit(f"err ~ {XV} + l2_units + res_median + C(period)", g, f"period {p_}")
# X-ray replication from the investigator's table
D = pd.read_csv(f"{PM}/paper_analysis_table.csv")
D["lab"] = D.last_author.fillna("").str.lower().str.replace(r"[^a-z]", "", regex=True)
D.loc[D.lab == "", "lab"] = D.pkey
XR = D[(D.method == "X-ray") & D.group.isin(["error", "clean"]) & D.published].dropna(subset=["res_median", XV])
fit(f"err ~ {XV} + {BASE}", XR, "X-ray papers (replication)")
ALL = D[D.group.isin(["error", "clean"]) & D.published].dropna(subset=["res_median", XV])
fit(f"err ~ {XV} + C(method) + {BASE}", ALL, "EM + X-ray pooled, method term (paper-method rows)")

V = pd.DataFrame(out)
V.to_csv(f"{OUT}/v06_heterogeneity.csv", index=False)

# ---- collect all verification tests and BH ----
parts = []
for fn, tag in [("v01_tests.csv", "v01"), ("v03_lab_covariates.csv", "v03"), ("v04_correction_bias.csv", "v04"),
                ("v05_superseded.csv", "v05"), ("v06_heterogeneity.csv", "v06")]:
    t = pd.read_csv(f"{OUT}/{fn}")
    t["script"] = tag
    parts.append(t)
A = pd.concat(parts, ignore_index=True)
A = A[~A.test.str.contains("09b|06f", regex=True)]  # summaries of resampling, not single tests
ok = A.p.notna()
A["q_bh"] = np.nan
A.loc[ok, "q_bh"] = multipletests(A.loc[ok, "p"], method="fdr_bh")[1]
A.to_csv(f"{OUT}/all_verification_tests_bh.csv", index=False)
print(A[["script", "test", "OR", "lo", "hi", "p", "q_bh", "n", "n_err"]].round(4).to_string())
