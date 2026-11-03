#!/usr/bin/env python3
"""Within-paper comparison (controls for lab, paper, journal, period): in cryo-EM papers that contain both
erroneous (err_first) and clean EM entries, are the erroneous entries the ones deposited later (e.g. added
during revision), at worse resolution, with shorter holds, or released before the paper?
Conditional logistic regression, strata = paper."""
import numpy as np
import pandas as pd
from statsmodels.discrete.conditional_models import ConditionalLogit
from scipy import stats

DEEP = "/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad/deeper"
OUT = "/home/user/SAbR/register_error_analysis/retest/deeper/paper_meta"
e = pd.read_csv(f"{DEEP}/entries.csv", parse_dates=["deposit", "release"])
M = pd.read_csv(f"{OUT}/paper_meta.csv", parse_dates=["received", "accepted", "pub_date"])
B = pd.read_csv(f"{OUT}/paper_base.csv")
amy = set(B[B.group == "amyloid_artifact"].pkey)
e = e[(e.method == "EM") & (e.deposit >= "2018-01-01") & e.doi.notna()].copy()
e["doi"] = e.doi.str.strip().str.lower()
e = e[~e.doi.isin(amy)]
e = e.merge(M[["doi", "received", "accepted", "pub_date"]], on="doi", how="left")
g = e.groupby("doi")
e["dep_offset"] = (e.deposit - g.deposit.transform("min")).dt.days
e["dep_rank_last"] = (e.deposit == g.deposit.transform("max")) & (g.deposit.transform("nunique") > 1)
e["after_submission"] = np.where(e.received.notna(), e.deposit > e.received + pd.Timedelta(days=7), np.nan)
e["hold"] = (e.release - e.deposit).dt.days
e["rel_before_pub"] = np.where(e.pub_date.notna(), e.release < e.pub_date - pd.Timedelta(days=7), np.nan)
e["err"] = e.err_first.astype(int)
e["n_v"] = e.n_V_chains
# discordant papers only
disc = e.groupby("doi").err.agg(["min", "max"])
disc = disc[(disc["min"] == 0) & (disc["max"] == 1)].index
d = e[e.doi.isin(disc)].copy()
print("discordant papers:", len(disc), "entries:", len(d), "error entries:", d.err.sum())
rows = []
for v, scale, lab in [("dep_offset", 30, "days after the paper's first deposition (per 30 d)"),
                      ("dep_rank_last", 1, "in the paper's last deposition batch"),
                      ("after_submission", 1, "deposited >7 d after submission"),
                      ("res", 0.5, "resolution (per 0.5 A)"),
                      ("hold", 30, "hold time (per 30 d)"),
                      ("rel_before_pub", 1, "released before the paper"),
                      ("n_v", 1, "V chains in entry (per chain)")]:
    x = d.dropna(subset=[v]).copy()
    x[v] = x[v].astype(float) / scale
    x = x[x.groupby("doi").err.transform("nunique") == 2]
    if x[v].nunique() < 2 or x.doi.nunique() < 5:
        continue
    try:
        m = ConditionalLogit(x.err, x[[v]], groups=x.doi).fit(disp=0)
        orr, p = float(np.exp(m.params[v])), float(m.pvalues[v])
        ci = np.exp(m.conf_int().loc[v]).tolist()
    except Exception as ex:
        orr, p, ci = np.nan, np.nan, [np.nan, np.nan]
    rows.append(dict(feature=v, label=lab, n_papers=x.doi.nunique(), n_entries=len(x),
                     err_mean=float(x[x.err == 1][v].mean() * scale), ctrl_mean=float(x[x.err == 0][v].mean() * scale),
                     cond_OR=orr, ci_lo=ci[0], ci_hi=ci[1], p=p))
R = pd.DataFrame(rows)
R.to_csv(f"{OUT}/tests_within_paper.csv", index=False)
pd.set_option("display.width", 250)
print(R.to_string())
