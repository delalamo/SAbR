#!/usr/bin/env python3
"""Pool every primary test of this lens, apply one Benjamini-Hochberg correction, add heterogeneity tests,
an X-ray secondary check and a browsable list of the error papers."""
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")
OUT = "/home/user/SAbR/register_error_analysis/retest/deeper/paper_meta"
D = pd.read_csv(f"{OUT}/paper_analysis_table.csv")
COV = "C(period) + l2_units + res_median"
rows = []
t = pd.read_csv(f"{OUT}/tests_em_papers.csv")
for _, r in t.iterrows():
    rows.append(dict(block="paper-level EM (" + r.set + ")", test=r.feature, label=r.label, n_err=r.n_err, n_ctrl=r.n_ctrl,
                     err_value=r.err_value, ctrl_value=r.ctrl_value, effect=r.adj_OR, ci_lo=r.ci_lo, ci_hi=r.ci_hi,
                     effect_type="adjusted OR (period, log2 n_units, resolution)", p=r.p_adj))
t = pd.read_csv(f"{OUT}/tests_competition.csv")
for _, r in t[t.covariates == COV].iterrows():
    rows.append(dict(block="antigen competition (PDB-wide)", test=r.feature, label=r.feature, n_err=r.n_err, n_ctrl=r.n_ctrl,
                     err_value=f"mean {r.err_mean:.2f}, median {r.err_median:.2f}", ctrl_value=f"mean {r.ctrl_mean:.2f}, median {r.ctrl_median:.2f}",
                     effect=r.adj_OR, ci_lo=r.ci_lo, ci_hi=r.ci_hi, effect_type="adjusted OR (period, log2 n_units, resolution)", p=r.p))
t = pd.read_csv(f"{OUT}/tests_within_paper.csv")
for _, r in t.iterrows():
    rows.append(dict(block="within-paper (conditional logit)", test=r.feature, label=r.label, n_err=np.nan, n_ctrl=np.nan,
                     err_value=f"{r.err_mean:.3g}", ctrl_value=f"{r.ctrl_mean:.3g}", effect=r.cond_OR, ci_lo=r.ci_lo, ci_hi=r.ci_hi,
                     effect_type=f"conditional OR ({r.n_papers} discordant papers, {r.n_entries} entries)", p=r.p))
t = pd.read_csv(f"{OUT}/network_global.csv")
for _, r in t.iterrows():
    rows.append(dict(block="co-authorship network", test=r.test, label=r.test, n_err=r.n_error, n_ctrl=r.n_papers - r.n_error,
                     err_value=f"observed {r.observed}", ctrl_value=f"null mean {r.null_mean:.1f}", effect=r.ratio,
                     effect_type="observed/expected (stratified permutation)", p=r.p_perm))
t = pd.read_csv(f"{OUT}/lab_dispersion_paper_level.csv")
for _, r in t.iterrows():
    rows.append(dict(block="lab clustering (paper level)", test="lab dispersion " + r.set, label="dispersion of error papers across labs",
                     n_err=r.n_err, n_ctrl=r.n_papers - r.n_err, err_value=f"{r.dispersion_obs:.2f}", ctrl_value=f"null {r.dispersion_null:.2f}",
                     effect=r.ratio, effect_type="observed/expected dispersion", p=r.p_perm))
# heterogeneity tests
EMP = D[(D.method == "EM") & D.group.isin(["error", "clean"]) & D.published].dropna(subset=["res_median"]).copy()
jj = EMP.journal_n.str.replace(r"(?i)^the ", "", regex=True)
top = jj.value_counts()
EMP["jgroup"] = np.where(jj.isin(top[top >= 15].index), jj, "other")
for var, lab in [("jgroup", "journal (19 groups)"), ("region_last", "last-author region (USA/China/Europe/other)"),
                 ("preprint_server", "preprint server")]:
    d = EMP.dropna(subset=[var]) if var != "preprint_server" else EMP[EMP.has_preprint == True].assign(preprint_server=lambda x: x.preprint_server.fillna("unknown").where(x.preprint_server.isin(["biorxiv", "Research Square"]), "other"))
    m0 = smf.logit(f"err ~ {COV}", d).fit(disp=0)
    m1 = smf.logit(f"err ~ C({var}) + {COV}", d).fit(disp=0)
    lr = 2 * (m1.llf - m0.llf)
    rows.append(dict(block="heterogeneity", test=var, label=lab, n_err=int(d.err.sum()), n_ctrl=int((d.err == 0).sum()),
                     effect=lr, effect_type=f"LR chi2 df={m1.df_model - m0.df_model:.0f}", p=stats.chi2.sf(lr, m1.df_model - m0.df_model)))
R = pd.DataFrame(rows)
ok = R.p.notna()
R.loc[ok, "q_BH_all"] = multipletests(R.loc[ok, "p"], method="fdr_bh")[1]
R = R.sort_values("p")
R.to_csv(f"{OUT}/all_tests_bh.csv", index=False)
pd.set_option("display.width", 260, "display.max_colwidth", 55, "display.max_rows", 300)
print(R[["block", "test", "n_err", "n_ctrl", "err_value", "ctrl_value", "effect", "p", "q_BH_all"]].head(30).to_string())
print("tests:", ok.sum(), " q<0.05:", (R.q_BH_all < 0.05).sum(), " q<0.1:", (R.q_BH_all < 0.1).sum())

# X-ray secondary (few features; n error papers small)
XR = D[(D.method == "X-ray") & D.group.isin(["error", "clean"]) & D.published].dropna(subset=["res_median"]).copy()
xr = []
for x in ["l_cites_per_year", "l_authors", "has_preprint", "rec_to_acc", "dep_to_pub", "fund_DFG", "kw_covid"]:
    d = XR.dropna(subset=[x]).copy()
    d[x] = d[x].astype(float)
    try:
        m = smf.logit(f"err ~ {x} + l2_units + res_median + dep_year", d).fit(disp=0)
        xr.append(dict(feature=x, n_err=int(d.err.sum()), n=len(d), err_mean=d[d.err == 1][x].mean(), ctrl_mean=d[d.err == 0][x].mean(),
                       OR=np.exp(m.params[x]), p=m.pvalues[x]))
    except Exception:
        pass
XRt = pd.DataFrame(xr)
XRt.to_csv(f"{OUT}/tests_xray_secondary.csv", index=False)
print(XRt.round(3).to_string())

# error-paper listing
CT = pd.read_csv(f"{OUT}/competition_table.csv")[["pkey", "comp_dep180", "first_in_pdb"]]
L = D[(D.method == "EM") & (D.group.isin(["error", "amyloid_artifact"]))].merge(CT, on="pkey", how="left")
cols = ["group", "pkey", "journal_n", "pub_date", "dep_first", "last_author", "last_country", "n_authors", "n_entries",
        "n_entries_pdb_total", "n_units", "n_err_units", "err_pdbs", "modes", "res_median", "sars2", "tool_paper",
        "has_preprint", "preprint_date", "pre_to_pub", "dep_to_pub", "rec_to_acc", "dep_before_preprint", "cites",
        "cite_pct", "comp_dep180", "first_in_pdb", "title"]
L[cols].sort_values(["group", "dep_first"]).to_csv(f"{OUT}/error_papers.csv", index=False)
print(len(L), "error/artifact papers listed")
