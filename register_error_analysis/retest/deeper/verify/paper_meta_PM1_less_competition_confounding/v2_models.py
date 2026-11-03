#!/usr/bin/env python3
"""Confounding checks for PM1 (error papers have fewer contemporaneous same-antigen competitors).
Paper level, cryo-EM, error vs clean papers (paper_meta definitions). Writes tests_confounding.csv."""
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")
OUT = "/home/user/SAbR/register_error_analysis/retest/deeper/verify/paper_meta_PM1_less_competition_confounding"
T = pd.read_csv(f"{OUT}/paper_table_verify.csv", parse_dates=["dep_first"])
T = T.dropna(subset=["res_median", "comp_dep180"]).copy()
T["any_comp"] = (T.comp_dep180 > 0).astype(int)
T["l_comp"] = np.log2(1 + T.comp_dep180)
T["l_acc"] = np.log2(T.n_antigen_acc)
T["lexp"] = np.log2(1 + T.lab_prior_ab.fillna(0))
T["l_novel"] = np.log2(1 + T.n_novel_units.fillna(0))
T["novel_frac"] = T.novel_frac.fillna(T.novel_frac.median())
T["sars2"] = T.sars2.astype(int)
T["t"] = (T.dep_first - pd.Timestamp("2018-01-01")).dt.days / 365.25
T["dyear"] = T.dep_first.dt.year.clip(upper=2025)  # 2026 merged into 2025 (few papers)
T["l_labs_all"] = np.log2(1 + T.all_labs_alltime)
T["l_labs_prior"] = np.log2(1 + T.all_labs_prior)
T["any_comp_before"] = (T.all_n_before > 0).astype(int)
T["any_comp_after"] = (T.all_n_after > 0).astype(int)
T["any_tgt"] = np.where(T.tgt_n.notna(), (T.tgt_n > 0).astype(float), np.nan)
T["l_tgt"] = np.log2(1 + T.tgt_n)
T["lab"] = T.plast.fillna("na")
T["capenz"] = T[["cap", "enz"]].max(axis=1)
T["lqz"] = T.lqz_all
T["lqz_c"] = T.lqz_clean_chains
T.to_csv(f"{OUT}/paper_table_model.csv", index=False)
print("n papers", len(T), "errors", T.err.sum())

BASE = "C(period) + l2_units + res_median"
FINE = "bs(t, df=5) + l2_units + res_median"
SIZE = " + l_acc + l2_entries"
CLS = " + cap + enz + fid_only + unbound_only"
LAB = " + lexp"
NOV = " + novel_frac"
VIR = " + sars2 + viral_org"
FULL = FINE + SIZE + CLS + LAB + NOV + VIR


def fit(formula, d, var, cluster=None, label="", subset="all EM papers"):
    d = d.dropna(subset=[c for c in [var] if c in d])
    try:
        if cluster is None:
            m = smf.logit(formula, d).fit(disp=0, maxiter=200)
        else:
            m = smf.glm(formula, d, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d[cluster])[0]})
        ci = np.exp(m.conf_int().loc[var])
        return dict(test=label, subset=subset, var=var, formula=formula.split("~")[1].strip(), se=("cluster:" + cluster) if cluster else "model",
                    n=len(d), n_err=int(d.err.sum()), err_mean=d[d.err == 1][var].mean(), ctrl_mean=d[d.err == 0][var].mean(),
                    OR=np.exp(m.params[var]), lo=ci[0], hi=ci[1], p=m.pvalues[var])
    except Exception as ex:
        return dict(test=label, subset=subset, var=var, formula=formula, n=len(d), n_err=int(d.err.sum()), OR=np.nan, p=np.nan, note=str(ex)[:80])


R = []
for var in ["any_comp", "l_comp"]:
    R.append(fit(f"err ~ {var} + {BASE}", T, var, label="reproduce (investigator model)"))
    R.append(fit(f"err ~ {var} + {BASE}", T, var, cluster="lab", label="lab-clustered SE"))
    R.append(fit(f"err ~ {var} + C(dyear) + l2_units + res_median", T, var, cluster="lab", label="deposit year categorical"))
    R.append(fit(f"err ~ {var} + {FINE}", T, var, cluster="lab", label="deposit date spline"))
    R.append(fit(f"err ~ {var} + {FINE}{SIZE}", T, var, cluster="lab", label="+ n antigen accessions + n entries"))
    R.append(fit(f"err ~ {var} + {FINE}{CLS}", T, var, cluster="lab", label="+ target class (capsid, enzyme, fiducial-only, unbound)"))
    R.append(fit(f"err ~ {var} + {FINE}{LAB}", T, var, cluster="lab", label="+ lab antibody experience"))
    R.append(fit(f"err ~ {var} + {FINE}{NOV}", T, var, cluster="lab", label="+ share of new V regions"))
    R.append(fit(f"err ~ {var} + {FINE}{VIR}", T, var, cluster="lab", label="+ SARS-CoV-2 + viral target"))
    R.append(fit(f"err ~ {var} + {FULL}", T, var, cluster="lab", label="FULL (spline time, size, class, lab exp, new V, SARS2, viral)"))
    R.append(fit(f"err ~ {var} + {FULL} + lqz", T, var, cluster="lab", label="FULL + paper median local fit lq_z (all V chains; partly circular)"))
    R.append(fit(f"err ~ {var} + {FULL} + lqz_c", T, var, cluster="lab", label="FULL + median lq_z of chains without any run"))
    R.append(fit(f"err ~ {var} + {FULL} + l_labs_all", T, var, cluster="lab", label="FULL + all-time antigen popularity (other labs ever)"))
    R.append(fit(f"err ~ {var} + {FULL} + l_labs_prior", T, var, cluster="lab", label="FULL + prior antigen popularity (other labs >180 d before)"))
# popularity alone, and contemporaneous before vs after
for var, lab in [("l_labs_all", "all-time popularity (other labs ever), alone"), ("l_labs_prior", "prior popularity (>180 d before)"),
                 ("any_comp_before", "competitor deposited in the 180 d BEFORE"), ("any_comp_after", "competitor deposited in the 180 d AFTER"),
                 ("any_tgt", "contacted antigen only (no partner/fiducial accessions)"), ("l_tgt", "contacted antigen only, log2")]:
    R.append(fit(f"err ~ {var} + {BASE}", T, var, label=lab + " [base model]"))
    R.append(fit(f"err ~ {var} + {FULL}", T, var, cluster="lab", label=lab + " [FULL]"))

# subsets (FULL minus terms that are constant in the subset)
subs = {
    "non-SARS-CoV-2": T[T.sars2 == 0],
    "non-SARS2, no capsid/enzyme/fiducial-only": T[(T.sars2 == 0) & (T.capenz == 0) & (T.fid_only == 0)],
    "no capsid/enzyme/fiducial-only (incl SARS2)": T[(T.capenz == 0) & (T.fid_only == 0)],
    "viral target, non-SARS2": T[(T.viral_org == 1) & (T.sars2 == 0)],
    "viral non-SARS2, no capsid": T[(T.viral_org == 1) & (T.sars2 == 0) & (T.cap == 0)],
    "non-viral target": T[T.viral_org == 0],
    "non-viral, no enzyme/fiducial-only": T[(T.viral_org == 0) & (T.enz == 0) & (T.fid_only == 0)],
    "dep 2018-2021": T[T.dep_first.dt.year <= 2021],
    "dep 2022-2026": T[T.dep_first.dt.year >= 2022],
    "dep 2020-2021": T[T.dep_first.dt.year.isin([2020, 2021])],
    "dep 2024-2026": T[T.dep_first.dt.year >= 2024],
    "single-antibody papers (n_units=1)": T[T.n_units == 1],
    "multi-antibody papers": T[T.n_units > 1],
    "papers with lab_prior_ab>=10": T[T.lab_prior_ab >= 10],
    "papers with lab_prior_ab<10": T[T.lab_prior_ab < 10],
}
SUBF = "bs(t, df=4) + l2_units + res_median + l_acc + lexp + novel_frac"
for name, d in subs.items():
    extra = ""
    if d.sars2.nunique() > 1: extra += " + sars2"
    if d.viral_org.nunique() > 1: extra += " + viral_org"
    if d.cap.sum() >= 3 and name.find("capsid") < 0: extra += " + cap"
    if d.enz.sum() >= 3 and name.find("enzyme") < 0: extra += " + enz"
    if d.fid_only.sum() >= 3 and name.find("fiducial") < 0: extra += " + fid_only"
    f = SUBF if d.dep_first.dt.year.nunique() > 3 else "t + l2_units + res_median + l_acc + lexp + novel_frac"
    for var in ["any_comp"]:
        R.append(fit(f"err ~ {var} + {BASE}", d, var, label="base model", subset=name) if d.period.nunique() > 1 else
                 fit(f"err ~ {var} + l2_units + res_median", d, var, label="base model", subset=name))
        R.append(fit(f"err ~ {var} + {f}{extra}", d, var, cluster="lab", label="adjusted", subset=name))

Rt = pd.DataFrame(R)
Rt["q_BH_this_file"] = multipletests(Rt.p.fillna(1), method="fdr_bh")[1]
Rt.to_csv(f"{OUT}/tests_confounding.csv", index=False)
pd.set_option("display.width", 250, "display.max_colwidth", 70, "display.max_rows", 200)
print(Rt[["test", "subset", "var", "n", "n_err", "err_mean", "ctrl_mean", "OR", "lo", "hi", "p"]].round(3).to_string())

# descriptive: covariate balance and correlation with competition
print("\nCovariate means by error status and by any competitor")
cols = ["cap", "enz", "fid_only", "unbound_only", "viral_org", "sars2", "lexp", "novel_frac", "l_acc", "l2_entries", "l2_units", "res_median", "t", "lqz", "lqz_c"]
print(pd.concat([T.groupby("err")[cols].mean().T.add_prefix("err="), T.groupby("any_comp")[cols].mean().T.add_prefix("comp=")], axis=1).round(3))
print("\nerror rate by any_comp x class")
T["cls3"] = np.select([T.sars2 == 1, T.cap == 1, T.enz == 1, T.fid_only == 1, T.viral_org == 1], ["SARS2", "capsid", "enzyme", "fiducial-only", "viral other"], "non-viral other")
print(T.groupby(["cls3", "any_comp"]).err.agg(["size", "sum", "mean"]).round(3))
