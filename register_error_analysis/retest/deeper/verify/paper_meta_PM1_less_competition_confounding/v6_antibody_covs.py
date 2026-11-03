#!/usr/bin/env python3
"""PM1 confounding, part 5: antibody-type covariates (nanobody, mouse, germline identity, atypical CDR3) and
non-linear resolution; then one BH correction over all tests in this verification folder (tests_all_bh.csv)."""
import glob, warnings
import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings("ignore")
OUT = "/home/user/SAbR/register_error_analysis/retest/deeper/verify/paper_meta_PM1_less_competition_confounding"
T = pd.read_csv(f"{OUT}/paper_table_model2.csv", parse_dates=["dep_first"])
T["vid_min"] = T.vid_min.fillna(T.vid_min.median())
B = "C(period) + l2_units + res_median"
PARS = B + " + cap + enz + fid_only + sars2"
R = []
def fit(formula, d, var, label):
    d = d.dropna(subset=[var])
    m = smf.glm(formula, d, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d["lab"])[0]})
    ci = np.exp(m.conf_int().loc[var])
    return dict(test=label, subset="all EM papers", var=var, n=len(d), n_err=int(d.err.sum()), OR=np.exp(m.params[var]), lo=ci[0], hi=ci[1], p=m.pvalues[var])
print(T.groupby("any_comp")[["any_vhh_y" if "any_vhh_y" in T else "any_vhh", "any_mouse_y" if "any_mouse_y" in T else "any_mouse", "vid_min"]].mean())
vh = "any_vhh_y" if "any_vhh_y" in T else "any_vhh"; mo = "any_mouse_y" if "any_mouse_y" in T else "any_mouse"
T["avhh"] = T[vh].astype(float); T["amouse"] = T[mo].astype(float)
R.append(fit(f"err ~ any_comp + {B} + avhh + amouse + vid_min", T, "any_comp", "base + nanobody + mouse + min germline identity"))
R.append(fit(f"err ~ any_comp + {PARS} + avhh + amouse + vid_min", T, "any_comp", "parsimonious + nanobody + mouse + min germline identity"))
R.append(fit(f"err ~ any_comp + C(period) + l2_units + bs(res_median, df=4) + cap + enz + fid_only + sars2", T, "any_comp", "parsimonious with resolution spline"))
R.append(fit(f"err ~ any_comp + C(period) + l2_units + res_median + res_min_v + cap + enz + fid_only + sars2", T, "any_comp", "parsimonious + best resolution"))
Rt = pd.DataFrame(R); Rt.to_csv(f"{OUT}/tests_confounding_5.csv", index=False)
print(Rt.round(4).to_string())
A = []
for f in sorted(glob.glob(f"{OUT}/tests_confounding*.csv")):
    d = pd.read_csv(f); d["file"] = f.split("/")[-1]; A.append(d)
A = pd.concat(A, ignore_index=True)
A = A[A.p.notna() & ~A.test.astype(str).str.startswith("leave-one")]
A["q_BH_all_verify"] = multipletests(A.p, method="fdr_bh")[1]
A.to_csv(f"{OUT}/tests_all_bh.csv", index=False)
print(len(A), "tests in BH")
pd.set_option("display.width", 250, "display.max_colwidth", 90)
key = A[A.test.str.contains("reproduce|parsimonious \\(|FULL \\(|Mantel|unit level, parsimonious \\(cluster lab|30% cluster, (base|parsimonious)$", regex=True)]
print(key[["file", "test", "subset", "var", "n", "n_err", "OR", "lo", "hi", "p", "q_BH_all_verify"]].round(4).to_string())
