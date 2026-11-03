#!/usr/bin/env python3
"""Robustness of the nominal paper-level positives.
(a) base: period + log2 n_units + resolution
(b) + SARS-CoV-2 + viral title + tool paper + CNS + CNS-sister journal
(c) unit level (antibody V region in paper), GEE logit clustered by paper: unit error ~ x + period + resolution + chain type
(d) by period (2018-20, 2021, 2022-26), base covariates."""
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
DEEP = "/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad/deeper"
OUT = "/home/user/SAbR/register_error_analysis/retest/deeper/paper_meta"
D = pd.read_csv(f"{OUT}/paper_analysis_table.csv")
CT = pd.read_csv(f"{OUT}/competition_table.csv")[["pkey", "comp_dep180", "first_in_pdb"]]
D = D.merge(CT, on="pkey", how="left")
D["any_comp"] = np.where(D.comp_dep180.notna(), (D.comp_dep180 > 0).astype(float), np.nan)
D["l_comp"] = np.log2(1 + D.comp_dep180)
D["viral_title"] = D.title_pdb.fillna("").str.contains(r"(?i)virus|viral|virion|capsid|spike|\bEnv\b|hemagglutinin|haemagglutinin|glycoprotein|SARS|HIV|influenza|ebola|lassa|nipah|RSV|dengue|zika|norovirus|rotavirus|polio|enterovirus|parvovirus|phage")
D["period3"] = D.period.map({"2018-19": "2018-20", "2020": "2018-20", "2021": "2021", "2022-23": "2022-26", "2024-26": "2022-26"})
X = D[(D.method == "EM") & D.group.isin(["error", "clean"])].dropna(subset=["res_median"]).copy()
for c in ["sars2", "viral_title", "tool_paper", "j_CNS", "j_sister"]:
    X[c] = X[c].fillna(False).astype(float)

# unit table
c = pd.read_csv(f"{DEEP}/chains.csv.gz", usecols=["pdb", "unit", "ctype", "err_first", "method", "res"])
B = pd.read_csv(f"{OUT}/paper_base.csv")
pk = {p: k for k, s in zip(B[B.method == "EM"].pkey, B[B.method == "EM"].pdbs) for p in s.split(";")}
c = c[c.method == "EM"].copy()
c["pkey"] = c.pdb.map(pk)
c = c.dropna(subset=["pkey"])
U = c.groupby(["pkey", "unit"]).agg(uerr=("err_first", "any"), ctype=("ctype", "first"), ures=("res", "median")).reset_index()
U["uerr"] = U.uerr.astype(int)

FEATS = [("l_cites_per_year", "log citations/yr"), ("cite_pct", "citation percentile"), ("l_authors", "log2 authors"),
         ("fund_DFG", "DFG funding"), ("preprint_lt90d_after_dep", "preprint within 90 d of deposition"),
         ("any_comp", "any same-antigen paper by another lab deposited within 180 d"),
         ("l_comp", "log2(1+same-antigen competitor papers within 180 d)"), ("first_in_pdb", "first structure of antigen in PDB"),
         ("tool_paper", "tool paper"), ("published", "published")]
BASE = "C(period) + l2_units + res_median"
FULL = BASE + " + sars2 + viral_title + tool_paper + j_CNS + j_sister"
rows = []


def logit(d, x, cov):
    d = d.dropna(subset=[x]).copy()
    d[x] = d[x].astype(float)
    m = smf.logit(f"err ~ {x} + {cov}", d).fit(disp=0, maxiter=200)
    return np.exp(m.params[x]), m.pvalues[x], len(d), int(d.err.sum())


for x, lab in FEATS:
    r = dict(feature=x, label=lab)
    try:
        r["OR_base"], r["p_base"], r["n"], r["n_err"] = logit(X, x, BASE)
    except Exception:
        pass
    try:
        cov = FULL.replace(" + tool_paper", "") if x == "tool_paper" else FULL
        r["OR_full"], r["p_full"], _, _ = logit(X, x, cov)
    except Exception:
        pass
    # unit level
    try:
        u = U.merge(X[["pkey", x, "period"]], on="pkey").dropna(subset=[x, "ures"])
        u[x] = u[x].astype(float)
        g = smf.gee(f"uerr ~ {x} + C(period) + ures + C(ctype)", "pkey", u, family=sm.families.Binomial(),
                    cov_struct=sm.cov_struct.Exchangeable()).fit()
        r["OR_unit_gee"], r["p_unit_gee"], r["n_units"] = np.exp(g.params[x]), g.pvalues[x], len(u)
    except Exception as ex:
        r["unit_err"] = str(ex)[:60]
    for per, d in X.groupby("period3"):
        try:
            o, p, n, ne = logit(d, x, "l2_units + res_median")
            r[f"OR_{per}"], r[f"p_{per}"] = o, p
        except Exception:
            pass
    rows.append(r)
R = pd.DataFrame(rows)
R.to_csv(f"{OUT}/tests_sensitivity.csv", index=False)
pd.set_option("display.width", 250)
print(R.round(4).to_string())
