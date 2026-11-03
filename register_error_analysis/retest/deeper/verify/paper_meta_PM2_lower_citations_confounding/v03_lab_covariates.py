#!/usr/bin/env python3
"""Lab-level confounders of the citation association.

v01 showed the citation effect is between-lab (Mundlak between OR 0.61, within-lab clogit 0.88 n.s.).
Here: does lab output / seniority (from the PDB-wide entry index, last-author position) or other
paper-profile variables (region, open access, preprint, countries, authors) absorb it?
Lab output = PDB entries (any method) with the same last author deposited BEFORE the paper's first EM deposition.
"""
import re, warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.discrete.conditional_models import ConditionalLogit

warnings.filterwarnings("ignore")
OUT = "/home/user/SAbR/register_error_analysis/retest/deeper/verify/paper_meta_PM2_lower_citations_confounding"
X = pd.read_csv(f"{OUT}/v01_analysis_frame.csv.gz", parse_dates=["dep_first", "pub_date"])


def norm(a):
    a = a.strip().lower()
    sur = a.split(",")[0]
    ini = re.sub(r"[^a-z]", "", a.split(",")[1])[:1] if "," in a else ""
    return re.sub(r"[^a-z]", "", sur) + "_" + ini


rows = []
with open("/tmp/supp/pdbwide/entries.idx", encoding="latin-1") as fh:
    for i, line in enumerate(fh):
        if i < 2:
            continue
        f = line.rstrip("\n").split("\t")
        if len(f) < 6 or not f[5].strip():
            continue
        auth = re.split(r"(?<=\.),\s+", f[5].strip())
        try:
            dt = pd.to_datetime(f[2], format="%m/%d/%y")
        except Exception:
            continue
        rows.append((norm(auth[-1]), dt, len(auth)))
E = pd.DataFrame(rows, columns=["la", "dep", "nauth"])
E = E.sort_values("dep")
print("PDB entries parsed", len(E))
X["la"] = X.last_author.fillna("").map(lambda s: norm(s) if s else "")
grp = {k: g.dep.values for k, g in E.groupby("la")}
prior, first = [], []
for la, d0 in zip(X.la, X.dep_first):
    v = grp.get(la)
    if v is None or la == "":
        prior.append(0); first.append(np.nan)
    else:
        prior.append(int((v < np.datetime64(d0)).sum())); first.append(v.min())
X["lab_prior_entries"] = prior
X["l2_lab_prior"] = np.log2(1 + X.lab_prior_entries)
X["lab_seniority_y"] = (X.dep_first - pd.to_datetime(pd.Series(first))).dt.days / 365.25
X["lab_seniority_y"] = X.lab_seniority_y.clip(lower=0).fillna(0)
X["has_preprint"] = X.has_preprint.fillna(False).astype(float)
X["open_access"] = X.open_access.fillna(False).astype(float)
X["region"] = X.region_last.fillna("unknown")
print(X.groupby("err")[["lab_prior_entries", "lab_seniority_y"]].median())
print("spearman cites vs lab prior", X[["l_cites_per_year", "l2_lab_prior", "lab_seniority_y"]].corr("spearman"))

XV = "l_cites_per_year"
BASE = "C(period) + l2_units + res_median"
out = []


def fit(f, d, name, x=XV, cluster=True):
    m = smf.logit(f, d).fit(disp=0, maxiter=300, cov_type="cluster", cov_kwds={"groups": pd.factorize(d.lab)[0]}) if cluster \
        else smf.logit(f, d).fit(disp=0, maxiter=300)
    ci = m.conf_int().loc[x]
    out.append(dict(test=name, x=x, OR=np.exp(m.params[x]), lo=np.exp(ci[0]), hi=np.exp(ci[1]), p=m.pvalues[x], n=len(d), n_err=int(d.err.sum())))
    print(f"{name:60s} {x:18s} OR={np.exp(m.params[x]):.3f} ({np.exp(ci[0]):.2f}-{np.exp(ci[1]):.2f}) p={m.pvalues[x]:.3g}")
    return m


fit(f"err ~ l2_lab_prior + {BASE}", X, "lab prior entries alone", x="l2_lab_prior")
fit(f"err ~ lab_seniority_y + {BASE}", X, "lab seniority alone", x="lab_seniority_y")
fit(f"err ~ {XV} + {BASE}", X, "base (cluster lab)")
m = fit(f"err ~ {XV} + l2_lab_prior + lab_seniority_y + {BASE}", X, "+ lab prior entries + seniority")
fit(f"err ~ {XV} + C(region) + open_access + has_preprint + n_countries + l_authors + {BASE}", X,
    "+ region, OA, preprint, countries, authors")
fit(f"err ~ {XV} + l2_lab_prior + lab_seniority_y + C(region) + open_access + has_preprint + n_countries + l_authors"
    f" + sars2 + viral_title + tool_paper + j_CNS + j_sister + any_comp + {BASE}", X.dropna(subset=["any_comp", "n_countries"]),
    "kitchen sink (lab, profile, antigen, journal, competition)")
# lab-mean of citations (between-lab) after lab covariates
X["lab_mean"] = X.groupby("lab")[XV].transform("mean")
X["within"] = X[XV] - X.lab_mean
fit(f"err ~ within + lab_mean + l2_lab_prior + lab_seniority_y + {BASE}", X, "Mundlak + lab covariates", x="lab_mean")
fit(f"err ~ within + lab_mean + l2_lab_prior + lab_seniority_y + {BASE}", X, "Mundlak + lab covariates", x="within")
# within-lab conditional logit restricted to labs with >=3 papers
dl = X[X.groupby("lab").err.transform("nunique") == 2]
cm = ConditionalLogit(dl.err, dl[[XV, "l2_units", "res_median"]], groups=pd.factorize(dl.lab)[0]).fit(disp=0)
ci = cm.conf_int().loc[XV]
out.append(dict(test="within-lab clogit (labs with both outcomes)", x=XV, OR=np.exp(cm.params[XV]), lo=np.exp(ci[0]), hi=np.exp(ci[1]),
                p=cm.pvalues[XV], n=len(dl), n_err=int(dl.err.sum())))
# within-lab citation rank: is the error paper the lab's least cited?
dl = dl.copy()
dl["rk"] = dl.groupby("lab")[XV].rank(pct=True)
e = dl[dl.err == 1]
print("within-lab mean pct rank of error papers", e.rk.mean(), "vs clean", dl[dl.err == 0].rk.mean())
# the papers per lab in single-paper labs vs multi
X["single_lab"] = (X.groupby("lab").err.transform("size") == 1).astype(int)
for s, g in X.groupby("single_lab"):
    try:
        fit(f"err ~ {XV} + {BASE}", g, f"labs with {'1 paper' if s else '>=2 papers'}", cluster=(s == 0))
    except Exception as ex:
        print(ex)
pd.DataFrame(out).to_csv(f"{OUT}/v03_lab_covariates.csv", index=False)
X[["pkey", "lab", "la", "lab_prior_entries", "lab_seniority_y"]].to_csv(f"{OUT}/v03_lab_table.csv.gz", index=False)
