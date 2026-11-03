#!/usr/bin/env python3
"""Adversarial confounding check of paper_meta finding PM2 ("error papers are cited less").

Base model reproduced from paper_meta/analysis.py:
  err ~ l_cites_per_year + C(period) + l2_units + res_median   (published cryo-EM papers, error vs clean)
Then: alternative time controls, size specification, journal, SARS-CoV-2/antigen class, lab clustering,
paper-level map fit / modelling covariates, error-definition variants, influence of single labs.
Writes v01_tests.csv.
"""
import re, warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

warnings.filterwarnings("ignore")
PM = "/home/user/SAbR/register_error_analysis/retest/deeper/paper_meta"
OUT = "/home/user/SAbR/register_error_analysis/retest/deeper/verify/paper_meta_PM2_lower_citations_confounding"
DEEP = "/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad/deeper"

D = pd.read_csv(f"{PM}/paper_analysis_table.csv", parse_dates=["pub_date", "dep_first"])
CT = pd.read_csv(f"{PM}/competition_table.csv")[["pkey", "comp_dep180", "first_in_pdb"]]
D = D.merge(CT, on="pkey", how="left")
D["any_comp"] = np.where(D.comp_dep180.notna(), (D.comp_dep180 > 0).astype(float), np.nan)
D["l_comp"] = np.log2(1 + D.comp_dep180)
D["viral_title"] = D.title_pdb.fillna("").str.contains(r"(?i)virus|viral|virion|capsid|spike|\bEnv\b|hemagglutinin|haemagglutinin|glycoprotein|SARS|HIV|influenza|ebola|lassa|nipah|RSV|dengue|zika|norovirus|rotavirus|polio|enterovirus|parvovirus|phage")
for c in ["sars2", "viral_title", "tool_paper", "j_CNS", "j_sister", "any_vhh", "any_mouse"]:
    D[c] = D[c].fillna(False).astype(float)
D["lab"] = D.last_author.fillna("").str.lower().str.replace(r"[^a-z]", "", regex=True)
D.loc[D.lab == "", "lab"] = D.pkey

# paper-level modelling covariates from chains (EM chains of the paper's EM entries)
ch = pd.read_csv(f"{DEEP}/chains.csv.gz", usecols=["pdb", "unit", "ctype", "method", "res", "lq_z", "irm", "fit",
                                                   "err_first", "err_any", "err_modes", "L3_atyp", "L1_atyp", "L2_atyp", "vhh", "v_id"])
B = pd.read_csv(f"{PM}/paper_base.csv")
pk = {p: k for k, s in zip(B[B.method == "EM"].pkey, B[B.method == "EM"].pdbs) for p in s.split(";")}
ch = ch[ch.method == "EM"].copy()
ch["pkey"] = ch.pdb.map(pk)
ch = ch.dropna(subset=["pkey"])
agg = ch.groupby("pkey").agg(lqz_med=("lq_z", "median"), lqz_min=("lq_z", "min"),
                             af_start=("irm", lambda s: (s == "AlphaFold model").mean()),
                             rigid=("fit", lambda s: s.str.contains("rigid", case=False, na=False).mean()),
                             any_atyp=("L3_atyp", "max"), vid_min=("v_id", "min"), frac_vhh=("vhh", "mean")).reset_index()
D = D.merge(agg, on="pkey", how="left")

X = D[(D.method == "EM") & D.group.isin(["error", "clean"]) & D.published].dropna(subset=["res_median", "l_cites_per_year"]).copy()
X["pub_year_c"] = X.pub_year.astype(int).astype(str)
X["pubq"] = (X.pub_date.dt.year + (X.pub_date.dt.quarter - 1) / 4.0)
X["age_y"] = (pd.Timestamp("2026-11-01") - X.pub_date).dt.days / 365.25
X["l_age"] = np.log(X.age_y.clip(lower=0.25))
X["l_cites_raw"] = np.log1p(X.cites)
X["n_units_cat"] = pd.cut(X.n_units, [0, 1, 2, 3, 5, 8, 1000], labels=["1", "2", "3", "4-5", "6-8", "9+"]).astype(str)
X["ln_units"] = np.log(X.n_units.clip(lower=1))
X["l2_pdbtot"] = np.log2(X.n_entries_pdb_total.clip(lower=1))
X["l2_entries"] = np.log2(X.n_entries)
jj = X.journal_n.fillna("").str.replace(r"(?i)^the ", "", regex=True).str.lower()
top = jj.value_counts()
X["jgroup"] = np.where(jj.isin(top[top >= 10].index), jj, "other")
print("n papers", len(X), "error", int(X.err.sum()))

rows = []
XV = "l_cites_per_year"


def rec(name, model_desc, m, x=XV, n=None, ne=None, note=""):
    b = m.params[x]
    ci = m.conf_int().loc[x]
    rows.append(dict(test=name, model=model_desc, OR=float(np.exp(b)), lo=float(np.exp(ci[0])), hi=float(np.exp(ci[1])),
                     p=float(m.pvalues[x]), n=n, n_err=ne, note=note))
    print(f"{name:55s} OR={np.exp(b):.3f} ({np.exp(ci[0]):.2f}-{np.exp(ci[1]):.2f}) p={m.pvalues[x]:.3g} n={n} ne={ne} {note}")


def logit(d, f, x=XV, cluster=None, name=None, desc=None, note=""):
    d = d.copy()
    if cluster is not None:
        m = smf.logit(f, d).fit(disp=0, maxiter=300, cov_type="cluster", cov_kwds={"groups": pd.factorize(d[cluster])[0]})
    else:
        m = smf.logit(f, d).fit(disp=0, maxiter=300)
    rec(name, desc or f, m, x, len(d), int(d.err.sum()), note)
    return m


BASE = "C(period) + l2_units + res_median"
# ---------- 0. reproduction ----------
logit(X, f"err ~ {XV} + {BASE}", name="00 reproduce base")
mw = stats.mannwhitneyu(X[X.err == 1][XV], X[X.err == 0][XV])
print("crude MW p", mw.pvalue)
rows.append(dict(test="00b crude Mann-Whitney", model="MW", p=mw.pvalue, n=len(X), n_err=int(X.err.sum()),
                 note=f"median err {X[X.err==1].cites.div(X[X.err==1].age_y.clip(lower=.25)).median():.2f} vs ctrl {X[X.err==0].cites.div(X[X.err==0].age_y.clip(lower=.25)).median():.2f} cites/yr"))
logit(X, f"err ~ {XV}", name="00c crude logit")
logit(X, f"err ~ {XV} + C(period)", name="00d period only")
logit(X, f"err ~ {XV} + l2_units", name="00e size only")
logit(X, f"err ~ {XV} + C(period) + res_median", name="00f period + res (no size)")

# ---------- 1. time control ----------
logit(X, f"err ~ {XV} + C(pub_year_c) + l2_units + res_median", name="01a pub-year FE instead of dep period")
logit(X, f"err ~ {XV} + C(period) + C(pub_year_c) + l2_units + res_median", name="01b dep period + pub-year FE")
logit(X, f"err ~ {XV} + bs(pubq, df=6) + l2_units + res_median", name="01c spline in pub date (df6)")
logit(X, f"err ~ l_cites_raw + l_age + C(period) + l2_units + res_median", x="l_cites_raw", name="01d raw log cites + log age")
logit(X, f"err ~ {XV} + C(dep_year) + l2_units + res_median", name="01e dep-year FE")

# ---------- 2. size specification ----------
logit(X, f"err ~ {XV} + C(period) + C(n_units_cat) + res_median", name="02a size as categories")
logit(X, f"err ~ {XV} + C(period) + l2_units + I(l2_units**2) + res_median", name="02b size quadratic")
logit(X, f"err ~ {XV} + C(period) + l2_units + l2_entries + l2_pdbtot + res_median", name="02c + n EM entries + n PDB entries")
logit(X, f"err ~ {XV} + C(period) + l2_units + l_authors + res_median", name="02d + log2 authors")
# cloglog with offset log(n_units): constant per-antibody hazard
d = X.copy()
m = smf.glm(f"err ~ {XV} + C(period) + res_median", d, family=sm.families.Binomial(link=sm.families.links.CLogLog()),
            offset=d.ln_units).fit()
rows.append(dict(test="02e cloglog offset log(n_units)", model="cloglog", OR=float(np.exp(m.params[XV])),
                 lo=float(np.exp(m.conf_int().loc[XV][0])), hi=float(np.exp(m.conf_int().loc[XV][1])), p=float(m.pvalues[XV]),
                 n=len(d), n_err=int(d.err.sum()), note="exp(beta) = hazard ratio per antibody"))
print("02e cloglog offset", np.exp(m.params[XV]), m.pvalues[XV])
# binomial count of error units out of units (per-antibody rate), cluster by paper (quasi-binomial)
d = X.copy()
d["n_ok"] = d.n_units - d.n_err_units
m = smf.glm(f"n_err_units + n_ok ~ {XV} + C(period) + res_median", d, family=sm.families.Binomial()).fit(scale="X2")
rows.append(dict(test="02f quasi-binomial per-antibody rate", model="glm binomial units, scale X2", OR=float(np.exp(m.params[XV])),
                 lo=float(np.exp(m.conf_int().loc[XV][0])), hi=float(np.exp(m.conf_int().loc[XV][1])), p=float(m.pvalues[XV]),
                 n=len(d), n_err=int(d.err.sum())))
print("02f quasi-binomial", np.exp(m.params[XV]), m.pvalues[XV])
# stratified comparison within exact n_units strata (van Elteren-like: stratified rank)
d = X.copy()
d["rk"] = d.groupby("n_units_cat")[XV].rank(pct=True)
m = smf.logit(f"err ~ rk + C(n_units_cat) + C(period) + res_median", d).fit(disp=0)
rows.append(dict(test="02g within-size-stratum citation rank", model="rank within n_units cat", OR=float(np.exp(m.params['rk'])),
                 lo=float(np.exp(m.conf_int().loc['rk'][0])), hi=float(np.exp(m.conf_int().loc['rk'][1])), p=float(m.pvalues['rk']),
                 n=len(d), n_err=int(d.err.sum()), note="OR for rank 0->1"))
print("02g within size rank", np.exp(m.params['rk']), m.pvalues['rk'])
for cat, g in X.groupby("n_units_cat"):
    e, c = g[g.err == 1][XV], g[g.err == 0][XV]
    if len(e) >= 3:
        p = stats.mannwhitneyu(e, c).pvalue
        rows.append(dict(test=f"02h size stratum n_units={cat}", model="MW", p=p, n=len(g), n_err=len(e),
                         note=f"median l_cpy err {e.median():.2f} vs {c.median():.2f}; err rate {len(e)/len(g):.2f}"))
        print(f"   n_units={cat}: n={len(g)} err={len(e)} median {e.median():.2f} vs {c.median():.2f} p={p:.3g}")

# ---------- 3. journal ----------
logit(X, f"err ~ {XV} + C(jgroup) + {BASE}", name="03a journal FE (>=10 papers)")
# conditional logit stratified by journal
from statsmodels.discrete.conditional_models import ConditionalLogit
d = X[X.jgroup != "other"].copy()
d = d[d.groupby("jgroup").err.transform("nunique") == 2]
cm = ConditionalLogit(d.err, d[[XV, "l2_units", "res_median"]].assign(**{f"p_{k}": (d.period == k).astype(float) for k in sorted(d.period.unique())[1:]}),
                      groups=pd.factorize(d.jgroup)[0]).fit(disp=0)
rows.append(dict(test="03b conditional logit within journal", model="clogit strata=journal", OR=float(np.exp(cm.params[XV])),
                 lo=float(np.exp(cm.conf_int().loc[XV][0])), hi=float(np.exp(cm.conf_int().loc[XV][1])),
                 p=float(cm.pvalues[XV]), n=len(d), n_err=int(d.err.sum())))
print("03b clogit journal", np.exp(cm.params[XV]), cm.pvalues[XV], len(d), d.err.sum())
# within journal x pub-year citation percentile
d = X.copy()
d["jy"] = d.jgroup + "_" + d.pub_year_c
d["cpct_jy"] = d.groupby("jy")[XV].rank(pct=True)
d = d[d.groupby("jy")[XV].transform("size") >= 5]
m = smf.logit(f"err ~ cpct_jy + {BASE}", d).fit(disp=0)
rows.append(dict(test="03c citation pct within journal x pub-year (cells>=5)", model="logit", OR=float(np.exp(m.params['cpct_jy'])),
                 lo=float(np.exp(m.conf_int().loc['cpct_jy'][0])), hi=float(np.exp(m.conf_int().loc['cpct_jy'][1])),
                 p=float(m.pvalues['cpct_jy']), n=len(d), n_err=int(d.err.sum()), note="OR for pct 0->1"))
print("03c pct within journal-year", np.exp(m.params['cpct_jy']), m.pvalues['cpct_jy'], len(d))

# ---------- 4. SARS-CoV-2 / antigen / paper type ----------
FULL = BASE + " + sars2 + viral_title + tool_paper + j_CNS + j_sister"
logit(X, f"err ~ {XV} + {FULL}", name="04a full covariates (investigator)")
for lab_, g in [("SARS-CoV-2 papers", X[X.sars2 == 1]), ("non-SARS-CoV-2 papers", X[X.sars2 == 0]),
                ("viral non-SARS2", X[(X.sars2 == 0) & (X.viral_title == 1)]), ("non-viral", X[(X.sars2 == 0) & (X.viral_title == 0)]),
                ("tool papers", X[X.tool_paper == 1]), ("non-tool papers", X[X.tool_paper == 0])]:
    try:
        logit(g, f"err ~ {XV} + {BASE}", name=f"04 stratum: {lab_}")
    except Exception as ex:
        print(lab_, ex)
logit(X, f"err ~ {XV} * sars2 + {BASE}", name="04g interaction model (main effect = non-SARS2)")
logit(X, f"err ~ {XV} + {FULL} + any_vhh + any_mouse", name="04h + nanobody, mouse")

# ---------- 5. competition / authors / novelty ----------
d = X.dropna(subset=["any_comp", "first_in_pdb"])
logit(d, f"err ~ {XV} + {BASE}", name="05a base on competition subset")
logit(d, f"err ~ {XV} + any_comp + {BASE}", name="05b + any competitor")
logit(d, f"err ~ {XV} + l_comp + first_in_pdb + {BASE}", name="05c + log competitors + first in PDB")
logit(d, f"err ~ {XV} + any_comp + first_in_pdb + l_authors + {FULL}", name="05d + comp + first + authors + full")

# ---------- 6. clustering by lab ----------
logit(X, f"err ~ {XV} + {BASE}", cluster="lab", name="06a base, cluster-robust by last author")
logit(X, f"err ~ {XV} + {FULL}", cluster="lab", name="06b full, cluster-robust by last author")
d = X.copy()
g = smf.gee(f"err ~ {XV} + {BASE}", "lab", d, family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()
rows.append(dict(test="06c GEE exchangeable by lab (paper level)", model="GEE", OR=float(np.exp(g.params[XV])),
                 lo=float(np.exp(g.conf_int().loc[XV][0])), hi=float(np.exp(g.conf_int().loc[XV][1])), p=float(g.pvalues[XV]),
                 n=len(d), n_err=int(d.err.sum())))
print("06c GEE lab", np.exp(g.params[XV]), g.pvalues[XV])
# lab-mean centering: within-lab vs between-lab citation component
d["lab_mean"] = d.groupby("lab")[XV].transform("mean")
d["within"] = d[XV] - d.lab_mean
d["lab_n"] = d.groupby("lab")[XV].transform("size")
m = smf.logit(f"err ~ within + lab_mean + {BASE}", d).fit(disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(d.lab)[0]})
for v in ["within", "lab_mean"]:
    rows.append(dict(test=f"06d Mundlak split: {v}", model="logit cluster lab", OR=float(np.exp(m.params[v])),
                     lo=float(np.exp(m.conf_int().loc[v][0])), hi=float(np.exp(m.conf_int().loc[v][1])), p=float(m.pvalues[v]),
                     n=len(d), n_err=int(d.err.sum())))
    print("06d", v, np.exp(m.params[v]), m.pvalues[v])
# conditional logit within lab (labs with both outcomes)
dl = d[d.groupby("lab").err.transform("nunique") == 2]
cm = ConditionalLogit(dl.err, dl[[XV, "l2_units", "res_median"]], groups=pd.factorize(dl.lab)[0]).fit(disp=0)
rows.append(dict(test="06e conditional logit within lab", model="clogit strata=last author", OR=float(np.exp(cm.params[XV])),
                 lo=float(np.exp(cm.conf_int().loc[XV][0])), hi=float(np.exp(cm.conf_int().loc[XV][1])),
                 p=float(cm.pvalues[XV]), n=len(dl), n_err=int(dl.err.sum()), note=f"{dl.lab.nunique()} labs with both outcomes"))
print("06e clogit within lab", np.exp(cm.params[XV]), cm.pvalues[XV], len(dl), dl.lab.nunique())
# one paper per lab (random pick, repeated)
rng = np.random.default_rng(0)
ors = []
for it in range(200):
    s = d.groupby("lab", group_keys=False).apply(lambda g: g.sample(1, random_state=int(rng.integers(1e9))))
    try:
        mm = smf.logit(f"err ~ {XV} + {BASE}", s).fit(disp=0)
        ors.append((np.exp(mm.params[XV]), mm.pvalues[XV]))
    except Exception:
        pass
ors = np.array(ors)
rows.append(dict(test="06f one random paper per lab (200 draws)", model="logit base", OR=float(np.median(ors[:, 0])),
                 lo=float(np.quantile(ors[:, 0], .025)), hi=float(np.quantile(ors[:, 0], .975)), p=float(np.median(ors[:, 1])),
                 n=int(d.lab.nunique()), note=f"frac draws p<0.05: {(ors[:,1]<0.05).mean():.2f}; CI = 2.5-97.5% of draw ORs; p = median"))
print("06f one per lab", np.median(ors[:, 0]), np.median(ors[:, 1]), (ors[:, 1] < .05).mean())

# ---------- 7. map fit / modelling ----------
d = X.dropna(subset=["lqz_med"])
logit(d, f"err ~ {XV} + {BASE}", name="07a base on lq subset")
logit(d, f"err ~ {XV} + lqz_med + {BASE}", name="07b + paper median map fit lq_z")
logit(d, f"err ~ {XV} + lqz_min + {BASE}", name="07c + paper min map fit lq_z")
logit(d, f"err ~ {XV} + lqz_min + af_start + rigid + {BASE}", name="07d + min lq_z + AF start + rigid fit")
logit(d.dropna(subset=["vid_min"]), f"err ~ {XV} + lqz_min + af_start + rigid + any_atyp + vid_min + frac_vhh + {BASE}",
      name="07e + atypical L3 + min germline id + nanobody")
# does citation correlate with map fit? (paper level)
r = stats.spearmanr(d[XV], d.lqz_med)
rows.append(dict(test="07f spearman cites/yr vs paper median lq_z", model="spearman", OR=np.nan, p=r.pvalue, n=len(d), note=f"rho={r.correlation:.3f}"))
print("07f", r)

# ---------- 8. error definition ----------
Y = D[(D.method == "EM") & D.group.isin(["error", "clean", "copied_only"]) & D.published].dropna(subset=["res_median", "l_cites_per_year"]).copy()
Y["err"] = (Y.group == "error").astype(int)
logit(Y, f"err ~ {XV} + {BASE}", name="08a copied-only papers counted as controls")
Y["err"] = Y.group.isin(["error", "copied_only"]).astype(int)
logit(Y, f"err ~ {XV} + {BASE}", name="08b any run (incl. copied/artifact) as outcome")
Z = D[(D.method == "EM") & D.group.isin(["error", "clean", "amyloid_artifact"]) & D.published].dropna(subset=["res_median", "l_cites_per_year"]).copy()
Z["err"] = Z.group.isin(["error", "amyloid_artifact"]).astype(int)
logit(Z, f"err ~ {XV} + {BASE}", name="08c amyloid papers re-included as errors")
# by mode family
X["m_cdr3fr4"] = X.modes.fillna("").str.contains("Dplus|Dminus|E_FR4")
X["m_fr1"] = X.modes.fillna("").str.contains("A_FR1")
X["m_mid"] = X.modes.fillna("").str.contains("F_CDR1|B_CDR2|C_FR3")
for mcol, lab_ in [("m_cdr3fr4", "CDR3/FR4 modes"), ("m_fr1", "FR1 N-term mode"), ("m_mid", "CDR1/CDR2/FR3 modes")]:
    d = X[(X.err == 0) | X[mcol]].copy()
    logit(d, f"err ~ {XV} + {BASE}", name=f"08d error papers restricted to {lab_}")
# single-error-unit vs multi-error-unit
d = X[(X.err == 0) | (X.n_err_units == 1)]
logit(d, f"err ~ {XV} + {BASE}", name="08e error papers with exactly 1 error antibody")

# ---------- 9. influence ----------
d = X.copy()
labs_err = d[d.err == 1].lab.value_counts()
print("labs with >=2 error papers:", labs_err[labs_err >= 2].to_dict())
d2 = d[~d.lab.isin(labs_err[labs_err >= 2].index)]
logit(d2, f"err ~ {XV} + {BASE}", name="09a drop labs with >=2 error papers")
# dfbeta-like leave-one-lab-out
res = []
for lb in d[d.err == 1].lab.unique():
    s = d[d.lab != lb]
    mm = smf.logit(f"err ~ {XV} + {BASE}", s).fit(disp=0)
    res.append((lb, np.exp(mm.params[XV]), mm.pvalues[XV]))
res = pd.DataFrame(res, columns=["lab", "OR", "p"])
rows.append(dict(test="09b leave-one-error-lab-out range", model="logit base", OR=float(res.OR.median()), lo=float(res.OR.min()),
                 hi=float(res.OR.max()), p=float(res.p.max()), n=len(res), note="lo/hi = min/max OR; p = max p"))
print("09b LOLO", res.OR.min(), res.OR.max(), res.p.max())
# trimming extremes of citations
lo_, hi_ = X[XV].quantile([.05, .95])
d = X[X[XV].between(lo_, hi_)]
logit(d, f"err ~ {XV} + {BASE}", name="09c trim 5% tails of citations/yr")
d = X[X.age_y >= 1.0]
logit(d, f"err ~ {XV} + {BASE}", name="09d papers published >=1 yr before citation snapshot")
d = X[X.age_y >= 2.0]
logit(d, f"err ~ {XV} + {BASE}", name="09e papers published >=2 yr before citation snapshot")

R = pd.DataFrame(rows)
R.to_csv(f"{OUT}/v01_tests.csv", index=False)
X.to_csv(f"{OUT}/v01_analysis_frame.csv.gz", index=False)
pd.set_option("display.width", 250, "display.max_colwidth", 60, "display.max_rows", 200)
print(R[["test", "OR", "lo", "hi", "p", "n", "n_err", "note"]].round(4).to_string())
