#!/usr/bin/env python3
"""Paper-level comparison of cryo-EM papers (EM entries deposited 2018+) with first-occurrence register errors vs
papers whose EM entries carry no disagreement run of any kind.

Model for each feature x:  logit P(error paper) ~ x + C(period) + log2(n_units) + res_median
  (n_units = antibody V regions in the paper: a paper's chance of >=1 error scales with how many it builds.)
Binary features are also tested with a period-stratified Cochran-Mantel-Haenszel test (no size adjustment).
Benjamini-Hochberg q over all primary tests (adjusted-logit p).
"""
import re, warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.contingency_tables import StratifiedTable

warnings.filterwarnings("ignore")
OUT = "/home/user/SAbR/register_error_analysis/retest/deeper/paper_meta"
B = pd.read_csv(f"{OUT}/paper_base.csv", parse_dates=["dep_first", "dep_last", "rel_first", "rel_last"])
M = pd.read_csv(f"{OUT}/paper_meta.csv", parse_dates=["epmc_first_pub", "epmc_epub", "cr_pub_online", "cr_pub_print",
                                                     "cr_created", "received", "revised", "accepted", "pm_aheadofprint",
                                                     "pub_date", "preprint_date"])
D = B.merge(M, on="doi", how="left")
D["err"] = (D.group == "error").astype(int)
D["published"] = D.doi.notna() & ~D.doi.fillna("").str.match(r"10\.1101/|10\.64898/|10\.21203/|10\.2139/")
D["preprint_only_citation"] = D.doi.fillna("").str.match(r"10\.1101/|10\.64898/|10\.21203/|10\.2139/")
D["l2_units"] = np.log2(D.n_units.clip(lower=1))
D["l2_entries"] = np.log2(D.n_entries)
D["l2_entries_pdb_total"] = np.log2(D.n_entries_pdb_total.clip(lower=1))

# ---- intervals (days)
day = lambda s: s.dt.days
D["pre_to_pub"] = day(D.pub_date - D.preprint_date)
D["dep_to_pub"] = day(D.pub_date - D.dep_first)
D["deplast_to_pub"] = day(D.pub_date - D.dep_last)
D["dep_minus_preprint"] = day(D.dep_first - D.preprint_date)
D["dep_before_preprint"] = np.where(D.preprint_date.notna(), D.dep_first < D.preprint_date, np.nan)
D["rec_to_acc"] = day(D.accepted - D.received)
D["acc_to_pub"] = day(D.pub_date - D.accepted)
D["dep_minus_received"] = day(D.dep_first - D.received)
D["dep_after_submission"] = np.where(D.received.notna(), D.dep_first > D.received + pd.Timedelta(days=7), np.nan)
D["deplast_after_submission"] = np.where(D.received.notna(), D.dep_last > D.received + pd.Timedelta(days=7), np.nan)
D["dep_after_acceptance"] = np.where(D.accepted.notna(), D.dep_last > D.accepted, np.nan)
D["released_before_pub"] = np.where(D.pub_date.notna(), D.rel_first < D.pub_date - pd.Timedelta(days=7), np.nan)
D["rel_minus_pub"] = day(D.rel_first - D.pub_date)
D["fast_review_lt60d"] = np.where(D.rec_to_acc.notna(), D.rec_to_acc < 60, np.nan)
D["preprint_lt90d_after_dep"] = np.where(D.dep_minus_preprint.notna(), D.dep_minus_preprint.between(-90, 90), np.nan)
D["l_authors"] = np.log2(D.n_authors)
D["many_countries"] = D.n_countries >= 3
D["multi_structure_paper"] = D.n_entries_pdb_total >= 5

# citations: percentile within publication year among all papers in this table with a citation count
D["pub_year"] = D.pub_date.dt.year
D["cites"] = D[["cites_epmc", "cites_cr"]].max(axis=1)
D["cite_pct"] = D.groupby("pub_year").cites.rank(pct=True)
D["l_cites_per_year"] = np.log1p(D.cites / ((pd.Timestamp("2026-11-01") - D.pub_date).dt.days / 365.25).clip(lower=0.25))

# journal tiers
j = D.journal_epmc.fillna(D.journal_pdb).fillna("")
D["journal_n"] = j
D["j_CNS"] = j.str.fullmatch(r"(?i)nature|science|cell")
D["j_sister"] = j.str.match(r"(?i)nature (?!communications)|cell (?!reports|discovery|research)|science (?!advances)|immunity|molecular cell|structure$|neuron") & ~D.j_CNS
D["j_natcommun"] = j.str.match(r"(?i)nature communications")
D["j_open_megajournal"] = j.str.match(r"(?i)nature communications|science advances|plos|elife|scientific reports|cell reports|communications biology|iscience|frontiers|mbio|embo")
D["region_last"] = D.last_country.map(lambda c: "USA" if c == "USA" else "China" if c in ("China", "Hong Kong", "Taiwan")
                                      else "Europe" if c in ("UK", "Germany", "France", "Switzerland", "Netherlands", "Belgium",
                                                             "Sweden", "Denmark", "Austria", "Italy", "Spain", "Finland", "Norway",
                                                             "Ireland", "Czechia", "Poland", "Portugal", "Hungary")
                                      else ("Other" if isinstance(c, str) else np.nan))
for r in ["USA", "China", "Europe", "Other"]:
    D[f"last_{r}"] = np.where(D.region_last.notna(), D.region_last == r, np.nan)

D.to_csv(f"{OUT}/paper_analysis_table.csv", index=False)

EM = D[(D.method == "EM") & D.group.isin(["error", "clean"])].copy()
EMP = EM[EM.published].copy()

COV = "C(period) + l2_units + res_median"


def fit(df, x, cov=COV):
    d = df.dropna(subset=[x, "res_median"]).copy()
    d[x] = d[x].astype(float)
    if d[x].nunique() < 2 or d.err.sum() < 3:
        return np.nan, (np.nan, np.nan), np.nan, len(d)
    try:
        m = smf.logit(f"err ~ Q('{x}') + {cov}", d).fit(disp=0, maxiter=200)
        b = m.params[f"Q('{x}')"]
        ci = m.conf_int().loc[f"Q('{x}')"]
        return float(np.exp(b)), (float(np.exp(ci[0])), float(np.exp(ci[1]))), float(m.pvalues[f"Q('{x}')"]), len(d)
    except Exception:
        return np.nan, (np.nan, np.nan), np.nan, len(d)


def cmh(df, x):
    d = df.dropna(subset=[x])
    tabs = []
    for _, g in d.groupby("period"):
        t = pd.crosstab(g[x].astype(bool), g.err.astype(bool)).reindex(index=[True, False], columns=[True, False], fill_value=0).values
        if t.sum(axis=1).min() > 0 and t.sum(axis=0).min() > 0:
            tabs.append(t.astype(float) + 0.0)
    if not tabs:
        return np.nan, np.nan
    st = StratifiedTable(tabs)
    return float(st.oddsratio_pooled), float(st.test_null_odds(correction=True).pvalue)


rows = []


def test_bin(df, x, label, cov=COV, set_name="EM published"):
    d = df.dropna(subset=[x])
    e, c = d[d.err == 1], d[d.err == 0]
    orr, ci, p, n = fit(df, x, cov)
    mh_or, mh_p = cmh(df, x)
    rows.append(dict(set=set_name, feature=x, label=label, kind="binary",
                     n_err=len(e), n_ctrl=len(c),
                     err_value=f"{e[x].astype(float).mean():.1%} ({int(e[x].astype(float).sum())}/{len(e)})",
                     ctrl_value=f"{c[x].astype(float).mean():.1%} ({int(c[x].astype(float).sum())}/{len(c)})",
                     adj_OR=orr, ci_lo=ci[0], ci_hi=ci[1], p_adj=p, mh_OR_period=mh_or, mh_p=mh_p, n_model=n))


def test_cont(df, x, label, scale=1.0, unit="", cov=COV, set_name="EM published"):
    d = df.dropna(subset=[x]).copy()
    e, c = d[d.err == 1][x], d[d.err == 0][x]
    if len(e) < 3:
        return
    mw = stats.mannwhitneyu(e, c).pvalue
    d[x + "_s"] = d[x] / scale
    orr, ci, p, n = fit(d, x + "_s", cov)
    rows.append(dict(set=set_name, feature=x, label=f"{label} (OR per {unit})", kind="continuous",
                     n_err=len(e), n_ctrl=len(c),
                     err_value=f"median {e.median():.3g} [IQR {e.quantile(.25):.3g}-{e.quantile(.75):.3g}]",
                     ctrl_value=f"median {c.median():.3g} [IQR {c.quantile(.25):.3g}-{c.quantile(.75):.3g}]",
                     adj_OR=orr, ci_lo=ci[0], ci_hi=ci[1], p_adj=p, mh_OR_period=np.nan, mh_p=mw, n_model=n))


# ---------------- tests on all EM papers (incl. unpublished) ----------------
test_bin(EM, "published", "Paper published in a journal (vs unpublished / preprint-only citation)", set_name="EM all")
test_bin(EM, "preprint_only_citation", "PDB citation is a preprint only", set_name="EM all")
test_cont(EM, "l2_units", "Antibody V regions in paper", 1, "doubling", cov="C(period) + res_median", set_name="EM all")
test_cont(EM, "l2_entries", "EM entries (3-5 A) in paper", 1, "doubling", set_name="EM all")
test_cont(EM, "hold_days_median", "Hold time deposition->release", 90, "90 d", set_name="EM all")
test_cont(EM, "dep_span_days", "Span of deposition dates within paper", 90, "90 d", set_name="EM all")
test_bin(EM, "tool_paper", "All antibodies are auxiliary/tools (paper not about the antibody)", set_name="EM all")
test_cont(EM, "frac_aux_units", "Fraction of auxiliary antibody units", 1, "0->1", set_name="EM all")
test_bin(EM, "sars2", "SARS-CoV-2 paper", set_name="EM all")
test_bin(EM, "novel_target", "Novel target (no earlier structure)", set_name="EM all")
test_cont(EM, "competitors", "Competing papers on same antigen (prior def.)", 1, "paper", set_name="EM all")

# ---------------- tests on published EM papers ----------------
X = EMP
test_cont(X, "l2_entries_pdb_total", "All PDB entries citing the paper", 1, "doubling")
test_bin(X, "multi_structure_paper", ">=5 PDB entries cite the paper")
test_bin(X, "has_preprint", "Preprint exists")
test_cont(X, "pre_to_pub", "Preprint -> publication interval", 90, "90 d")
test_cont(X, "dep_to_pub", "First EM deposition -> publication", 90, "90 d")
test_cont(X, "deplast_to_pub", "Last EM deposition -> publication", 90, "90 d")
test_cont(X, "dep_minus_preprint", "Deposition minus preprint date (+ = deposited after preprint)", 90, "90 d")
test_bin(X, "dep_before_preprint", "Deposition preceded preprint")
test_bin(X, "preprint_lt90d_after_dep", "Preprint within 90 d of deposition")
test_cont(X, "rec_to_acc", "Received -> accepted (review time)", 30, "30 d")
test_bin(X, "fast_review_lt60d", "Review time < 60 d")
test_cont(X, "acc_to_pub", "Accepted -> published", 30, "30 d")
test_cont(X, "dep_minus_received", "Deposition minus submission date", 90, "90 d")
test_bin(X, "dep_after_submission", "First EM structure deposited >7 d after submission")
test_bin(X, "deplast_after_submission", "Some EM structure deposited >7 d after submission (added in revision)")
test_bin(X, "dep_after_acceptance", "Some EM structure deposited after acceptance")
test_bin(X, "released_before_pub", "Structures released >7 d before the paper")
test_cont(X, "rel_minus_pub", "Release minus publication", 30, "30 d")
test_cont(X, "l_authors", "Number of authors", 1, "doubling")
test_cont(X, "n_countries", "Countries among author affiliations", 1, "country")
test_bin(X, "many_countries", ">=3 countries")
for r in ["USA", "China", "Europe", "Other"]:
    test_bin(X, f"last_{r}", f"Last author in {r}")
test_bin(X, "aff_industry", "Any industry affiliation")
for fk in ["NIH", "NIAID", "HHMI", "Gates", "Wellcome", "China_funders", "EU_ERC", "UKRI_MRC", "DFG", "Japan_funders",
           "CEPI", "DoD_DARPA", "BARDA_HHS", "NSF", "SNSF", "CIHR_Canada", "Industry", "Fast_Grants_COVID"]:
    test_bin(X, f"fund_{fk}", f"Funder: {fk}")
test_cont(X, "n_funder_records", "Number of funder records", 1, "record")
test_bin(X, "open_access", "Open access")
test_cont(X, "cite_pct", "Citation percentile within pub-year", 0.25, "quartile")
test_cont(X, "l_cites_per_year", "log citations per year", 1, "log unit")
test_bin(X, "j_CNS", "Journal: Nature/Science/Cell")
test_bin(X, "j_sister", "Journal: CNS sister journal")
test_bin(X, "j_natcommun", "Journal: Nature Communications")
test_bin(X, "j_open_megajournal", "Journal: broad OA/rapid journals")
test_bin(X, "pt_letter_brief", "Letter / brief report article type")
test_bin(X, "pt_nih_extramural", "PubMed: NIH extramural support")
test_bin(X, "pt_nih_intramural", "PubMed: NIH intramural")
test_bin(X, "has_review_relation", "Published peer review linked (Crossref)")
X["has_erratum"] = X.n_erratum > 0
test_bin(X, "has_erratum", "Paper has an erratum")
X["has_comment"] = X.n_comment_in > 0
test_bin(X, "has_comment", "Paper has a linked comment/News&Views")
for k in ["kw_covid", "kw_vaccine", "kw_neutraliz", "kw_broad", "kw_structural_basis", "kw_nanobody", "kw_fiducial",
          "kw_antibody", "kw_hiv", "kw_influenza", "kw_receptor_mech", "kw_therapeutic", "kw_escape",
          "kw_human_infection", "kw_cryoem_title"]:
    test_bin(X, k, f"Title keyword {k[3:]}")

R = pd.DataFrame(rows)
R["q_adj"] = np.nan
ok = R.p_adj.notna()
R.loc[ok, "q_adj"] = multipletests(R.loc[ok, "p_adj"], method="fdr_bh")[1]
R = R.sort_values("p_adj")
R.to_csv(f"{OUT}/tests_em_papers.csv", index=False)
pd.set_option("display.width", 250, "display.max_colwidth", 70, "display.max_rows", 200)
print(R[["feature", "n_err", "n_ctrl", "err_value", "ctrl_value", "adj_OR", "ci_lo", "ci_hi", "p_adj", "q_adj", "mh_OR_period", "mh_p"]].to_string())

# ---------------- journal heterogeneity ----------------
jj = EMP.journal_n.str.replace(r"(?i)^the ", "", regex=True)
top = jj.value_counts()
top = top[top >= 15].index
EMP["jgroup"] = np.where(jj.isin(top), jj, "other")
jt = EMP.groupby("jgroup").agg(n=("err", "size"), err=("err", "sum"), units=("n_units", "sum"),
                               err_units=("n_err_units", "sum")).assign(rate=lambda d: d.err / d.n,
                                                                        unit_rate=lambda d: d.err_units / d.units)
m0 = smf.logit(f"err ~ {COV}", EMP.dropna(subset=["res_median"])).fit(disp=0)
m1 = smf.logit(f"err ~ C(jgroup) + {COV}", EMP.dropna(subset=["res_median"])).fit(disp=0)
lr = 2 * (m1.llf - m0.llf)
pj = stats.chi2.sf(lr, m1.df_model - m0.df_model)
jt.sort_values("rate").to_csv(f"{OUT}/journal_rates.csv")
print(jt.sort_values("rate"))
print(f"journal heterogeneity LR={lr:.1f} df={m1.df_model - m0.df_model:.0f} p={pj:.3g}")

# ---------------- country heterogeneity ----------------
c_ = EMP.dropna(subset=["region_last", "res_median"])
m0 = smf.logit(f"err ~ {COV}", c_).fit(disp=0)
m1 = smf.logit(f"err ~ C(region_last) + {COV}", c_).fit(disp=0)
print("region heterogeneity p=", stats.chi2.sf(2 * (m1.llf - m0.llf), m1.df_model - m0.df_model))
print(c_.groupby("region_last").agg(n=("err", "size"), rate=("err", "mean")))
