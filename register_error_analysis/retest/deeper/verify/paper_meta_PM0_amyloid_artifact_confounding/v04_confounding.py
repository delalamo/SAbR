#!/usr/bin/env python3
"""Can year / method / resolution / chain type / SARS-CoV-2 / paper clustering / map quality / disagreement extent
explain why all AL fibril entries are flagged?  And what does leaving them in do to other results?"""
import json, numpy as np, pandas as pd, statsmodels.formula.api as smf, statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests
B = "/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad"
PM = "/home/user/SAbR/register_error_analysis/retest/deeper/paper_meta"
OUT = "/home/user/SAbR/register_error_analysis/retest/deeper/verify/paper_meta_PM0_amyloid_artifact_confounding"
AMY = ['6ic3','6z1o','7nsl','9els','9faa','9fab','9fac','9liv','9liw','9lix','9liy','9lj0','9lja','9ljb','9v91']
PAPER = {'6ic3':'Radamaker2019','6z1o':'Radamaker2021a','7nsl':'Radamaker2021b','9els':'Saelices2024','9faa':'Ricagno2024',
         '9fab':'Ricagno2024','9fac':'Ricagno2024','9liv':'LiD2025','9liw':'LiD2025','9lix':'LiD2025','9liy':'LiD2025',
         '9lj0':'LiD2025','9lja':'LiD2025','9ljb':'LiD2025','9v91':'HuangH2025'}
CH = pd.read_csv(f"{B}/deeper/chains.csv.gz"); E = pd.read_csv(f"{B}/deeper/entries.csv")
tests = []

def pb_upper(ps, k):
    dist = np.zeros(len(ps) + 1); dist[0] = 1
    for p in ps:
        dist[1:] = dist[1:] * (1 - p) + dist[:-1] * p; dist[0] *= (1 - p)
    return float(dist[k:].sum())

# ---- entry-level covariates
cc = CH.groupby("pdb").agg(n_L=("ctype", lambda s: (s == "L").sum()), n_H=("ctype", lambda s: (s == "H").sum()),
                          vhh=("vhh", "max"), lq_z=("lq_z", "median"), paper=("paper", "first"))
X = E.merge(cc, on="pdb", how="left")
X["amy"] = X.pdb.isin(AMY)
X["y"] = X.err_first.astype(int)
X["ybin"] = pd.cut(X.year, [0, 2017, 2019, 2020, 2021, 2023, 2026], labels=["<=17", "18-19", "20", "21", "22-23", "24-26"]).astype(str)
X["l2v"] = np.log2(X.n_V_chains.clip(lower=1))
X["has_L"] = (X.n_L > 0).astype(int)
X["no_H"] = (X.n_H == 0).astype(int)
X["sars2"] = X.sars2.astype(int)
X["paper"] = X.paper.fillna(X.pdb)
EM = X[(X.method == "EM")].copy()
fit_on = EM[~EM.amy & (EM.err_first | ~EM.err_any)].dropna(subset=["res"])  # error vs clean, as in the project
for form in ["y ~ C(ybin) + res + l2v", "y ~ C(ybin) + res + l2v + has_L + sars2", "y ~ C(ybin) + res + l2v + has_L + no_H + sars2"]:
    m = smf.glm(form, fit_on, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(fit_on.paper)[0]})
    A = EM[EM.amy].copy(); A["p"] = m.predict(A)
    # paper level: P(paper flagged) = 1 - prod(1-p_entry) (independence within paper, generous to the null)
    pp = A.assign(pp=PAPER).groupby(A.pdb.map(PAPER)).p.apply(lambda s: 1 - np.prod(1 - s))
    r = dict(test=f"expected flagged AL entries | {form}", n=len(A), observed=int(A.y.sum()), expected=A.p.sum(),
             p=pb_upper(A.p.values, int(A.y.sum())), papers_obs=7, papers_expected=pp.sum(), p_papers=pb_upper(pp.values, 7))
    tests.append(r); print(r)
print(EM[EM.amy][["pdb", "year", "res", "n_V_chains", "has_L", "no_H"]].assign(p=m.predict(EM[EM.amy])).round(3).to_string())

# ---- direct comparator: EM entries 2018+ whose only V chains are light chains (no heavy, no VHH)
lc_only = EM[(EM.no_H == 1) & (EM.year >= 2018)]
print("EM entries 2018+ with light-chain V only:", lc_only.groupby("amy").agg(n=("y", "size"), err=("y", "sum")).to_dict())
print(lc_only[~lc_only.amy][["pdb", "year", "res", "err_first", "err_any", "title"]].to_string())

# ---- chain-type matched comparator: lambda units in EM, same period (2018+), same resolution band (3.0-3.9)
CU = CH[(CH.method == "EM") & (CH.ctype == "L") & (CH.year >= 2018) & CH.res.between(3.0, 3.95)].copy()
CU["amy"] = CU.pdb.isin(AMY)
U = CU.groupby("unit").agg(amy=("amy", "max"), err=("err_first", "max"), errany=("err_any", "max"), year=("year", "min"),
                           paper=("paper", "first"))
U = U[U.err | ~U.errany | U.amy]
tab = U.groupby("amy").agg(n=("err", "size"), err=("err", "sum"))
tab["rate"] = tab.err / tab.n
print("EM lambda units 2018+, 3.0-3.95 A:\n", tab)
ft = stats.fisher_exact([[tab.loc[True, "err"], tab.loc[True, "n"] - tab.loc[True, "err"]],
                         [tab.loc[False, "err"], tab.loc[False, "n"] - tab.loc[False, "err"]]])
tests.append(dict(test="lambda units EM 2018+ 3.0-3.95A: AL fibril vs other", n=int(tab.n.sum()), observed=f"{int(tab.loc[True,'err'])}/{int(tab.loc[True,'n'])}",
                  expected=f"{int(tab.loc[False,'err'])}/{int(tab.loc[False,'n'])} ({tab.loc[False,'rate']:.1%})", p=ft[1]))
for yb, g in U.groupby(pd.cut(U.year, [2017, 2021, 2023, 2026])):
    t = g.groupby("amy").err.agg(["size", "sum"]); print(yb, t.to_dict())

# ---- map quality (simpler factor): real errors sit in poor density; AL fibrils?
L = EM[(EM.year >= 2018) & EM.lq_z.notna()].copy()
L["grp"] = np.where(L.amy, "AL", np.where(L.err_first, "error", np.where(~L.err_any, "clean", "other")))
L = L[L.grp != "other"]
print(L.groupby("grp").lq_z.describe().round(2))
for g in ["AL", "error"]:
    d = L[L.grp.isin([g, "clean"])].copy(); d["g"] = (d.grp == g).astype(int)
    m = smf.ols("lq_z ~ g + res + C(ybin)", d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.paper)[0]})
    mw = stats.mannwhitneyu(d[d.g == 1].lq_z, d[d.g == 0].lq_z)
    tests.append(dict(test=f"entry lq_z: {g} vs clean (EM 2018+), adj res+year, cluster paper", n=f"{(d.g==1).sum()} vs {(d.g==0).sum()}",
                      observed=round(d[d.g == 1].lq_z.median(), 2), expected=round(d[d.g == 0].lq_z.median(), 2),
                      effect=round(m.params["g"], 2), ci=f"{m.conf_int().loc['g',0]:.2f}..{m.conf_int().loc['g',1]:.2f}", p=m.pvalues["g"], p_mw=mw.pvalue))
    print(tests[-1])

# ---- extent of disagreement (definition): fraction of numbered V residues where SAbR != ANARCI
S = pd.read_csv(f"{B}/deeper/cache/sabr_scores_all.csv.gz")
S["fdis"] = S.ndis / S.nres.clip(lower=1)
ch = CH[CH.method == "EM"][["pdb", "chain", "err_first", "err_any"]].merge(S[["pdb", "chain", "fdis", "nres"]], on=["pdb", "chain"])
ch["grp"] = np.where(ch.pdb.isin(AMY), "AL", np.where(ch.err_first, "error", np.where(~ch.err_any, "clean", "other")))
fd = ch.groupby("grp").fdis.describe(percentiles=[.5, .9, .99]).round(3); print(fd)
fd.to_csv(f"{OUT}/disagreement_fraction_EM.csv")
print("non-AL EM chains with >50% residues disagreeing:", ch[(ch.grp != "AL") & (ch.fdis > 0.5)][["pdb", "chain", "grp", "fdis", "nres"]].to_string())
tests.append(dict(test="chains with >50% V residues SAbR!=ANARCI: AL vs non-AL EM error chains",
                  n=f"{(ch.grp=='AL').sum()} vs {(ch.grp=='error').sum()}",
                  observed=f"{(ch[ch.grp=='AL'].fdis>0.5).sum()}/{(ch.grp=='AL').sum()}",
                  expected=f"{(ch[ch.grp=='error'].fdis>0.5).sum()}/{(ch.grp=='error').sum()}",
                  p=stats.fisher_exact([[(ch[ch.grp=='AL'].fdis>0.5).sum(), (ch[ch.grp=='AL'].fdis<=0.5).sum()],
                                        [(ch[ch.grp=='error'].fdis>0.5).sum(), (ch[ch.grp=='error'].fdis<=0.5).sum()]])[1]))

# ---- geometry (from v03): separation test
G = pd.read_csv(f"{OUT}/fold_geometry_chains.csv")
al = G[G.grp == "AL_fibril"].drop_duplicates("pdb"); non = G[~G.grp.str.startswith("AL")]
mw = stats.mannwhitneyu(al.thick, non.thick)
tests.append(dict(test="CA-cloud thickness (3rd PC SD, A): AL fibril entries vs non-AL EM chains", n=f"{len(al)} vs {len(non)}",
                  observed=f"median {al.thick.median():.1f} (max {al.thick.max():.1f})", expected=f"median {non.thick.median():.1f} (min {non.thick.min():.1f})", p=mw.pvalue))

# ---- what leaving them in does: EM entry error rate by year, lambda vs kappa unit rates, mode 'other'
yr = EM[EM.year >= 2018].groupby("year").agg(n=("y", "size"), err=("y", "sum"), amy=("amy", "sum"))
yr["rate_with"] = yr.err / yr.n; yr["rate_without"] = (yr.err - yr.amy) / (yr.n - yr.amy)
print(yr.round(4)); yr.to_csv(f"{OUT}/em_error_rate_by_year_with_without_AL.csv")
CL = CH[(CH.method == "EM") & (CH.year >= 2018) & CH.ctype.isin(["K", "L"])]
UL = CL.groupby("unit").agg(ct=("ctype", "first"), err=("err_first", "max"), errany=("err_any", "max"), amy=("pdb", lambda s: s.isin(AMY).any()))
UL = UL[UL.err | ~UL.errany]
for lab, d in [("with AL", UL), ("without AL", UL[~UL.amy])]:
    t = d.groupby("ct").err.agg(["size", "sum", "mean"]); print("light-chain units EM 2018+", lab, t.round(4).to_dict())
    ft = stats.fisher_exact([[t.loc["L", "sum"], t.loc["L", "size"] - t.loc["L", "sum"]], [t.loc["K", "sum"], t.loc["K", "size"] - t.loc["K", "sum"]]])
    tests.append(dict(test=f"lambda vs kappa unit error rate, EM 2018+ ({lab})", n=int(t["size"].sum()),
                      observed=f"L {int(t.loc['L','sum'])}/{int(t.loc['L','size'])} ({t.loc['L','mean']:.1%})",
                      expected=f"K {int(t.loc['K','sum'])}/{int(t.loc['K','size'])} ({t.loc['K','mean']:.1%})", effect=round(ft[0], 2), p=ft[1]))

T = pd.DataFrame(tests)
T["q_bh"] = multipletests(T.p.astype(float), method="fdr_bh")[1]
T.to_csv(f"{OUT}/tests_confounding.csv", index=False)
pd.set_option("display.width", 250, "display.max_colwidth", 90)
print(T.to_string())
