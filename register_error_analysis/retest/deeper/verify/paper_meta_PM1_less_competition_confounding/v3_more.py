#!/usr/bin/env python3
"""PM1 confounding, part 2.
 a) which class term drives the attenuation (capsid / enzyme / fiducial-only);
 b) local-fit and unbuilt-residue adjustment (complete cases);
 c) parsimonious adjusted model (fewer parameters: 121 events);
 d) unit level (antibody-in-paper), cluster-robust by lab;
 e) copied/artifact-only papers counted as non-error controls instead of excluded;
 f) leave-one-lab-out and leave-one-class-out influence;
 g) within-class Mantel-Haenszel.
Writes tests_confounding_2.csv"""
import json, warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.contingency_tables import StratifiedTable

warnings.filterwarnings("ignore")
DEEP = "/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad/deeper"
SAB = "/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad/sabr/out"
OUT = "/home/user/SAbR/register_error_analysis/retest/deeper/verify/paper_meta_PM1_less_competition_confounding"
T = pd.read_csv(f"{OUT}/paper_table_model.csv", parse_dates=["dep_first"])

# ---- unbuilt V residues per chain (internal gaps between first and last built V residue + N-term start offset)
unb = {}
for fn in ["new.jsonl", "snapshot_rest.jsonl", "validation.jsonl"]:
    with open(f"{SAB}/{fn}") as f:
        for l in f:
            r = json.loads(l)
            res = r.get("residues") or []
            si = [x[3] for x in res if x[3] is not None]
            if not si:
                continue
            span = max(si) - min(si) + 1
            unb[(r["pdb"].lower(), str(r["chain"]))] = (span - len(set(si)), min(si))
c = pd.read_csv(f"{DEEP}/chains.csv.gz", usecols=["pdb", "chain", "unit", "err_first", "err_any", "lq_z", "method"])
c["gap"] = [unb.get((p, str(ch)), (np.nan, np.nan))[0] for p, ch in zip(c.pdb, c.chain)]
c["has_gap"] = (c.gap > 0).astype(float).where(c.gap.notna())
print("chains with unbuilt-gap info:", c.gap.notna().mean(), " mean has_gap err_first vs no run:",
      c[c.err_first].has_gap.mean(), c[~c.err_any].has_gap.mean())
cg = c.groupby("pdb")
gp = []
for _, p in T.iterrows():
    cc = c[c.pdb.isin(p.pdbs.split(";"))]
    gp.append(dict(pkey=p.pkey, gap_frac_all=cc.has_gap.mean(), gap_frac_clean=cc[~cc.err_any].has_gap.mean()))
T = T.merge(pd.DataFrame(gp), on="pkey")
T["capenz"] = T[["cap", "enz"]].max(axis=1)


def fit(formula, d, var, cluster="lab", label="", subset="all EM papers"):
    cols = [x for x in ["err", var, "lqz", "lqz_c", "gap_frac_all", "gap_frac_clean"] if x in formula or x == var or x == "err"]
    d = d.dropna(subset=cols)
    try:
        if cluster is None:
            m = smf.logit(formula, d).fit(disp=0, maxiter=200)
        else:
            m = smf.glm(formula, d, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d[cluster])[0]})
        ci = np.exp(m.conf_int().loc[var])
        return dict(test=label, subset=subset, var=var, formula=formula.split("~")[1].strip(), n=len(d), n_err=int(d.err.sum()),
                    err_mean=d[d.err == 1][var].mean(), ctrl_mean=d[d.err == 0][var].mean(),
                    OR=np.exp(m.params[var]), lo=ci[0], hi=ci[1], p=m.pvalues[var])
    except Exception as ex:
        return dict(test=label, subset=subset, var=var, formula=formula, n=len(d), OR=np.nan, p=np.nan, note=str(ex)[:100])


R = []
B = "C(period) + l2_units + res_median"
FINE = "bs(t, df=5) + l2_units + res_median"
FULL = FINE + " + l_acc + l2_entries + cap + enz + fid_only + unbound_only + lexp + novel_frac + sars2 + viral_org"
# a) decomposition
for add, lab in [("", "base (period, size, res)"), (" + cap", "+ capsid only"), (" + enz", "+ enzyme only"), (" + fid_only", "+ fiducial-only only"),
                 (" + cap + enz", "+ capsid + enzyme"), (" + cap + enz + fid_only", "+ capsid + enzyme + fiducial-only"),
                 (" + C(cls_mode)", "+ modal target class (13 classes)")]:
    R.append(fit(f"err ~ any_comp + {B}{add}", T, "any_comp", label="decomposition: " + lab))
R.append(fit(f"err ~ any_comp + {B}", T[T.fid_only == 0], "any_comp", label="base, fiducial-only papers dropped", subset="no fiducial-only"))
R.append(fit(f"err ~ any_comp + {B}", T[T.capenz == 0], "any_comp", label="base, capsid+enzyme papers dropped", subset="no capsid/enzyme"))
R.append(fit(f"err ~ any_comp + {B}", T[T.cap == 0], "any_comp", label="base, capsid papers dropped", subset="no capsid"))
# b) local map fit / unbuilt residues (complete cases)
for add, lab in [(" + lqz", "+ paper median lq_z (all V chains)"), (" + lqz_c", "+ median lq_z of run-free chains"),
                 (" + gap_frac_all", "+ share of V chains with unbuilt internal residues"),
                 (" + gap_frac_clean", "+ share of run-free V chains with unbuilt residues")]:
    R.append(fit(f"err ~ any_comp + {B}{add}", T, "any_comp", label="base" + add + " " + lab))
    R.append(fit(f"err ~ any_comp + {FULL}{add}", T, "any_comp", label="FULL" + add + " " + lab))
R.append(fit(f"lqz ~ 1", T, "lqz", label="dummy", subset="dummy") if False else {})
# c) parsimonious
PARS = "C(period) + l2_units + res_median + cap + enz + fid_only + sars2"
R.append(fit(f"err ~ any_comp + {PARS}", T, "any_comp", label="parsimonious (period, size, res, capsid, enzyme, fiducial-only, SARS2)"))
R.append(fit(f"err ~ l_comp + {PARS}", T, "l_comp", label="parsimonious, log2 competitors"))
R.append(fit(f"err ~ any_tgt + {PARS}", T, "any_tgt", label="parsimonious, contacted antigen only"))
R.append(fit(f"err ~ any_comp + {PARS} + lexp", T, "any_comp", label="parsimonious + lab experience"))
R.append(fit(f"err ~ any_comp + {PARS} + lexp + C(period):lexp", T, "any_comp", label="parsimonious + lab exp x period"))
# interaction with lab experience (subgroup claim check)
T["inexp"] = (T.lab_prior_ab < 10).astype(int)
m = smf.glm(f"err ~ any_comp*inexp + {PARS}", T, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(T.lab)[0]})
R.append(dict(test="interaction any_comp x inexperienced lab (<10 prior Ab entries)", subset="all", var="any_comp:inexp", n=len(T), n_err=int(T.err.sum()),
              OR=np.exp(m.params["any_comp:inexp"]), lo=np.exp(m.conf_int().loc["any_comp:inexp"][0]), hi=np.exp(m.conf_int().loc["any_comp:inexp"][1]), p=m.pvalues["any_comp:inexp"]))

# g) Mantel-Haenszel within class strata x period
T["cls3"] = np.select([T.sars2 == 1, T.cap == 1, T.enz == 1, T.fid_only == 1, T.viral_org == 1], ["SARS2", "capsid", "enzyme", "fiducial-only", "viral other"], "non-viral other")
T["per2"] = np.where(T.dep_first.dt.year <= 2021, "<=2021", ">=2022")
tabs = []
for (k, pp), g in T.groupby(["cls3", "per2"]):
    t = pd.crosstab(g.any_comp, g.err).reindex(index=[1, 0], columns=[1, 0], fill_value=0).values
    if (t.sum(axis=1) > 0).all() and (t.sum(axis=0) > 0).all():
        tabs.append(t.astype(float))
st = StratifiedTable(tabs)
R.append(dict(test="Mantel-Haenszel any_comp, strata = class(6) x period(2)", subset="all", var="any_comp", n=len(T), n_err=int(T.err.sum()),
              OR=st.oddsratio_pooled, lo=st.oddsratio_pooled_confint()[0], hi=st.oddsratio_pooled_confint()[1], p=st.test_null_odds().pvalue,
              note=f"BD homogeneity p={st.test_equal_odds().pvalue:.3f}; strata={len(tabs)}"))

# f) influence: leave-one-lab-out (error labs) and leave-one-class-out
labs = T[T.err == 1].lab.value_counts()
lo_rows = []
for lb in labs.index:
    d = T[T.lab != lb]
    r = fit(f"err ~ any_comp + {PARS}", d, "any_comp", label=f"LOLO drop {lb}")
    lo_rows.append(dict(lab=lb, n_err_papers=labs[lb], OR=r["OR"], p=r["p"]))
LO = pd.DataFrame(lo_rows)
LO.to_csv(f"{OUT}/leave_one_lab_out.csv", index=False)
R.append(dict(test="leave-one-error-lab-out (parsimonious model): OR range", subset=f"{len(LO)} labs", var="any_comp",
              OR=LO.OR.median(), lo=LO.OR.min(), hi=LO.OR.max(), p=LO.p.max(), note=f"p range {LO.p.min():.4f}-{LO.p.max():.4f}"))

# e) copied/artifact-only papers as controls: need their competition; recompute via v1 comp() logic quickly
P = pd.read_csv("/home/user/SAbR/register_error_analysis/retest/deeper/paper_meta/paper_analysis_table.csv", parse_dates=["dep_first"])
co = P[(P.method == "EM") & (P.group == "copied_only")]
print("copied-only EM papers:", len(co))

# d) unit level
c2 = pd.read_csv(f"{DEEP}/chains.csv.gz", usecols=["pdb", "unit", "err_first", "err_any", "method", "vhh", "species", "ctype", "res"])
rows = []
for _, p in T.iterrows():
    cc = c2[c2.pdb.isin(p.pdbs.split(";"))]
    for u, g in cc.groupby("unit"):
        e = bool(g.err_first.any())
        if not e and g.err_any.any():
            continue  # copied/artifact-only unit
        rows.append(dict(pkey=p.pkey, unit=u, uerr=int(e), vhh=int((g.vhh == 1).any()), mouse=int((g.species == "mouse").any())))
U = pd.DataFrame(rows).merge(T[["pkey", "any_comp", "l_comp", "period", "t", "res_median", "l2_units", "cap", "enz", "fid_only", "sars2", "viral_org", "lexp", "lab", "novel_frac"]], on="pkey")
U = U.rename(columns={"uerr": "err"})
for f, lab in [(f"err ~ any_comp + {B}", "unit level, base"), (f"err ~ any_comp + {PARS}", "unit level, parsimonious"),
               (f"err ~ any_comp + {PARS} + vhh + mouse + lexp", "unit level, parsimonious + nanobody + mouse + lab exp")]:
    for cl in ["pkey", "lab"]:
        R.append(fit(f, U, "any_comp", cluster=cl, label=lab + f" (cluster {cl})", subset="antibody units in EM error/clean papers"))

Rt = pd.DataFrame([r for r in R if r])
Rt.to_csv(f"{OUT}/tests_confounding_2.csv", index=False)
pd.set_option("display.width", 250, "display.max_colwidth", 80, "display.max_rows", 200)
print(Rt[["test", "subset", "n", "n_err", "err_mean", "ctrl_mean", "OR", "lo", "hi", "p"]].round(4).to_string())
print(LO.sort_values("OR").to_string())
print(T.groupby(["any_comp"])[["lqz", "lqz_c", "gap_frac_all", "gap_frac_clean"]].mean())
print(T.groupby(["err"])[["lqz", "lqz_c", "gap_frac_all", "gap_frac_clean"]].mean())
T.to_csv(f"{OUT}/paper_table_model2.csv", index=False)
