#!/usr/bin/env python3
"""Is the size-adjusted citation OR an artifact of mis-modelling paper size?

Simulate error labels that depend ONLY on paper size (number of antibodies), period and resolution,
never on citations; refit the investigator's base logit (err ~ cites + C(period) + l2_units + res_median)
and record the citation OR.  If the base model mis-adjusts size, the null ORs will be shifted away from 1.

Two generators:
  (a) binomial: each antibody independently erroneous with p(period, resolution) fitted from the data;
  (b) beta-binomial: same mean, overdispersion (within-paper clustering of errors) matched to the data.
Also reports the observed OR's position in each null distribution.
"""
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
OUT = "/home/user/SAbR/register_error_analysis/retest/deeper/verify/paper_meta_PM2_lower_citations_confounding"
X = pd.read_csv(f"{OUT}/v01_analysis_frame.csv.gz")
XV = "l_cites_per_year"
BASE = "C(period) + l2_units + res_median"
obs = smf.logit(f"err ~ {XV} + {BASE}", X).fit(disp=0)
obs_or = np.exp(obs.params[XV])

d = X.copy()
d["n_ok"] = d.n_units - d.n_err_units
g = smf.glm("n_err_units + n_ok ~ C(period) + res_median", d, family=sm.families.Binomial()).fit()
p_hat = g.predict(d).values
n = d.n_units.values.astype(int)
# overdispersion: Pearson chi2 / df -> beta-binomial rho  (phi = 1 + (n-1) rho, use mean n)
phi = g.pearson_chi2 / g.df_resid
nbar = np.average(n, weights=n)
rho = max((phi - 1) / (nbar - 1), 1e-4)
print(f"observed OR {obs_or:.3f}; per-antibody p mean {p_hat.mean():.3f}; phi {phi:.2f}; rho {rho:.3f}")
rng = np.random.default_rng(1)
res = {"binomial": [], "betabinomial": []}
for it in range(500):
    for kind in res:
        if kind == "binomial":
            k = rng.binomial(n, p_hat)
        else:
            a = p_hat * (1 - rho) / rho
            b = (1 - p_hat) * (1 - rho) / rho
            k = rng.binomial(n, rng.beta(a, b))
        d["err"] = (k > 0).astype(int)
        try:
            m = smf.logit(f"err ~ {XV} + {BASE}", d).fit(disp=0)
            res[kind].append(np.exp(m.params[XV]))
        except Exception:
            pass
rows = []
for kind, v in res.items():
    v = np.array(v)
    rows.append(dict(generator=kind, n_sims=len(v), null_median_OR=np.median(v), null_2_5=np.quantile(v, .025),
                     null_97_5=np.quantile(v, .975), frac_null_le_obs=(v <= obs_or).mean(), observed_OR=obs_or,
                     sim_err_papers_mean=np.nan))
    print(kind, np.median(v), np.quantile(v, [.025, .975]), (v <= obs_or).mean())
pd.DataFrame(rows).to_csv(f"{OUT}/v02_size_null_sim.csv", index=False)
