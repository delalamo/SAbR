"""Within-paper conditional logit for DE2 adjusted for chain type (H / K / L / VHH have different reference RMSD
offsets) and for the number of V chains in the entry (multi-copy complexes); matched model + n_V_chains."""
import warnings, numpy as np, pandas as pd
import statsmodels.formula.api as smf
from statsmodels.discrete.conditional_models import ConditionalLogit
from s00_reproduce import load_sample, sets, D, OUT
warnings.filterwarnings('ignore')
X = load_sample()
F = pd.read_csv(f'{OUT}/features_recomputed.csv')
ENT = pd.read_csv(f'{D}/entries.csv')[['pdb', 'n_V_chains']]
X = X.merge(F, on=['pdb', 'chain'], how='left').merge(ENT, on='pdb', how='left')
e, mc, wp = sets(X)
rows = []
for var in ['fw_orig', 'fw_seqref', 'fw_core']:
    d = pd.concat([e, wp]).dropna(subset=[var])
    d = d[d.pp.isin(d[d.err == 1].pp) & d.pp.isin(d[d.err == 0].pp)].copy()
    d['x'] = (d[var] - d[var].mean()) / d[var].std()
    dm = pd.get_dummies(d.ct, drop_first=True).astype(float)
    for lab, cols in [('none', ['x']), ('chain type', ['x'] + list(dm.columns)),
                      ('chain type + res + n_V_chains', ['x'] + list(dm.columns) + ['res', 'n_V_chains'])]:
        Z = pd.concat([d[['x', 'res', 'n_V_chains']], dm], axis=1)
        try:
            m = ConditionalLogit(d.err.values, Z[cols].values, groups=pd.factorize(d.pp)[0]).fit(disp=0)
            rows.append(dict(test=f'within paper, adjusted for {lab}', metric=var, n_err=int(d.err.sum()), n_ctl=int((d.err == 0).sum()),
                             n_papers=d.pp.nunique(), OR_perSD=float(np.exp(m.params[0])),
                             lo=float(np.exp(m.conf_int()[0][0])), hi=float(np.exp(m.conf_int()[0][1])), p=float(m.pvalues[0])))
        except Exception as ex:
            rows.append(dict(test=f'within paper, adjusted for {lab}', metric=var, note=str(ex)[:60]))
    dd = pd.concat([e, mc]).dropna(subset=[var]).copy(); dd['x'] = (dd[var] - dd[var].mean()) / dd[var].std()
    m = smf.logit('err ~ x + C(yb) + res + C(ct) + np.log(n_V_chains)', data=dd).fit(disp=0, cov_type='cluster', cov_kwds={'groups': pd.factorize(dd.pp)[0]})
    rows.append(dict(test='matched + log n_V_chains', metric=var, n_err=int(dd.err.sum()), n_ctl=int((dd.err == 0).sum()),
                     OR_perSD=float(np.exp(m.params['x'])), p=float(m.pvalues['x'])))
T = pd.DataFrame(rows); T.to_csv(f'{OUT}/tests_within_paper_ct.csv', index=False)
pd.set_option('display.width', 220); print(T.round(4).to_string())
print('error vs matched n_V_chains median', e.n_V_chains.median(), mc.n_V_chains.median())
