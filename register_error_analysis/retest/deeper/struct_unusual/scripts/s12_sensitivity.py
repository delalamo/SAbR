"""Sensitivity checks for the main positives:
 (1) + SARS-CoV-2 covariate; (2) 2020-2022 deposits only; (3) matched-set (case minus mean of its own matched
 controls) Wilcoxon signed-rank; (4) + local map fit (q_out / q_V) as covariate for geometry metrics;
 (5) per-mode breakdown of beyond-site geometry. Output tables/sensitivity.csv, tables/mode_breakdown.csv"""
import warnings, numpy as np, pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from common import TAB
warnings.filterwarnings('ignore')
s = pd.read_csv(f'{TAB}/sample.csv', dtype={'chain': str})
U = pd.read_csv(f'{TAB}/unit_metrics.csv', dtype={'chain': str, 'partner': str})
U = U.merge(s[['unit', 'pdb', 'sars2']].drop_duplicates(['unit', 'pdb']), on=['unit', 'pdb'], how='left')
U['case'] = (U.role_cc == 'case').astype(int)
U['sars2'] = U.sars2.astype(float)
U['Dmode'] = U.modes.fillna('').str.contains('D')
rows = []


def fit(y, df, extra='', binary=False, label=''):
    extra_cols = [t.strip() for t in extra.replace('+', ' ').split() if t.strip() in df.columns]
    x = df.dropna(subset=[y] + extra_cols).copy()
    x[y] = x[y].astype(float)
    f = f'{y} ~ case + C(cclass) + C(ybin) + res' + extra
    g = pd.factorize(x.pdb)[0]
    try:
        if binary:
            m = smf.logit(f, x).fit(disp=0, cov_type='cluster', cov_kwds={'groups': g})
            eff, lo, hi = np.exp(m.params['case']), *np.exp(m.conf_int().loc['case'])
        else:
            m = smf.ols(f, x).fit(cov_type='cluster', cov_kwds={'groups': g})
            eff, lo, hi = m.params['case'], *m.conf_int().loc['case']
        rows.append(dict(metric=y, model=label, n_case=int(x.case.sum()), n_ctrl=int((1 - x.case).sum()),
                         effect=eff, ci_lo=lo, ci_hi=hi, p=m.pvalues['case']))
    except Exception as e:
        rows.append(dict(metric=y, model=label, note=str(e)[:80]))


def matched(y, df, label='matched-set Wilcoxon'):
    c = df[df.case == 1].set_index('set_id')[y]
    k = df[df.case == 0].groupby('set_id')[y].mean()
    dd = (c - k).dropna()
    if len(dd) > 5:
        rows.append(dict(metric=y, model=label, n_case=len(dd), effect=dd.median(), mean_diff=dd.mean(),
                         p=stats.wilcoxon(dd).pvalue))


PU = U[U.partner_err_any == False]
H = U[U.cclass.isin(['H', 'VHH'])]
for y, binary, df in [('rama_allowed_out', False, U), ('clash_out', False, U), ('rota_out_out', False, U),
                      ('p_rama_allowed_V', False, PU), ('p_clash_V', False, PU), ('q_site_minus_out', False, U),
                      ('rama_out_site', True, U), ('caca_out_site', True, U), ('h3_kinked', True, H)]:
    fit(y, df, '', binary, 'base')
    fit(y, df, ' + sars2', binary, '+ SARS-CoV-2')
    fit(y, df[df.year.between(2020, 2022)], '', binary, '2020-2022 only')
    if y not in ('q_site_minus_out',):
        fit(y, df, ' + q_out' if 'p_' not in y else ' + p_q_V', binary, '+ local map fit')
    matched(y, df)
S = pd.DataFrame(rows)
S.to_csv(f'{TAB}/sensitivity.csv', index=False)
pd.set_option('display.width', 220)
print(S.round(4).to_string())

# per-mode breakdown (first-listed mode) of beyond-site metrics in cases vs all controls
c = U[U.case == 1].copy(); c['mode1'] = c.modes.fillna('').str.split(';').str[0]
cols = ['rama_allowed_out', 'clash_out', 'rota_out_out', 'q_out', 'q_site_minus_out', 'p_rama_allowed_V']
mb = c.groupby('mode1')[cols].mean()
mb['n'] = c.groupby('mode1').size()
mb.loc['ALL CONTROLS'] = list(U[U.case == 0][cols].mean()) + [int((U.case == 0).sum())]
mb.round(3).to_csv(f'{TAB}/mode_breakdown.csv')
print(mb.round(2))

# ---- same sensitivity checks for the prediction-based and within-entry metrics
PC = pd.read_csv(f'{TAB}/pred_compare_chain.csv', dtype={'chain': str})
meta = s.groupby('pdb').agg(res=('res', 'first'), ybin=('ybin', 'first'), year=('year', 'first'), sars2=('sars2', 'max'))
PC = PC.join(meta, on='pdb'); PC['case'] = (PC.role_cc == 'case').astype(int); PC['sars2'] = PC.sars2.astype(float)
PC['cclass'] = PC.cclass.replace({'K': 'VL', 'L': 'VL'})
E = pd.read_csv(f'{TAB}/entry_vrpt.csv').join(meta, on='pdb').join(s.groupby('pdb').role_cc.first(), on='pdb')
E['case'] = (E.role_cc == 'case').astype(int); E['sars2'] = E.sars2.astype(float); E['cclass'] = 'x'
rows = []
for y, df in [('fw_rmsd_sabr', PC), ('cdr_bin_agree', PC), ('pred_err_site', PC), ('gap_rama_allowed', E), ('gap_clash', E)]:
    fit(y, df, '', False, 'base')
    fit(y, df, ' + sars2', False, '+ SARS-CoV-2')
    fit(y, df[df.year.between(2020, 2022)], '', False, '2020-2022 only')
S2 = pd.DataFrame(rows)
S2.to_csv(f'{TAB}/sensitivity_pred_entry.csv', index=False)
print(S2.round(4).to_string())
