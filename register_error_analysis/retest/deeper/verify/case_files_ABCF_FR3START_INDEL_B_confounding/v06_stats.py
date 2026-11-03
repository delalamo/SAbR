"""Re-test T7 (ANARCI indel at IMGT 66-75 -> first-occurrence mode B) with confounder adjustment and artifact sensitivity.
Unit = antibody V region in one paper (chains.csv 'unit'); AL amyloid excluded. Error = unit with a first-occurrence B run;
control = unit with no disagreement run of any kind (as in the original) or, alternatively, any unit without a B run.
Output: tables/v06_stats.csv"""
import numpy as np, pandas as pd, statsmodels.formula.api as smf, statsmodels.api as sm, warnings
from scipy.stats import fisher_exact
from statsmodels.stats.contingency_tables import StratifiedTable
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings('ignore')
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/case_files_ABCF_FR3START_INDEL_B_confounding'
CH = pd.read_pickle(f'{OUT}/tables/chain_features.pkl')
def units(method):
    E = CH[CH.method == method].copy()
    E['ybin'] = pd.cut(E.year, [0, 2019, 2021, 2023, 2100], labels=['<=2019', '2020-21', '2022-23', '>=2024']).astype(str)
    U = E.groupby('unit').agg(paper=('paper', 'first'), lab=('last_author', 'first'), year=('year', 'first'), ybin=('ybin', 'first'),
                              res=('res', 'mean'), cclass=('cclass', 'first'), err_any=('err_any', 'max'), sars2=('sars2', 'max'),
                              modes=('modes_first', lambda s: set().union(*s)), feat=('fr3start_indel', 'max'),
                              feat_first=('fr3start_indel', 'first'), pdb=('pdb', 'first'), L2_atyp=('L2_atyp', 'max')).reset_index()
    U['yB'] = U.modes.map(lambda s: 'B_CDR2FR3' in s).astype(int)
    U['sars2'] = U.sars2.fillna(False).astype(int)
    U['feat'] = U.feat.fillna(0).astype(int); U['feat_first'] = U.feat_first.fillna(0).astype(int)
    return U
UE, UX = units('EM'), units('X-ray')
# artifact-reproduced units (v03 predicted models by both tools and/or v05 same-pattern high-res X-ray reproduce the run)
# exact reproduction: the deposited SAbR-vs-ANARCI per-residue pattern recurs identically on map-free predicted models (IB and IgFold)
# and/or on a <2.6 A X-ray structure with the same ANARCI indel (9pyn = same antibody as 9pyh, 2.34 A)
ART = {'8bw0', '8r4d', '9pyh', '8snc'}
AMBIG = {'9om5', '8jag'}   # predicted models also disagree in the window, but with a different placement; 9om5: 3/6 NCS copies agree
OFFSITE = {'22ge'}                                    # B run at IMGT 53-60, not at the 69A insertion; not reproduced
rows = []
def fit(name, d, x='feat', covs='C(ybin) + res + C(cclass)', group='paper'):
    d = d.copy()
    ne, nc = int(d.y.sum()), int((d.y == 0).sum())
    a, b = int(d[d.y == 1][x].sum()), int(d[d.y == 0][x].sum())
    t = [[a, ne - a], [b, nc - b]]
    orf, pf = fisher_exact(t)
    r = dict(test=name, n_error=ne, n_control=nc, error_value=f'{a}/{ne} ({a/ne:.1%})', control_value=f'{b}/{nc} ({b/nc:.2%})',
             OR_fisher=orf, p_fisher=pf)
    # Mantel-Haenszel over year bin x chain class strata (exact-ish, no model)
    try:
        tabs = []
        for _, g in d.groupby(['ybin', 'cclass']):
            tt = np.array([[g[(g.y == 1)][x].sum(), ((g.y == 1) & (g[x] == 0)).sum()], [g[(g.y == 0)][x].sum(), ((g.y == 0) & (g[x] == 0)).sum()]], float)
            if tt[0].sum() > 0 and tt[1].sum() > 0: tabs.append(tt)
        st = StratifiedTable(tabs)
        r.update(OR_MH=st.oddsratio_pooled, p_MH=st.test_null_odds(correction=True).pvalue)
    except Exception as ex: r.update(note_MH=repr(ex)[:80])
    try:
        if min(a, ne - a, b, nc - b) == 0: raise ValueError('zero cell')
        m = smf.logit(f'y ~ {x} + {covs}', data=d).fit(disp=0, cov_type='cluster', cov_kwds={'groups': pd.factorize(d[group])[0]})
        r.update(OR_adj=np.exp(m.params[x]), lo=np.exp(m.conf_int().loc[x, 0]), hi=np.exp(m.conf_int().loc[x, 1]), p_adj=m.pvalues[x],
                 covs=covs + f' | cluster={group}')
    except Exception as ex: r.update(note_adj=repr(ex)[:80])
    rows.append(r); print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
def base(U, ctrl='clean', drop=()):
    d = U.copy(); d['y'] = d.yB
    d = d[~d.pdb.isin(drop) | (d.y == 0)] if drop else d
    d = d[~((d.y == 1) & d.pdb.isin(drop))]
    d = d[(d.y == 1) | (d.err_any == False)] if ctrl == 'clean' else d
    d = d[d.cclass.isin(set(d[d.y == 1].cclass))]
    return d
# 1 reproduction
d = base(UE); fit('EM reproduce T7 (feature of first chain; covs ybin,res,cclass; cluster paper)', d, x='feat_first')
fit('EM T7 feature any chain of unit', d)
fit('EM T7 + SARS-CoV-2 covariate', d, covs='C(ybin) + res + C(cclass) + sars2')
fit('EM T7 cluster by last author', d, group='lab')
fit('EM T7 + atypical CDR2 length covariate', d, covs='C(ybin) + res + C(cclass) + L2_atyp')
fit('EM T7 heavy/VHH only', d[d.cclass.isin(['H', 'VHH'])])
fit('EM T7 non-SARS-CoV-2 only', d[d.sars2 == 0], covs='C(ybin) + res + C(cclass)')
fit('EM T7 deposited 2022+ only', d[d.year >= 2022], covs='res + C(cclass)')
# 2 alternative control: every unit without a first-occurrence B run (incl. other-mode errors and artifact/copy-only units)
fit('EM T7 control = all non-B units', base(UE, ctrl='all'))
# 3 artifact sensitivity
fit('EM T7 excluding 4 units whose run is reproduced exactly map-free (predictions / same-pattern hires X-ray)', base(UE, drop=ART))
fit('EM T7 excluding exact + ambiguous sequence-driven units (6)', base(UE, drop=ART | AMBIG))
fit('EM T7 excluding all sequence-driven + off-site run (22ge, run 53-60 not at the 69A insertion)', base(UE, drop=ART | AMBIG | OFFSITE))
# 4 X-ray 3-5 A (method check)
fit('X-ray 3-5 A T7', base(UX))
T = pd.DataFrame(rows)
T['q_fisher'] = multipletests(T.p_fisher.fillna(1), method='fdr_bh')[1]
T.to_csv(f'{OUT}/tables/v06_stats.csv', index=False)
pd.set_option('display.width', 250)
print(T[['test', 'error_value', 'control_value', 'OR_fisher', 'p_fisher', 'q_fisher', 'OR_MH', 'p_MH', 'OR_adj', 'lo', 'hi', 'p_adj']].round(5).to_string())
print('\nX-ray feature units:'); print(UX[UX.feat == 1][['pdb', 'cclass', 'year', 'res', 'err_any', 'modes', 'lab']].to_string())
print('\nEM feature units by year:'); print(UE[UE.feat == 1].groupby(['ybin', 'yB']).size())
print('EM B units by ybin:'); print(base(UE).groupby(['ybin', 'y']).size().unstack())
