"""Rabbit antibodies (ANARCI all-species call, ../seq_novelty) and first-occurrence errors, cryo-EM units.
Lead from the case files: 9mnb (mode A) and 9ml5 (mode F) are rabbit Fabs; the near-miss control 9ml7 is rabbit too.
Output: tables/rabbit_tests.csv"""
import pandas as pd, numpy as np, statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/case_files_ABCF'
C = pd.read_csv(f'{SP}/deeper/chains.csv.gz', low_memory=False)
N = pd.read_csv('/home/user/SAbR/register_error_analysis/retest/deeper/seq_novelty/tables/chain_novelty.csv.gz', usecols=['pdb', 'chain', 'q_species_all'])
amy = set(pd.read_csv('/home/user/SAbR/register_error_analysis/retest/deeper/pdb_meta/tables/amyloid_fibril_chains.csv').pdb.str.lower())
C = C.merge(N, on=['pdb', 'chain'], how='left'); C = C[(C.method == 'EM') & ~C.pdb.isin(amy)]
C['rabbit'] = (C.q_species_all == 'rabbit').astype(int)
C['cclass'] = np.where(C.ctype == 'H', np.where(C.vhh == 1, 'VHH', 'H'), C.ctype)
C['ybin'] = pd.cut(C.year, [0, 2019, 2021, 2023, 2100], labels=['a', 'b', 'c', 'd']).astype(str)
U = C.groupby('unit').agg(paper=('paper', 'first'), rabbit=('rabbit', 'max'), err_first=('err_first', 'max'), err_any=('err_any', 'max'),
                          modes=('err_modes', lambda s: ';'.join(s.dropna())), res=('res', 'mean'), ybin=('ybin', 'first'), cclass=('cclass', 'first')).reset_index()
rows = []
for lab, sel in [('any first-occurrence error', lambda d: d.err_first == True), ('A_FR1Nterm', lambda d: d.modes.str.contains('A_FR1Nterm')),
                 ('F_CDR1', lambda d: d.modes.str.contains('F_CDR1')), ('B_CDR2FR3', lambda d: d.modes.str.contains('B_CDR2FR3')), ('C_FR3', lambda d: d.modes.str.contains('C_FR3'))]:
    d = U.copy(); d['y'] = (sel(d) & (d.err_first == True)).astype(int); d = d[(d.y == 1) | (d.err_any == False)]
    d = d[d.cclass.isin(['H', 'K', 'L'])]
    a = d[d.y == 1].rabbit; b = d[d.y == 0].rabbit
    try:
        m = smf.logit('y ~ rabbit + ybin + res + cclass', data=d).fit(disp=0, cov_type='cluster', cov_kwds={'groups': pd.factorize(d.paper)[0]})
        orr, (lo, hi), p = np.exp(m.params['rabbit']), np.exp(m.conf_int().loc['rabbit']), m.pvalues['rabbit']
    except Exception as e:
        orr = lo = hi = p = np.nan
    rows.append(dict(test=f'rabbit V region -> {lab}', n_error=len(a), n_control=len(b), error_value=f'{a.mean():.1%} ({a.sum()}/{len(a)})',
                     control_value=f'{b.mean():.2%} ({b.sum()}/{len(b)})', OR=orr, lo=lo, hi=hi, p=p))
R = pd.DataFrame(rows); R['q'] = multipletests(R.p.fillna(1), method='fdr_bh')[1]; R.to_csv(f'{OUT}/tables/rabbit_tests.csv', index=False)
print(R.round(4).to_string())
print(C[C.rabbit == 1].groupby(['year']).size().to_dict())
