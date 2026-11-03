# Extra sensitivity checks: (a) chain-level robust flag for unit-level test, (b) corrected capsid class + robust + drop the two
# groups carrying the robust signal, (c) lighter artifact filter that does not blanket-flag bovine chains, (d) Prasad norovirus runs.
import warnings, numpy as np, pandas as pd, statsmodels.formula.api as smf, statsmodels.api as sm
warnings.filterwarnings('ignore')
pd.set_option('display.width', 250)
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/antigen_competition_capsid_target_integrity'
EM = pd.read_csv(f'{OUT}/em_entries_verify.csv.gz')
bov = set(EM[EM.ab_orgs.fillna('').str.contains('Bos taurus') | EM.title.fillna('').str.contains(r'bovine|cattle|\bcow\b', case=False, regex=True)].pdb)
runs = pd.read_csv(f'{SP}/deeper/runs_first_occurrence.csv'); runs = runs[runs.pdb.isin(EM.pdb)].copy()
runs['qa'] = (runs['mode'] == 'D0') | ((runs['mode'] == 'F_CDR1') & runs.delta.isna()) | runs.pdb.isin(bov)
runs['qb'] = (runs['mode'] == 'D0') | ((runs['mode'] == 'F_CDR1') & runs.delta.isna()) | (runs.delta.abs() >= 5)   # no bovine blanket
CONS = {'HAFENSTEIN': ['Hafenstein, S.', 'Hafenstein, S.L.', 'Charnesky, A.J.'], 'ZHU P.': ['Zhu, P.', 'Hou, B.D.']}
lab2c = {l: c for c, ls in CONS.items() for l in ls}
EM['cons'] = EM.last_author.map(lab2c).fillna('other')
def glm(d, f='y ~ cap + C(ycat) + res', grp='paper'):
    mod = smf.glm(f, d, family=sm.families.Binomial()).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(d[grp])[0]})
    ci = mod.conf_int().loc['cap']; return np.exp(mod.params['cap']), np.exp(ci[0]), np.exp(ci[1]), mod.pvalues['cap']
def rate(d): return f'{100*d.y.mean():.1f}% ({int(d.y.sum())}/{len(d)})'
rows = []
def rep(label, d, grp='paper'):
    r = glm(d, grp=grp); a = d[d.cap == 1]; b = d[d.cap == 0]
    rows.append(dict(test=label, capsid=rate(a), other=rate(b), OR=r[0], lo=r[1], hi=r[2], p=r[3], n=len(d), n_err=int(d.y.sum())))
    print(f'  {label:72s} capsid {rate(a):>15s} other {rate(b):>17s} OR {r[0]:.2f} ({r[1]:.2f}-{r[2]:.2f}) p={r[3]:.2g}')
qb = runs.groupby('pdb').qb.all()
EM['all_qb'] = EM.pdb.map(qb).fillna(False).astype(bool)
print('== (c) lighter filter (D0, within-CDR1 gap placement, >=5-position shifts; bovine chains not blanket-flagged)')
print('   capsid error entries dropped:', ' '.join(sorted(EM[(EM.y == 1) & EM.all_qb & (EM.cap == 1)].pdb)), '| non-capsid dropped:', ' '.join(sorted(EM[(EM.y == 1) & EM.all_qb & (EM.cap == 0)].pdb)))
Dc = EM[~((EM.y == 1) & EM.all_qb)]
rep('lighter filter', Dc)
rep('lighter filter, drop Hafenstein + Zhu P.', Dc[Dc.cons == 'other'])
print('\n== (b) corrected class (+ norovirus VP1 / rotavirus VP4) with robust filter, then drop the two groups')
d = EM.copy(); d['cap'] = d.cap_fix
dr = d[~(((d.y == 1) & d.all_q) | d.pdb.isin(bov))]
rep('corrected class, robust', dr)
rep('corrected class, robust, drop Hafenstein', dr[dr.cons != 'HAFENSTEIN'])
rep('corrected class, robust, drop Zhu P.', dr[dr.cons != 'ZHU P.'])
rep('corrected class, robust, drop Hafenstein + Zhu P.', dr[dr.cons == 'other'])
rep('corrected class, robust, drop Hafenstein + Zhu P. + Prasad', dr[(dr.cons == 'other') & (dr.last_author != 'Prasad, B.V.V.')])
print('\n== (d) Prasad norovirus error runs')
print(runs[runs.last_author == 'Prasad, B.V.V.'][['pdb', 'chain', 'mode', 'imgt0', 'imgt1', 'delta', 'length', 'origin']].to_string())
print('\n== (a) unit level with chain-level robust flag')
ch = pd.read_csv(f'{SP}/deeper/chains.csv.gz', usecols=['pdb', 'chain', 'unit', 'res', 'deposit', 'err_first', 'err_any'])
ch = ch[ch.pdb.isin(EM.pdb)].merge(EM[['pdb', 'cap', 'paper', 'last_author']], on='pdb')
cq = runs.groupby(['pdb', 'chain']).qa.all().rename('chain_allq').reset_index()
ch = ch.merge(cq, on=['pdb', 'chain'], how='left')
ch['robust_err'] = ch.err_first & ~ch.chain_allq.fillna(False).astype(bool) & ~ch.pdb.isin(bov)
U = ch.groupby('unit').agg(err=('err_first', 'max'), rerr=('robust_err', 'max'), anyrun=('err_any', 'max'), cap=('cap', 'max'), res=('res', 'median'),
                           dy=('deposit', lambda s: pd.to_datetime(s).dt.year.min()), paper=('paper', 'first'), bov=('pdb', lambda s: bool(set(s) & bov)), la=('last_author', 'first')).reset_index()
U['ycat'] = U.dy.clip(lower=2017).astype(str)
Uo = U[U.err | ~U.anyrun].copy(); Uo['y'] = Uo.err.astype(int)
rep('units, original errors', Uo)
Ur = U[(U.rerr | ~U.anyrun) & ~U.bov].copy(); Ur['y'] = Ur.rerr.astype(int)
rep('units, robust errors (chain-level flag)', Ur)
Ur['cons'] = Ur.la.map(lab2c).fillna('other')
rep('units, robust errors, drop Hafenstein + Zhu P.', Ur[Ur.cons == 'other'])
pd.DataFrame(rows).to_csv(f'{OUT}/extra_tests.csv', index=False)
