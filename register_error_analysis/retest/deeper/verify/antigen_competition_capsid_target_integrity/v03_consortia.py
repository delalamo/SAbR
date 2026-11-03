# Curated consortia (from audit + citation author lists, see v02 output) and leave-one-consortium-out,
# with original and "robust" error definitions; plus unit-level (antibody-in-paper) check.
import warnings, numpy as np, pandas as pd, statsmodels.formula.api as smf, statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings('ignore')
pd.set_option('display.width', 250)
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/antigen_competition_capsid_target_integrity'
EM = pd.read_csv(f'{OUT}/em_entries_verify.csv.gz')
bov = set(EM[EM.ab_orgs.fillna('').str.contains('Bos taurus') | EM.title.fillna('').str.contains(r'bovine|cattle|\bcow\b', case=False, regex=True)].pdb)
EM['robust_drop'] = (EM.y == 1) & EM.all_q | EM.pdb.isin(bov)
# curated groups: corresponding authors that co-author each other's capsid papers
CONS = {
 'LOU (Lou Z. / Li K., FMDV, bovine Abs)': ['Lou, Z.', 'Kun, L.', 'Li, K.'],
 'WANGX (Wang X. IBP / Rao Z.: Feng R., Fu W. E3 papers)': ['Wang, X.', 'Feng, R.', 'Fu, W.', 'Rao, Z.', 'Wang, N.', 'Yu, Q.'],
 'XIAMEN (Xia N. / Li S.W. / Cheng T.)': ['Li, S.W.', 'Li, S.', 'Cheng, T.', 'Xia, N.', 'Xia, N.S.', 'Zheng, Q.'],
 'HAFENSTEIN': ['Hafenstein, S.', 'Hafenstein, S.L.', 'Charnesky, A.J.'],
 'ZHU P.': ['Zhu, P.', 'Hou, B.D.'],
}
lab2c = {l: c for c, ls in CONS.items() for l in ls}
EM['cons'] = EM.last_author.map(lab2c).fillna('LAB:' + EM.last_author.fillna(EM.paper))
def glm(d, f='y ~ cap + C(ycat) + res', grp='paper'):
    mod = smf.glm(f, d, family=sm.families.Binomial()).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(d[grp])[0]})
    ci = mod.conf_int().loc['cap']; return np.exp(mod.params['cap']), np.exp(ci[0]), np.exp(ci[1]), mod.pvalues['cap']
def rate(d): return f'{100*d.y.mean():.1f}% ({int(d.y.sum())}/{len(d)})'
rows = []
def rep(label, d, grp='paper'):
    r = glm(d, grp=grp); a = d[d.cap == 1]; b = d[d.cap == 0]
    rows.append(dict(test=label, capsid=rate(a), other=rate(b), OR=r[0], lo=r[1], hi=r[2], p=r[3], n=len(d), n_err=int(d.y.sum())))
    print(f'  {label:70s} capsid {rate(a):>15s} other {rate(b):>17s} OR {r[0]:.2f} ({r[1]:.2f}-{r[2]:.2f}) p={r[3]:.2g}')
for nm, D in [('original', EM), ('robust', EM[~EM.robust_drop])]:
    print(f'\n== {nm} errors: capsid error entries / capsid entries by curated consortium')
    c = D[D.cap == 1].groupby('cons').agg(entries=('y', 'size'), err=('y', 'sum'), papers=('paper', 'nunique'), err_papers=('paper', lambda s: D.loc[s.index][D.loc[s.index].y == 1].paper.nunique())).sort_values('err', ascending=False)
    print(c[c.err > 0].to_string()); print('  capsid consortia/labs with >=1 error:', int((c.err > 0).sum()), 'of', len(c))
    rep(f'{nm}: all', D)
    rep(f'{nm}: SE clustered by curated consortium', D, grp='cons')
    top = list(c[c.err > 0].index)
    for k in top:
        rep(f'{nm}: drop {k[:45]}', D[D.cons != k])
    rep(f'{nm}: drop top-2 ({", ".join(x.split(" ")[0] for x in top[:2])})', D[~D.cons.isin(top[:2])])
    rep(f'{nm}: drop top-3 ({", ".join(x.split(" ")[0] for x in top[:3])})', D[~D.cons.isin(top[:3])])
    # consortium-level unit analysis (a consortium's capsid work is one unit; non-capsid units = labs)
    P = D.groupby(['cons', 'cap']).agg(y=('y', 'max'), n=('y', 'size'), res=('res', 'median'), ycat=('ycat', lambda s: s.mode().iat[0])).reset_index(); P['ln'] = np.log2(P.n)
    a = int(P[P.cap == 1].y.sum()); na = int((P.cap == 1).sum()); cc = int(P[P.cap == 0].y.sum()); nc = int((P.cap == 0).sum())
    orf, pf = stats.fisher_exact([[a, na - a], [cc, nc - cc]])
    mod = smf.glm('y ~ cap + C(ycat) + res + ln', P, family=sm.families.Binomial()).fit()
    rows.append(dict(test=f'{nm}: consortium x capsid units, logit +year+res+log2(n)', capsid=f'{a}/{na}', other=f'{cc}/{nc}', OR=np.exp(mod.params['cap']), lo=np.exp(mod.conf_int().loc['cap', 0]), hi=np.exp(mod.conf_int().loc['cap', 1]), p=mod.pvalues['cap'], n=len(P), n_err=int(P.y.sum())))
    rows.append(dict(test=f'{nm}: consortium x capsid units, Fisher', capsid=f'{a}/{na}', other=f'{cc}/{nc}', OR=orf, lo=np.nan, hi=np.nan, p=pf, n=len(P), n_err=int(P.y.sum())))
    print(f'  consortium-level units: capsid {a}/{na} vs other {cc}/{nc}; Fisher OR {orf:.2f} p={pf:.2g}; adjusted OR {np.exp(mod.params["cap"]):.2f} p={mod.pvalues["cap"]:.2g}')

# unit level (antibody in paper), EM: unit has first-occurrence error vs no run of any kind
ch = pd.read_csv(f'{SP}/deeper/chains.csv.gz', usecols=['pdb', 'unit', 'method', 'res', 'deposit', 'err_first', 'err_any'])
ch = ch[ch.pdb.isin(EM.pdb)]
ch = ch.merge(EM[['pdb', 'cap', 'all_q', 'paper']], on='pdb')
ch['bov'] = ch.pdb.isin(bov)
U = ch.groupby('unit').agg(err=('err_first', 'max'), anyrun=('err_any', 'max'), cap=('cap', 'max'), res=('res', 'median'), dy=('deposit', lambda s: pd.to_datetime(s).dt.year.min()),
                           allq=('all_q', 'max'), bov=('bov', 'max'), paper=('paper', 'first')).reset_index()
U = U[U.err | ~U.anyrun].copy(); U['y'] = U.err.astype(int); U['ycat'] = U.dy.clip(lower=2017).astype(str)
print('\n== Unit level (antibody in paper), EM')
rep('units, original errors', U)
Ur = U[~((U.y == 1) & U.allq) & ~U.bov]
rep('units, robust errors', Ur)
T = pd.DataFrame(rows); T['q_BH_within_file'] = multipletests(T.p, method='fdr_bh')[1]
T.to_csv(f'{OUT}/consortium_tests.csv', index=False)
