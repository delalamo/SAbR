"""Recompute the lambda CDR1-length signal (L1=8 vs L1=9) with and without the AL fibrils, under several
plausible definitions (chain vs unit; all methods vs EM; control = all non-error vs clean-only), plus an
adjusted model (year bin, resolution, method) with paper-clustered SEs and a leave-one-paper-out check."""
import numpy as np, pandas as pd, statsmodels.formula.api as smf, statsmodels.api as sm
from scipy import stats
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad/deeper'
AMY = {'6ic3', '6z1o', '7nsl', '9els', '9faa', '9fab', '9fac', '9liv', '9liw', '9lix', '9liy', '9lj0', '9lja', '9ljb', '9v91'}
ch = pd.read_csv(f'{SP}/chains.csv.gz')
ch['amy'] = ch.pdb.isin(AMY)
lam = ch[ch.ctype == 'L'].copy()

def rate(df):
    return f'{df.err_first.mean()*100:.1f}% ({df.err_first.sum()}/{len(df)})'

rows = []
for meth in ['all', 'EM']:
    for ctrl in ['all_nonerr', 'clean_only']:
        for amy in ['with', 'without']:
            d = lam if meth == 'all' else lam[lam.method == 'EM']
            if ctrl == 'clean_only':
                d = d[d.err_first | ~d.err_any]
            if amy == 'without':
                d = d[~d.amy]
            for level in ['chain', 'unit']:
                x = d if level == 'chain' else d.groupby('unit').agg(err_first=('err_first', 'max'), L1=('L1', 'first')).reset_index()
                a = x[x.L1 == 8]; b = x[x.L1 == 9]
                t = stats.fisher_exact([[a.err_first.sum(), len(a) - a.err_first.sum()], [b.err_first.sum(), len(b) - b.err_first.sum()]])
                rows.append(dict(method=meth, control=ctrl, amyloid=amy, level=level, L1_8=rate(a), L1_9=rate(b), OR=t[0], p_fisher=t[1]))
R = pd.DataFrame(rows)
pd.set_option('display.width', 250)
print(R.to_string(index=False))

# adjusted chain-level model (EM, clean-only control) with paper clusters, with/without amyloid
for amy in ['with', 'without']:
    d = lam[(lam.method == 'EM') & (lam.err_first | ~lam.err_any) & lam.L1.isin([8, 9])].copy()
    if amy == 'without':
        d = d[~d.amy]
    d['y'] = d.err_first.astype(int); d['L1_8'] = (d.L1 == 8).astype(int)
    d['yb'] = pd.cut(pd.to_datetime(d.deposit).dt.year, [0, 2018, 2019, 2020, 2021, 2022, 2023, 3000]).astype(str)
    d['cl'] = d.paper.fillna(d.pdb)
    m = smf.glm('y ~ L1_8 + C(yb) + res', data=d, family=sm.families.Binomial()).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(d.cl)[0]})
    print(f'EM lambda chains, clean-only control, {amy} amyloid: n={len(d)} err={d.y.sum()} adj OR(L1=8)={np.exp(m.params.L1_8):.2f} p={m.pvalues.L1_8:.3g}')
    # unit-level (one row per unit)
    u = d.groupby('unit').agg(y=('y', 'max'), L1_8=('L1_8', 'first'), yb=('yb', 'first'), res=('res', 'mean'), cl=('cl', 'first')).reset_index()
    try:
        m2 = smf.glm('y ~ L1_8 + C(yb) + res', data=u, family=sm.families.Binomial()).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(u.cl)[0]})
        print(f'   unit level: n={len(u)} err={u.y.sum()} rate L1=8 {u[u.L1_8==1].y.mean()*100:.1f}% vs L1=9 {u[u.L1_8==0].y.mean()*100:.1f}%  adj OR={np.exp(m2.params.L1_8):.2f} p={m2.pvalues.L1_8:.3g}')
    except Exception as e:
        print('unit model failed', e)
    # leave-one-paper-out on the without-amyloid unit Fisher test
    if amy == 'without':
        ps = []
        for p in u[u.y == 1].cl.unique():
            uu = u[u.cl != p]
            a = uu[uu.L1_8 == 1]; b = uu[uu.L1_8 == 0]
            ps.append((p, stats.fisher_exact([[a.y.sum(), len(a) - a.y.sum()], [b.y.sum(), len(b) - b.y.sum()]])[1]))
        ps = pd.DataFrame(ps, columns=['dropped_paper', 'p'])
        print('   LOPO Fisher p (units, without amyloid): min %.3g median %.3g max %.3g' % (ps.p.min(), ps.p.median(), ps.p.max()))
        a = u[u.L1_8 == 1]; b = u[u.L1_8 == 0]
        print('   L1=8 error units:', a.y.sum(), 'of', len(a), '; L1=9 error units:', b.y.sum(), 'of', len(b))
        print('   L1=8 error units list:', d[(d.y == 1) & (d.L1_8 == 1)][['pdb', 'v_gene', 'err_modes', 'last_author']].drop_duplicates().to_string(index=False))
R.to_csv('/home/user/SAbR/register_error_analysis/retest/deeper/verify/pdb_meta_amyloid_artifact_integrity/lambda_L1_variants.csv', index=False)
