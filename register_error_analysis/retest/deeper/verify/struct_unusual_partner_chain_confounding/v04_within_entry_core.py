"""(a) Symmetric tail checks for the partner clash effect (trim/winsorise both groups, not only cases).
(b) Is the within-entry subset representative?  Partner-vs-control effect restricted to case partners that have other
    error-free same-class V chains in the same entry, and the effect for those other chains themselves.
(c) Domain-core distortion (core-set CA RMSD to the ABangle consensus, SAbR numbering) for EVERY V chain in the sample
    entries, so the partner can be compared with (i) controls, adjusted for chain Q / resolution / year / paper
    clustering, and (ii) other error-free chains of the same class in the same entry.
Output tables/tails.csv, tables/core_all_chains.csv, tables/core_tests.csv"""
import sys, pickle, warnings, numpy as np, pandas as pd, statsmodels.formula.api as smf
from scipy import stats
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/user/SAbR/register_error_analysis/retest/deeper/struct_unusual/scripts')
import common  # read-only use of the investigator's ABangle frame helpers
SP = common.SP
CACHE = f'{SP}/deeper/cache/struct_unusual'
INV = '/home/user/SAbR/register_error_analysis/retest/deeper/struct_unusual/tables'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/struct_unusual_partner_chain_confounding/tables'
BASE = 'C(cclass) + C(ybin) + res'

U = pd.read_csv(f'{INV}/unit_metrics.csv', dtype={'chain': str, 'partner': str})
U['case'] = (U.role_cc == 'case').astype(int)
PU = U[U.partner_err_any == False].copy()


def ols(y, x, rhs=BASE, grp='paper'):
    x = x.dropna(subset=[y])
    m = smf.ols(f'{y} ~ case + {rhs}', x).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(x[grp])[0]})
    return dict(n_case=int(x.case.sum()), n_ctrl=int((1 - x.case).sum()), case_mean=x[x.case == 1][y].mean(),
                ctrl_mean=x[x.case == 0][y].mean(), effect=m.params['case'], ci_lo=m.conf_int().loc['case', 0],
                ci_hi=m.conf_int().loc['case', 1], p=m.pvalues['case'])


rows = []
# (a) symmetric tails
for y in ['p_clash_V', 'p_rama_allowed_V']:
    for q in (0.95, 0.90):
        cut = PU[y].quantile(q)
        x = PU.copy(); x[y] = x[y].clip(upper=cut)
        rows.append(dict(metric=y, model=f'winsorise both groups at pooled {int(q*100)}th pct ({cut:.1f})', **ols(y, x)))
        keep = pd.concat([g[g[y] <= g[y].quantile(q)] for _, g in PU.groupby('case')])
        rows.append(dict(metric=y, model=f'trim top {int(round((1-q)*100))}% within each group', **ols(y, keep)))
    x = PU.copy(); x['above_ctrl_p90'] = (x[y] > PU[PU.case == 0][y].quantile(0.9)).astype(float)
    rows.append(dict(metric=y, model='share above control 90th pct (LPM)', **ols('above_ctrl_p90', x)))

# (b) representativeness of the within-entry subset
W = pd.read_csv(f'{OUT}/within_entry.csv', dtype={'partner': str})
wk = set(zip(W[W.case == 1].pdb, W[W.case == 1].partner))
PU['has_within'] = [(p, c) in wk for p, c in zip(PU.pdb, PU.partner)]
x = PU[(PU.case == 0) | PU.has_within]
rows.append(dict(metric='p_clash_V', model='case partners WITH other clean same-class chains in the entry vs controls', **ols('p_clash_V', x)))
x = PU[(PU.case == 0) | ~PU.has_within]
rows.append(dict(metric='p_clash_V', model='case partners WITHOUT other clean same-class chains vs controls', **ols('p_clash_V', x)))
T = pd.DataFrame(rows); T.to_csv(f'{OUT}/tails.csv', index=False)
pd.set_option('display.width', 250)
print(T.round(4).to_string())

# (c) core RMSD for every V chain
V = pickle.load(open(f'{CACHE}/vchains.pkl', 'rb'))
ch = pd.read_csv(f'{SP}/deeper/chains.csv.gz', dtype={'chain': str},
                 usecols=['pdb', 'chain', 'ctype', 'vhh', 'err_any', 'err_first', 'seq', 'res', 'year', 'paper'])
chk = ch.set_index(['pdb', 'chain'])
Rr = pd.read_pickle(f'{CACHE}/residues.pkl'); Rr['chain'] = Rr.chain.astype(str)
Rr['clash'] = (Rr.nclash.fillna(0) > 0) * 100.0
cq = Rr.groupby(['pdb', 'chain']).agg(q=('q', 'mean'), clash=('clash', 'mean')).reset_index()
crow = []
for (p, c), rec in V.items():
    if (p, c) not in chk.index:
        continue
    r = chk.loc[(p, c)]
    ct = r.ctype
    ca = {}
    for x in rec['res']:
        n = x['sabr']
        if n and n[1] == '' and 'CA' in x['atoms']:
            ca[n[0]] = x['atoms']['CA']
    f = common._frame(ca, ct, 'H' if ct == 'H' else 'L')
    crow.append(dict(pdb=p, chain=c, ctype=ct, vhh=r.vhh, err_any=bool(r.err_any), err_first=bool(r.err_first),
                     core_rms=f[4] if f else np.nan, seq=r.seq, res=r.res, year=r.year, paper=r.paper))
CA = pd.DataFrame(crow).merge(cq, on=['pdb', 'chain'], how='left')
CA['cls'] = np.where(CA.vhh == 1, 'VHH', np.where(CA.ctype == 'H', 'H', 'VL'))
CA.drop(columns=['seq']).to_csv(f'{OUT}/core_all_chains.csv', index=False)
cak = CA.set_index(['pdb', 'chain'])

# partner rows (error-free partner of a case unit) and control partner rows
PU['p_core'] = [cak.core_rms.get((p, c), np.nan) for p, c in zip(PU.pdb, PU.partner)]
PU['p_qq'] = [cak.q.get((p, c), np.nan) for p, c in zip(PU.pdb, PU.partner)]
tr = []
tr.append(dict(test='partner core RMSD vs control partners (cclass+ybin+res, cluster paper)', **ols('p_core', PU)))
tr.append(dict(test='... + partner Q', **ols('p_core', PU, BASE + ' + p_qq')))
tr.append(dict(test='... + partner Q + partner clash rate', **ols('p_core', PU, BASE + ' + p_qq + p_clash_V')))
tr.append(dict(test='... 2020-2022 only', **ols('p_core', PU[PU.year.between(2020, 2022)])))
# within-entry: partner core vs other error-free same-class chains in the same entry
wr = []
for _, u in PU.iterrows():
    me = cak.loc[(u.pdb, u.partner)] if (u.pdb, u.partner) in cak.index else None
    if me is None or not np.isfinite(me.core_rms):
        continue
    oth = CA[(CA.pdb == u.pdb) & (CA.cls == me.cls) & (CA.chain != u.partner) & (~CA.err_any) & CA.core_rms.notna()]
    if not len(oth):
        continue
    wr.append(dict(case=u.case, pdb=u.pdb, paper=u.paper, partner=u.partner, d_core=me.core_rms - oth.core_rms.mean(),
                   d_q=me.q - oth.q.mean(), n_oth=len(oth)))
WR = pd.DataFrame(wr).drop_duplicates(['pdb', 'partner'])
for c, x in WR.groupby('case'):
    tr.append(dict(test=f'within-entry partner minus other clean same-class chains, {"case" if c else "control"} entries',
                   n_case=len(x), effect=x.d_core.mean(), case_mean=x.d_core.median(), p=stats.wilcoxon(x.d_core).pvalue))
m = smf.ols('d_core ~ case', WR).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(WR.paper)[0]})
tr.append(dict(test='within-entry DiD (case - control)', n_case=int(WR.case.sum()), n_ctrl=int((1 - WR.case).sum()),
               effect=m.params['case'], ci_lo=m.conf_int().loc['case', 0], ci_hi=m.conf_int().loc['case', 1], p=m.pvalues['case']))
# all error-free V chains in case entries (not just the partner) vs chains in control entries
CA['case_entry'] = CA.pdb.isin(set(PU[PU.case == 1].pdb) | set(U[U.case == 1].pdb)).astype(int)
CA['ybin'] = pd.cut(CA.year, [0, 2019, 2020, 2021, 2022, 2023, 2024, 2100]).astype(str)
x = CA[~CA.err_any].rename(columns={'case_entry': 'case', 'cls': 'cclass'})
x = x[x.pdb.isin(set(U.pdb))]
for y in ['core_rms', 'clash']:
    tr.append(dict(test=f'ALL error-free V chains in case entries vs control entries: {y}', **ols(y, x)))
    tr.append(dict(test=f'... + chain Q: {y}', **ols(y, x, BASE + ' + q')))
TT = pd.DataFrame(tr); TT.to_csv(f'{OUT}/core_tests.csv', index=False)
print(TT.round(4).to_string())
