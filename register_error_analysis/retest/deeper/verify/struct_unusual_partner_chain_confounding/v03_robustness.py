"""Further robustness checks for 'partner_chain':
 (1) model form: binomial GLM on clash-residue counts, median regression, log(1+y);
 (2) influence: leave-one-paper-out, leave-one-lab-out (cases), dropping the 5/10 highest case partners;
 (3) case x resolution interaction;
 (4) within-entry contrast: partner vs other error-free V chains of the same type in the SAME entry
     (removes all entry-level confounders: map, resolution, year, lab, software, SARS-CoV-2);
 (5) the 'intermediate core RMSD' detail, recomputed with adjustment and paper clustering.
Output tables/robustness.csv, tables/within_entry.csv, tables/core_rmsd.csv"""
import warnings, numpy as np, pandas as pd, statsmodels.formula.api as smf, statsmodels.api as sm
from scipy import stats
warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
CACHE = f'{SP}/deeper/cache/struct_unusual'
INV = '/home/user/SAbR/register_error_analysis/retest/deeper/struct_unusual/tables'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/struct_unusual_partner_chain_confounding/tables'

U = pd.read_csv(f'{INV}/unit_metrics.csv', dtype={'chain': str, 'partner': str})
s = pd.read_csv(f'{INV}/sample.csv', dtype={'chain': str})
U = U.merge(s[['unit', 'pdb', 'sars2', 'last_author']].drop_duplicates(['unit', 'pdb']), on=['unit', 'pdb'], how='left')
U['case'] = (U.role_cc == 'case').astype(int)
PU = U[U.partner_err_any == False].copy()
PU['k_clash'] = np.round(PU.p_clash_V * PU.p_n_V / 100).astype(int)
PU['k_allw'] = np.round(PU.p_rama_allowed_V * PU.p_n_V / 100).astype(int)
BASE = 'C(cclass) + C(ybin) + res'
g = pd.factorize(PU.paper)[0]
rows = []


def ols(y, df, rhs=BASE, label='', grp='paper'):
    x = df.dropna(subset=[y]).copy()
    m = smf.ols(f'{y} ~ case + {rhs}', x).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(x[grp])[0]})
    return dict(metric=y, model=label, n_case=int(x.case.sum()), n_ctrl=int((1 - x.case).sum()), effect=m.params['case'],
                ci_lo=m.conf_int().loc['case', 0], ci_hi=m.conf_int().loc['case', 1], p=m.pvalues['case'])


for y, k in [('p_clash_V', 'k_clash'), ('p_rama_allowed_V', 'k_allw')]:
    # binomial GLM on counts out of n residues (odds ratio), cluster by paper
    x = PU.dropna(subset=[y]).copy()
    X = pd.get_dummies(x[['case', 'cclass', 'ybin', 'res']], columns=['cclass', 'ybin'], drop_first=True).astype(float)
    X = sm.add_constant(X)
    m = sm.GLM(np.column_stack([x[k], x.p_n_V - x[k]]), X, family=sm.families.Binomial()).fit(
        cov_type='cluster', cov_kwds={'groups': pd.factorize(x.paper)[0]})
    rows.append(dict(metric=y, model='binomial GLM (OR per residue), cluster paper', n_case=int(x.case.sum()),
                     n_ctrl=int((1 - x.case).sum()), effect=np.exp(m.params['case']),
                     ci_lo=np.exp(m.conf_int().loc['case', 0]), ci_hi=np.exp(m.conf_int().loc['case', 1]), p=m.pvalues['case']))
    # quasi-binomial dispersion check
    m2 = sm.GLM(np.column_stack([x[k], x.p_n_V - x[k]]), X, family=sm.families.Binomial()).fit(scale='X2')
    rows.append(dict(metric=y, model='quasi-binomial GLM (OR), Pearson dispersion', effect=np.exp(m2.params['case']),
                     p=m2.pvalues['case'], note=f'dispersion {m2.scale:.1f}'))
    # median regression
    mq = smf.quantreg(f'{y} ~ case + {BASE}', x).fit(q=0.5)
    rows.append(dict(metric=y, model='median regression', effect=mq.params['case'], ci_lo=mq.conf_int().loc['case', 0],
                     ci_hi=mq.conf_int().loc['case', 1], p=mq.pvalues['case']))
    x['ly'] = np.log1p(x[y])
    rows.append(ols('ly', x, label=f'log(1+{y}) OLS, cluster paper'))
    rows.append(dict(metric=y, model='raw Mann-Whitney', p=stats.mannwhitneyu(x[x.case == 1][y], x[x.case == 0][y]).pvalue,
                     effect=x[x.case == 1][y].median() - x[x.case == 0][y].median(), note='median difference'))
    # case x resolution interaction
    mi = smf.ols(f'{y} ~ case*res + C(cclass) + C(ybin)', x).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(x.paper)[0]})
    rows.append(dict(metric=y, model='case x resolution (per A) interaction', effect=mi.params['case:res'], p=mi.pvalues['case:res']))
    # influence
    effs = []
    for pp in x[x.case == 1].paper.unique():
        r = ols(y, x[x.paper != pp]); effs.append((r['effect'], r['p']))
    effs = np.array(effs)
    rows.append(dict(metric=y, model='leave-one-case-paper-out: min / max effect, max p', effect=effs[:, 0].min(),
                     ci_hi=effs[:, 0].max(), p=effs[:, 1].max()))
    effs = []
    for la in x[x.case == 1].last_author.dropna().unique():
        r = ols(y, x[x.last_author != la]); effs.append((r['effect'], r['p']))
    effs = np.array(effs)
    rows.append(dict(metric=y, model='leave-one-case-lab-out: min / max effect, max p', effect=effs[:, 0].min(),
                     ci_hi=effs[:, 0].max(), p=effs[:, 1].max()))
    for kdrop in (5, 10):
        top = x[x.case == 1].nlargest(kdrop, y).index
        r = ols(y, x.drop(top), label=f'drop the {kdrop} highest case partners'); rows.append(r)
R = pd.DataFrame(rows)
R.to_csv(f'{OUT}/robustness.csv', index=False)
pd.set_option('display.width', 250)
print(R.round(4).to_string())

# ---------------- (4) within-entry contrast
Rr = pd.read_pickle(f'{CACHE}/residues.pkl'); Rr['chain'] = Rr.chain.astype(str)
Rr['clash'] = Rr.nclash.fillna(0) > 0
Rr['allw'] = Rr.rama == 'Allowed'
cm = Rr.groupby(['pdb', 'chain']).agg(n=('idx', 'size'), clash=('clash', 'mean'), allw=('allw', 'mean'), q=('q', 'mean')).reset_index()
cm[['clash', 'allw']] *= 100
ch = pd.read_csv(f'{SP}/deeper/chains.csv.gz', dtype={'chain': str}, usecols=['pdb', 'chain', 'ctype', 'vhh', 'err_any', 'err_first', 'seq'])
cm = cm.merge(ch, on=['pdb', 'chain'], how='left')
cm['cls'] = np.where(cm.vhh == 1, 'VHH', np.where(cm.ctype == 'H', 'H', 'VL'))
wrows = []
for _, u in PU.iterrows():
    pc = u.partner
    me = cm[(cm.pdb == u.pdb) & (cm.chain == pc)]
    if not len(me):
        continue
    me = me.iloc[0]
    oth = cm[(cm.pdb == u.pdb) & (cm.cls == me.cls) & (cm.chain != pc) & (cm.err_any == False)]
    oth_same = oth[oth.seq == me.seq]
    if not len(oth):
        continue
    wrows.append(dict(case=u.case, pdb=u.pdb, paper=u.paper, partner=pc, n_other=len(oth), n_other_sameseq=len(oth_same),
                      d_clash=me.clash - oth.clash.mean(), d_allw=me.allw - oth.allw.mean(), d_q=me.q - oth.q.mean(),
                      d_clash_sameseq=(me.clash - oth_same.clash.mean()) if len(oth_same) else np.nan,
                      d_allw_sameseq=(me.allw - oth_same.allw.mean()) if len(oth_same) else np.nan))
W = pd.DataFrame(wrows)
W.to_csv(f'{OUT}/within_entry.csv', index=False)
wsum = []
for c, x in W.groupby('case'):
    x1 = x.drop_duplicates(['pdb', 'partner'])
    for col in ['d_clash', 'd_allw', 'd_q', 'd_clash_sameseq', 'd_allw_sameseq']:
        v = x1[col].dropna()
        if len(v) >= 5:
            wsum.append(dict(group='case' if c else 'control', metric=col, n=len(v), n_entries=x1.loc[v.index].pdb.nunique(),
                             mean=v.mean(), median=v.median(), p_wilcoxon=stats.wilcoxon(v).pvalue))
WS = pd.DataFrame(wsum)
# difference-in-differences: case vs control within-entry gaps
for col in ['d_clash', 'd_allw']:
    x = W.dropna(subset=[col])
    m = smf.ols(f'{col} ~ case', x).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(x.paper)[0]})
    wsum.append(dict(group='case minus control (DiD)', metric=col, n=len(x), mean=m.params['case'], p_wilcoxon=m.pvalues['case']))
WS = pd.DataFrame(wsum); WS.to_csv(f'{OUT}/within_entry_summary.csv', index=False)
print(WS.round(4).to_string())

# ---------------- (5) core RMSD detail (partner of an error chain vs error chain vs controls)
F = pd.read_csv(f'{INV}/fv_orientation.csv', dtype={'H': str, 'L': str})
F = F.merge(s[['unit', 'pdb', 'paper']].drop_duplicates(['unit', 'pdb']), on=['unit', 'pdb'], how='left')
crow = []
for _, f in F.iterrows():
    for side, err, anyf, rms in [('H', f.H_err, f.H_any, f.rmsH_sabr), ('L', f.L_err, f.L_any, f.rmsL_sabr)]:
        other_err = f.L_err if side == 'H' else f.H_err
        if f.role_cc == 'control':
            grp = 'control'
        elif err:
            grp = 'error chain'
        elif not anyf and other_err:
            grp = 'error-free partner'
        else:
            continue
        crow.append(dict(pdb=f.pdb, paper=f.paper, side=side, grp=grp, rms=rms, res=f.res, ybin=f.ybin))
CR = pd.DataFrame(crow).drop_duplicates(['pdb', 'side', 'rms'])
CR.to_csv(f'{OUT}/core_rmsd.csv', index=False)
print(CR.groupby(['grp', 'side']).rms.agg(['size', 'mean', 'median']).round(3))
for grp in ['error-free partner', 'error chain']:
    x = CR[CR.grp.isin([grp, 'control'])].copy(); x['case'] = (x.grp == grp).astype(int)
    m = smf.ols('rms ~ case + C(side) + C(ybin) + res', x).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(x.paper)[0]})
    print(grp, 'vs control core RMSD: effect', round(m.params['case'], 3), 'CI', m.conf_int().loc['case'].round(3).tolist(),
          'p', m.pvalues['case'], 'MW p', stats.mannwhitneyu(x[x.case == 1].rms, x[x.case == 0].rms).pvalue)
