"""Cross-case tests of the specifics seen in the case files (coordinate features from s02), cryo-EM only, one chain per unit.

Errors : D+/D-/E first-occurrence units (EM). Main set = runs independently confirmed by framework superposition
         (or not testable there); sensitivity = all.
Controls: (1) matched clean EM units (same deposition year x chain type x nanobody), (2) clean units from the same papers.
Models : logit(error ~ x + year bin + resolution + chain type), cluster-robust SE by paper; and conditional logit
         within paper (errors vs clean units of the same paper). BH q over all tests in this script.
"""
import sys, warnings
sys.path.insert(0, '/home/user/SAbR/register_error_analysis/retest/deeper/case_files_DE/scripts')
from common import *
import statsmodels.formula.api as smf
from statsmodels.discrete.conditional_models import ConditionalLogit
from statsmodels.stats.multitest import multipletests
from scipy.stats import fisher_exact, mannwhitneyu
warnings.filterwarnings('ignore')

X = pd.read_pickle(f'{CACHE}/X_struct.pkl')
ER = pd.read_pickle(f'{CACHE}/err_struct.pkl')[['pdb', 'chain', 'confirm']]
X = X.merge(ER, on=['pdb', 'chain'], how='left')
CH = chains()
X = X.merge(CH[['pdb', 'chain', 'unit', 'paper', 'deposit']].rename(columns={'unit': 'u', 'paper': 'pp'}), on=['pdb', 'chain'], how='left')
X = X[X.method == 'EM'].copy()
X['err'] = (X.group == 'error').astype(int)
X['yb'] = pd.cut(X.year, [0, 2019, 2020, 2021, 2023, 2030], labels=['<=2019', '2020', '2021', '2022-23', '>=2024']).astype(str)
X['ct'] = np.where(X.vhh == 1, 'VHH', X.ctype)
X['cdr3_ag_any'] = (X.cdr3_ag_contacts > 0).astype(float)
X['cdr3_ag_frac'] = X.cdr3_ag_contacts / X.cdr3_built
X['v_ag_any'] = (X.v_ag_contacts > 0).astype(float)
X['partner_contact_any'] = (X.cdr3_partner_contacts > 0).astype(float)
X['fr4_trunc'] = (X.last_built_imgt < 125).astype(float)
X['fr4_unbuilt'] = (X.fr4_unbuilt_after118_10 > 0).astype(float)
X['seqres_trunc'] = (X.fr4_seqres_after118 < 8).astype(float)
X['sc_trunc_any'] = (X.sc_trunc_site > 0).astype(float)
X['fw_rmsd_gt1'] = (X.fw_rmsd > 1.2).astype(float)
X['w118_exposed'] = (X.w118_ca10 < 14).astype(float)
feats = ['cdr3_ag_any', 'cdr3_ag_frac', 'v_ag_any', 'partner_contact_any', 'fr4_trunc', 'fr4_unbuilt', 'seqres_trunc', 'sc_trunc_any',
         'b_site_rel', 'fw_rmsd', 'w118_exposed']
# one chain per unit
X = X.sort_values(['group', 'pdb', 'chain']).drop_duplicates(['group', 'u'])
out = []
modes_all = {'all D/E': MODES, 'D+': ['Dplus_CDR3short'], 'D-': ['Dminus_CDR3long'], 'E': ['E_FR4bulge']}
for errset in ['confirmed', 'all']:
    for mname, ms in modes_all.items():
        e = X[(X.group == 'error') & X['mode'].isin(ms)]
        if errset == 'confirmed': e = e[e.confirm != 'not confirmed']
        for cg in ['matched_clean', 'within_paper_clean']:
            c = X[X.group == cg]
            if cg == 'within_paper_clean': c = c[c.pp.isin(e.pp)]
            d = pd.concat([e, c])
            for f in feats:
                dd = d.dropna(subset=[f])
                if dd[f].nunique() < 2 or dd.err.sum() < 5: continue
                ev, cv = dd[dd.err == 1][f].mean(), dd[dd.err == 0][f].mean()
                row = dict(error_set=errset, mode=mname, control=cg, feature=f, n_err=int(dd.err.sum()), n_ctl=int((dd.err == 0).sum()),
                           err_mean=ev, ctl_mean=cv)
                try:
                    if cg == 'matched_clean':
                        z = dd.copy(); z['x'] = (z[f] - z[f].mean()) / (z[f].std() if z[f].nunique() > 2 else 1)
                        m = smf.logit('err ~ x + C(yb) + res + C(ct)', data=z).fit(disp=0, cov_type='cluster', cov_kwds={'groups': pd.factorize(z.pp)[0]})
                        row.update(OR=float(np.exp(m.params['x'])), p=float(m.pvalues['x']), adjusted='year bin, resolution, chain type; cluster by paper')
                    else:
                        z = dd[dd.pp.isin(dd[dd.err == 1].pp) & dd.pp.isin(dd[dd.err == 0].pp)].copy()
                        z['x'] = (z[f] - z[f].mean()) / (z[f].std() if z[f].nunique() > 2 else 1)
                        m = ConditionalLogit(z.err.values, z[['x']].values, groups=pd.factorize(z.pp)[0]).fit(disp=0)
                        row.update(OR=float(np.exp(m.params[0])), p=float(m.pvalues[0]), n_papers=z.pp.nunique(), adjusted='within paper (conditional logit)')
                except Exception as ex:
                    row.update(OR=np.nan, p=np.nan, adjusted=f'fit failed: {str(ex)[:40]}')
                row['scale'] = 'per SD' if dd[f].nunique() > 2 else 'binary'
                if row['scale'] == 'binary':
                    a = int(dd[(dd.err == 1)][f].sum()); b = int((dd.err == 1).sum()) - a
                    c_ = int(dd[(dd.err == 0)][f].sum()); d_ = int((dd.err == 0).sum()) - c_
                    row['fisher_OR'], row['fisher_p'] = fisher_exact([[a, b], [c_, d_]])
                    row['n_err_with'] = a; row['n_ctl_with'] = c_
                    # model OR unusable under (quasi-)separation: fall back to Fisher
                    if not np.isfinite(row.get('OR', np.nan)) or row.get('OR', 1) > 1e3 or row.get('OR', 1) < 1e-3 or min(a, b, c_, d_) == 0:
                        row['OR'], row['p'], row['adjusted'] = row['fisher_OR'], row['fisher_p'], 'unadjusted Fisher (separation)'
                out.append(row)
T = pd.DataFrame(out)
ok = T.p.notna()
T.loc[ok, 'q'] = multipletests(T.loc[ok, 'p'], method='fdr_bh')[1]
T.to_csv(f'{OUT}/tables/struct_tests.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 400)
print(T[(T.error_set == 'confirmed')].round(3).to_string())
