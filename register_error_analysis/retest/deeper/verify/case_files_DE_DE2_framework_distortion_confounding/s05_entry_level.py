"""Is the framework distortion a property of the error CHAIN or of the whole ENTRY/Fab model?
(a) refined 'isolated D/E error' definition (no non-D/E run in the chain; no other first-occurrence error chain with a
    different V sequence in the entry; symmetry copies of the same antibody are not counted as partners);
(b) entry fixed-effects regression among D/E error entries: error chain vs clean V chains of the same entry, adjusted
    for chain type;
(c) chain-type offsets in the matched controls (so the paired H-vs-L comparison is interpretable)."""
import warnings, numpy as np, pandas as pd
import statsmodels.formula.api as smf
from statsmodels.discrete.conditional_models import ConditionalLogit
from scipy.stats import mannwhitneyu
from s00_reproduce import load_sample, sets, D, OUT, MODES
warnings.filterwarnings('ignore')

X = load_sample()
F = pd.read_csv(f'{OUT}/features_recomputed.csv')
CH = pd.read_csv(f'{D}/chains.csv.gz')
R = pd.read_csv(f'{D}/runs_first_occurrence.csv'); RA = pd.read_csv(f'{D}/runs_all.csv')
X = X.merge(F, on=['pdb', 'chain'], how='left').merge(CH[['pdb', 'chain', 'vseq']], on=['pdb', 'chain'], how='left')
e, mc, wp = sets(X)
vs = dict(zip(zip(CH.pdb, CH.chain), CH.vseq))
errc = R[['pdb', 'chain']].drop_duplicates()
errc['vseq'] = [vs.get(k) for k in zip(errc.pdb, errc.chain)]
other_err_seq = {}
for p, g in errc.groupby('pdb'): other_err_seq[p] = set(g.vseq)
e = e.copy()
e['partner_err'] = [len(other_err_seq.get(p, set()) - {v}) > 0 for p, v in zip(e.pdb, e.vseq)]
e['isolated'] = (~e.partner_err) & (e.n_other_runs == 0)
print('error units', len(e), 'partner (different V seq) has first-occ error', e.partner_err.sum(),
      'non-D/E run in same chain', (e.n_other_runs > 0).sum(), 'isolated', e.isolated.sum())
rows = []


def fit(d, var='fw_orig', label=''):
    d = d.dropna(subset=[var]).copy(); d['x'] = (d[var] - d[var].mean()) / d[var].std()
    m = smf.logit('err ~ x + C(yb) + res + C(ct)', data=d).fit(disp=0, cov_type='cluster', cov_kwds={'groups': pd.factorize(d.pp)[0]})
    return dict(test=label, metric=var, n_err=int(d.err.sum()), n_ctl=int((d.err == 0).sum()), err_mean=d[d.err == 1][var].mean(),
                ctl_mean=d[d.err == 0][var].mean(), err_median=d[d.err == 1][var].median(), ctl_median=d[d.err == 0][var].median(),
                OR_perSD=float(np.exp(m.params['x'])), lo=float(np.exp(m.conf_int().loc['x', 0])),
                hi=float(np.exp(m.conf_int().loc['x', 1])), p=float(m.pvalues['x']),
                mw_p=float(mannwhitneyu(d[d.err == 1][var], d[d.err == 0][var]).pvalue))


for var in ['fw_orig', 'fw_seqref', 'fw_core']:
    for lab, sub in [('all confirmed', e), ('partner chain (different V seq) also has a first-occ error', e[e.partner_err]),
                     ('no partner error', e[~e.partner_err]), ('non-D/E run in same chain', e[e.n_other_runs > 0]),
                     ('isolated D/E (no other run in chain, no partner error)', e[e.isolated]),
                     ('not isolated', e[~e.isolated])]:
        if len(sub) >= 5: rows.append(fit(pd.concat([sub, mc]), var, f'{lab} vs matched'))
# within paper for isolated errors
d = pd.concat([e[e.isolated], wp]).dropna(subset=['fw_orig'])
d = d[d.pp.isin(d[d.err == 1].pp) & d.pp.isin(d[d.err == 0].pp)].copy(); d['x'] = (d.fw_orig - d.fw_orig.mean()) / d.fw_orig.std()
try:
    m = ConditionalLogit(d.err.values, d[['x']].values, groups=pd.factorize(d.pp)[0]).fit(disp=0)
    rows.append(dict(test='isolated D/E vs clean units same paper (conditional logit)', metric='fw_orig', n_err=int(d.err.sum()),
                     n_ctl=int((d.err == 0).sum()), err_mean=d[d.err == 1].fw_orig.mean(), ctl_mean=d[d.err == 0].fw_orig.mean(),
                     OR_perSD=float(np.exp(m.params[0])), p=float(m.pvalues[0]), note=f'{d.pp.nunique()} papers'))
except Exception as ex: print(ex)

# (b) entry fixed effects among error entries: error chains vs clean V chains of the same entries
G = pd.read_csv(f'{OUT}/features_extra.csv'); S = pd.read_csv(f'{OUT}/extra_sample.csv')
P = S[S.set == 'partner'].merge(G, on=['pdb', 'chain'])
ra = set(zip(RA.pdb, RA.chain))
P = P[[k not in ra for k in zip(P.pdb, P.chain)]]
P['ct'] = np.where(P.vhh == 1, 'VHH', P.ctype)
EE = pd.concat([e[['pdb', 'chain', 'ct', 'fw_orig', 'fw_seqref', 'fw_core', 'q_fw']].assign(errchain=1),
                P[['pdb', 'chain', 'ct', 'fw_orig', 'fw_seqref', 'fw_core', 'q_fw']].assign(errchain=0)])
EE = EE[EE.pdb.isin(P.pdb) & EE.pdb.isin(e.pdb)]
# also add the other (copy) chains of the error antibody? no: one row per chain as computed
for var in ['fw_orig', 'fw_seqref', 'fw_core']:
    EV = EE.dropna(subset=[var])
    m = smf.ols(f'{var} ~ errchain + C(ct) + C(pdb)', data=EV).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(EV.pdb)[0]})
    rows.append(dict(test='entry fixed effects: error chain vs clean V chain of the SAME entry (+chain type)', metric=var,
                     n_err=int(EE.errchain.sum()), n_ctl=int((EE.errchain == 0).sum()), diff_A=float(m.params['errchain']),
                     lo=float(m.conf_int().loc['errchain', 0]), hi=float(m.conf_int().loc['errchain', 1]), p=float(m.pvalues['errchain']),
                     note=f'{EE.pdb.nunique()} entries; diff_A in Angstrom'))
    m = smf.ols(f'{var} ~ errchain + C(ct) + q_fw + C(pdb)', data=EV.dropna(subset=['q_fw'])).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(EV.dropna(subset=['q_fw']).pdb)[0]})
    rows.append(dict(test='entry fixed effects + local Q-score', metric=var, n_err=int(EE.errchain.sum()),
                     n_ctl=int((EE.errchain == 0).sum()), diff_A=float(m.params['errchain']), p=float(m.pvalues['errchain'])))
# same model on matched + error + clean partners: how much of the error-vs-matched excess is shared by clean partners
Z = pd.concat([e.assign(grp='error chain'), P[P.pdb.isin(e.pdb)].assign(grp='clean partner in error entry', pp=P.paper,
              yb=pd.cut(P.year, [0, 2019, 2020, 2021, 2023, 2030], labels=['<=2019', '2020', '2021', '2022-23', '>=2024']).astype(str)),
               mc.assign(grp='matched clean')])
for var in ['fw_orig', 'fw_seqref']:
    ZV = Z.dropna(subset=[var])
    m = smf.ols(f"{var} ~ C(grp, Treatment('matched clean')) + C(yb) + res + C(ct)", data=ZV).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(ZV.pp)[0]})
    for k in m.params.index:
        if k.startswith('C(grp'):
            rows.append(dict(test=f'3-group model vs matched clean: {k.split("[T.")[1][:-1]}', metric=var, diff_A=float(m.params[k]),
                             lo=float(m.conf_int().loc[k, 0]), hi=float(m.conf_int().loc[k, 1]), p=float(m.pvalues[k]),
                             n_err=int((Z.grp == k.split('[T.')[1][:-1]).sum()), n_ctl=int((Z.grp == 'matched clean').sum()),
                             note='adjusted year bin, res, chain type; cluster paper'))
print(mc.groupby('ct').fw_orig.agg(['mean', 'median', 'size']))
T = pd.DataFrame(rows)
T.to_csv(f'{OUT}/tests_entry_level.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 80)
print(T.round(4).to_string())
