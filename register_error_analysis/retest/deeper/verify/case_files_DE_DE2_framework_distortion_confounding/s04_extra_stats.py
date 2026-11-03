"""DE2 extra tests: sensitivity exclusions, lab clustering, partner chains in the same entry, same antibody in other
models, other error modes, and whether the independent 'confirmation' test is sensitive to framework distortion."""
import warnings, numpy as np, pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import mannwhitneyu, wilcoxon, spearmanr, fisher_exact
from statsmodels.stats.multitest import multipletests
from s00_reproduce import load_sample, sets, D, OUT, MODES
warnings.filterwarnings('ignore')

X = load_sample()
F = pd.read_csv(f'{OUT}/features_recomputed.csv')
CH = pd.read_csv(f'{D}/chains.csv.gz')
R = pd.read_csv(f'{D}/runs_first_occurrence.csv'); RA = pd.read_csv(f'{D}/runs_all.csv')
X = X.merge(F, on=['pdb', 'chain'], how='left').merge(CH[['pdb', 'chain', 'last_author', 'sars2', 'err_any', 'err_first']],
                                                       on=['pdb', 'chain'], how='left')
e, mc, wp = sets(X)
rows = []


def fit(d, var='fw_orig', groups='pp', label='', extra=''):
    d = d.dropna(subset=[var]).copy(); d['x'] = (d[var] - d[var].mean()) / d[var].std()
    f = 'err ~ x + C(yb) + res + C(ct)' + extra
    m = smf.logit(f, data=d).fit(disp=0, cov_type='cluster', cov_kwds={'groups': pd.factorize(d[groups].fillna('na'))[0]})
    return dict(test=label, n_err=int(d.err.sum()), n_ctl=int((d.err == 0).sum()), err_mean=d[d.err == 1][var].mean(),
                ctl_mean=d[d.err == 0][var].mean(), OR_perSD=float(np.exp(m.params['x'])),
                lo=float(np.exp(m.conf_int().loc['x', 0])), hi=float(np.exp(m.conf_int().loc['x', 1])), p=float(m.pvalues['x']),
                mw_p=float(mannwhitneyu(d[d.err == 1][var], d[d.err == 0][var]).pvalue))


rows.append(fit(pd.concat([e, mc]), label='base, matched (cluster paper)'))
rows.append(fit(pd.concat([e, mc]), groups='last_author', label='base, matched (cluster last author / lab)'))
e2 = e[e.n_other_runs == 0]
rows.append(fit(pd.concat([e2, mc]), label='excl. 13 error chains that also carry a non-D/E run'))
# partner chain in same entry has a first-occurrence error
errchains = R.groupby('pdb').chain.apply(set).to_dict()
e['partner_err'] = [len(errchains.get(p, set()) - {c}) > 0 for p, c in zip(e.pdb, e.chain)]
rows.append(fit(pd.concat([e[~e.partner_err], mc]), label='excl. error chains whose entry has another first-occurrence error chain'))
e3 = e[(~e.partner_err) & (e.n_other_runs == 0)]
rows.append(fit(pd.concat([e3, mc]), label='excl. both (single isolated D/E error, no other runs in chain or entry)'))
rows.append(fit(pd.concat([e3, mc]), var='fw_core', label='fw_core, single isolated D/E error'))
# top-paper influence: drop the 5 papers with most error units / most extreme
top = e.groupby('pp').fw_orig.max().sort_values(ascending=False).index[:5]
rows.append(fit(pd.concat([e[~e.pp.isin(top)], mc]), label='excl. 5 papers with the most distorted error chain'))
rows.append(fit(pd.concat([e[e.fw_orig < 2.5], mc[mc.fw_orig < 2.5]]), label='truncate fw_orig>=2.5 A (outliers removed)'))

# ---------- partner chains ----------
G = pd.read_csv(f'{OUT}/features_extra.csv')
S = pd.read_csv(f'{OUT}/extra_sample.csv')
P = S[S.set == 'partner'].merge(G, on=['pdb', 'chain'])
ra = set(zip(RA.pdb, RA.chain))
P['has_run'] = [k in ra for k in zip(P.pdb, P.chain)]
P['ct'] = np.where(P.vhh == 1, 'VHH', P.ctype)
P['yb'] = pd.cut(P.year, [0, 2019, 2020, 2021, 2023, 2030], labels=['<=2019', '2020', '2021', '2022-23', '>=2024']).astype(str)
P['pp'] = P.paper
Pc = P[~P.has_run].sort_values(['pdb', 'chain']).drop_duplicates('unit')  # clean partner chains, one per unit
Pc = Pc[Pc.pdb.isin(e.pdb)]
d = pd.concat([Pc.assign(err=1), mc])
r = fit(d, label='CLEAN partner V chains of D/E error entries (as "cases") vs matched clean'); r['n_err'] = len(Pc); rows.append(r)
# paired: error chain vs clean partner in the same entry (different chain type mostly)
pe = e[['pdb', 'chain', 'fw_orig', 'fw_seqref', 'ct']].merge(Pc[['pdb', 'chain', 'fw_orig', 'fw_seqref', 'ct']], on='pdb', suffixes=('_err', '_partner'))
pe = pe.groupby('pdb').agg(err=('fw_orig_err', 'mean'), partner=('fw_orig_partner', 'mean'),
                           err_s=('fw_seqref_err', 'mean'), partner_s=('fw_seqref_partner', 'mean')).reset_index()
pe.to_csv(f'{OUT}/paired_error_vs_clean_partner.csv', index=False)
rows.append(dict(test='paired, same entry: error chain vs clean partner chain (fw_orig)', n_err=len(pe), n_ctl=len(pe),
                 err_mean=pe.err.mean(), ctl_mean=pe.partner.mean(), p=float(wilcoxon(pe.err - pe.partner).pvalue),
                 note=f'error higher in {(pe.err > pe.partner).sum()}/{len(pe)}'))
rows.append(dict(test='paired, same entry: error chain vs clean partner chain (fw_seqref)', n_err=len(pe), n_ctl=len(pe),
                 err_mean=pe.err_s.mean(), ctl_mean=pe.partner_s.mean(), p=float(wilcoxon(pe.err_s - pe.partner_s).pvalue),
                 note=f'error higher in {(pe.err_s > pe.partner_s).sum()}/{len(pe)}'))
# matched controls: within-entry H vs L difference distribution, for scale
# ---------- same antibody elsewhere ----------
Sv = S[S.set == 'same_vseq_clean'].merge(G, on=['pdb', 'chain'])
ev = e.merge(CH[['pdb', 'chain', 'vseq']], on=['pdb', 'chain'])
sv = ev[['pdb', 'chain', 'vseq', 'fw_orig']].merge(Sv[['pdb', 'chain', 'vseq', 'fw_orig', 'method']], on='vseq', suffixes=('_err', '_same'))
sv.to_csv(f'{OUT}/same_antibody_other_models.csv', index=False)
g = sv.groupby(['pdb_err', 'chain_err']).agg(err=('fw_orig_err', 'first'), same=('fw_orig_same', 'median'), n=('pdb_same', 'size')).reset_index()
print(g)
rows.append(dict(test='same antibody (identical V seq), clean model elsewhere: error vs median clean copy', n_err=len(g),
                 n_ctl=int(g.n.sum()), err_mean=g.err.mean(), ctl_mean=g.same.mean(),
                 p=float(wilcoxon(g.err - g.same).pvalue) if len(g) >= 5 else np.nan,
                 note=f'error higher in {(g.err > g.same).sum()}/{len(g)}'))
# ---------- other error modes ----------
O = S[S.set == 'other_mode_error'].merge(G, on=['pdb', 'chain'])
O['ct'] = np.where(O.vhh == 1, 'VHH', O.ctype)
O['yb'] = pd.cut(O.year, [0, 2019, 2020, 2021, 2023, 2030], labels=['<=2019', '2020', '2021', '2022-23', '>=2024']).astype(str)
O['pp'] = O.paper
for md in ['all other modes', 'A_FR1Nterm', 'B_CDR2FR3', 'C_FR3', 'F_CDR1', 'other']:
    o = O if md == 'all other modes' else O[O['mode'] == md]
    if len(o) < 8: continue
    rows.append(fit(pd.concat([o.assign(err=1), mc]), label=f'other first-occurrence error mode: {md} vs matched'))
    rows.append(fit(pd.concat([o.assign(err=1), mc]), var='fw_core', label=f'other mode {md}, fw_core'))
# D/E vs other modes directly
d = pd.concat([e.assign(err=1), O.assign(err=0)])
r = fit(d, label='D/E errors vs other-mode errors (is distortion specific to D/E?)'); rows.append(r)

# ---------- is the independent confirmation sensitive to framework distortion? ----------
XS = pd.read_pickle('/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad/deeper/cache/case_files_DE/X_struct.pkl')
cl = XS[XS.group != 'error'].drop_duplicates(['pdb', 'chain'])
cl['offslot'] = cl[[f'slotshift_{n}' for n in range(118, 125)]].abs().gt(0).any(axis=1)
hi = cl.fw_rmsd > 1.4
t = [[int((cl.offslot & hi).sum()), int((~cl.offslot & hi).sum())], [int((cl.offslot & ~hi).sum()), int((~cl.offslot & ~hi).sum())]]
orr, pp_ = fisher_exact(t)
rows.append(dict(test='clean controls: FR4 off-slot (confirmation false positive) in fw_rmsd>1.4 vs <=1.4', n_err=int(hi.sum()),
                 n_ctl=int((~hi).sum()), err_mean=t[0][0] / max(1, hi.sum()), ctl_mean=t[1][0] / max(1, (~hi).sum()),
                 OR_perSD=orr, p=pp_, note='rates in err_mean/ctl_mean columns; OR is Fisher'))

T = pd.DataFrame(rows)
ok = T.p.notna()
T.loc[ok, 'q_within_file'] = multipletests(T.loc[ok, 'p'], method='fdr_bh')[1]
T.to_csv(f'{OUT}/tests_extra.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 90)
print(T.round(4).to_string())
