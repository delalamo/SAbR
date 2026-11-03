"""Backbone-geometry fingerprint from wwPDB validation XML (per-residue Ramachandran class):
  T3  beyond the error site: % Ramachandran 'allowed' among V residues >3 residues from ANY SAbR/ANARCI disagreeing
      residue (tallies bins d4_10 + d11p + clean; the stricter 'Vex3' definition of the verifiers, which also removes the
      run +-3). Chain level, EM and X-ray.
  T4  antibody vs rest of the same model: entry-level gap = % allowed in all V residues minus % allowed in non-V residues
      (constant domains + other chains), as in verify/struct_unusual_antibody_vs_rest_of_model_confounding (gap_V_A).
      Also the gap with disagreement-free V chains only (gap_Vclean_A).
  T7a within an entry: error chain vs error-free (err_any==0) V chains of the same entry (beyond-site %allowed).
Tallies: EM from deeper/cache/verify_su_gap/tallies.pkl (all EM entries, built by that verifier's v02_parse.py);
X-ray from scratchpad/paperfp/tallies_xray.pkl (s03_parse_vrpt_xray.py, same code).
Output: tables/rama_chain.csv.gz, tables/rama_entry.csv.gz, tables/rama_tests.csv, tables/rama_within_entry.csv"""
import sys, numpy as np, pandas as pd, statsmodels.formula.api as smf
from scipy import stats
sys.path.insert(0, '/home/user/SAbR/register_error_analysis/retest/paper_confirmed/model_fingerprint/scripts')
from common import SP, TAB, load_frame, ybin_win, het_z, wald_equal
import binsuite
from binsuite import suite

D = load_frame()
D = D[~D.amyloid].copy()
T = pd.concat([pd.read_pickle(f'{SP}/deeper/cache/verify_su_gap/tallies.pkl'), pd.read_pickle(f'{SP}/paperfp/tallies_xray.pkl')])
T['chain'] = T.chain.astype(str)
V = T[T.cat == 'V']
# ---- chain level
def chain_agg(m, tag):
    g = V[m].groupby(['pdb', 'chain'])[['n_rama', 'allowed', 'rama_out', 'clash', 'n']].sum()
    return pd.DataFrame({f'{tag}_n': g.n_rama, f'{tag}_A': 100 * g.allowed / g.n_rama.replace(0, np.nan),
                         f'{tag}_Cl': 100 * g.clash / g.n.replace(0, np.nan)})
CH = chain_agg(V.dbin.isin(['d4_10', 'd11p', 'clean']), 'out').join(chain_agg(V.dbin.notna(), 'V'), how='outer').reset_index()
CH = CH.merge(D, on=['pdb', 'chain'], how='inner')
CH.loc[CH.out_n < 30, 'out_A'] = np.nan
CH['cclass'] = np.where(CH.vhh == 1, 'VHH', CH.ctype)
CH.to_csv(f'{TAB}/rama_chain.csv.gz', index=False)
print('chains with beyond-site Rama:', CH.out_A.notna().sum(), CH.groupby(['grp', 'method']).out_A.apply(lambda s: s.notna().sum()).to_dict())

res = []
binsuite.DIRECTION = 1
# suite() uses covariates year bin + resolution (+method); add chain class by residualising? -> run with cclass via a wrapper
_orig = binsuite._covs
binsuite._covs = lambda d, years='ybin_w': _orig(d, years) + ['C(cclass)']
res += suite(CH, 'out_A', 'cont', 'T3 beyond-site % Ramachandran allowed (V residues >3 from any disagreement)', CH.out_A.notna())
res += suite(CH, 'out_Cl', 'cont', 'T3b beyond-site clash residues per 100 (same residues)', CH.out_A.notna())
binsuite._covs = _orig

# ---- entry level (T4)
def ent_agg(m, tag):
    g = T[m].groupby('pdb')[['n_rama', 'allowed', 'n', 'clash']].sum()
    return pd.DataFrame({f'{tag}_n': g.n_rama, f'{tag}_A': 100 * g.allowed / g.n_rama.replace(0, np.nan)})
E = ent_agg(T.cat == 'V', 'V').join(ent_agg((T.cat == 'V') & (T.dbin == 'clean'), 'Vclean')).join(
    ent_agg(T.cat.isin(['C', 'O']), 'rest')).join(ent_agg(T.cat == 'O', 'O'))
E.loc[E.rest_n < 30, ['rest_A']] = np.nan
E['gap_V_A'] = E.V_A - E.rest_A; E['gap_Vclean_A'] = E.Vclean_A - E.rest_A
E.loc[E.Vclean_n < 30, 'gap_Vclean_A'] = np.nan
grp = D.groupby('pdb').agg(method=('method', 'first'), res=('res', 'first'), year=('year', 'first'), cluster=('cluster', 'first'),
                           n_strict=('grp', lambda s: (s == 'paper_strict').sum()),
                           n_incl=('grp', lambda s: s.isin(['paper_strict', 'paper_loose', 'paper_copy']).sum()),
                           n_rest=('grp', lambda s: (s == 'rest_case').sum()),
                           all_control=('grp', lambda s: (s == 'control').all()),
                           all_clean=('err_any', lambda s: (~s.astype(bool)).all()),
                           flagged=('paper_flagged_entry', 'max'), window=('in_paper_window', 'max'), unit=('unit', 'first'))
E = E.join(grp, how='inner').reset_index().rename(columns={'index': 'pdb'})
E['grp'] = np.select([E.n_strict > 0, E.n_incl > 0, E.n_rest > 0, E.all_control, E.all_clean],
                     ['paper_strict', 'paper_copy', 'rest_case', 'control', 'clean_other'], 'other')
E['ybin_w'] = ybin_win(E.year.values)
from common import ybin
E['ybin'] = ybin(E.year.values)
E.to_csv(f'{TAB}/rama_entry.csv.gz', index=False)
print(E.groupby(['grp', 'method']).gap_V_A.apply(lambda s: f'{s.notna().sum()} mean {s.mean():.2f}').to_string())
binsuite.DIRECTION = 1
res += suite(E, 'gap_V_A', 'cont', 'T4 entry: V minus non-V % Ramachandran allowed (pp)', E.gap_V_A.notna())
res += suite(E, 'gap_Vclean_A', 'cont', 'T4b entry: disagreement-free V chains minus non-V % allowed (pp)', E.gap_Vclean_A.notna())
res += suite(E, 'rest_A', 'cont', 'T4c entry: non-V (antigen, constant) % allowed', E.rest_A.notna())

R = pd.DataFrame(res)
R.to_csv(f'{TAB}/rama_tests.csv', index=False)

# ---- full-set reproduction on this metric (EM, all years, first-occurrence vs clean): chain and entry
x = CH[(CH.method == 'EM') & CH.out_A.notna() & (CH.err_first | ~CH.err_any)].copy(); x['case'] = x.err_first.astype(int)
from common import ybin as yb
x['yb'] = yb(x.year.values)
m = smf.ols('out_A ~ case + C(yb) + res + C(cclass)', x).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(x.cluster)[0]})
print(f'full-set reproduction, EM chains, all first-occurrence vs clean: {m.params["case"]:+.2f} ({m.conf_int().loc["case",0]:+.2f},{m.conf_int().loc["case",1]:+.2f}) p={m.pvalues["case"]:.2g} n={int(x.case.sum())}/{int((1-x.case).sum())}')

# ---- T7a within entry: case chains vs error-free V chains of the same entry
rows = []
for cs_lab, groups in [('strict', ['paper_strict']), ('incl_copies', ['paper_strict', 'paper_loose', 'paper_copy']), ('rest_case', ['rest_case'])]:
    for meth in ['EM', 'X-ray', 'all']:
        c = CH[CH.out_A.notna()] if meth == 'all' else CH[CH.out_A.notna() & (CH.method == meth)]
        ents = set(c[c.grp.isin(groups)].pdb) & set(c[~c.err_any].pdb)
        w = c[c.pdb.isin(ents) & (c.grp.isin(groups) | ~c.err_any)].copy(); w['case'] = w.grp.isin(groups).astype(int)
        if len(ents) < 3:
            rows.append(dict(case_set=cs_lab, method=meth, n_entries=len(ents))); continue
        mm = smf.ols('out_A ~ case + C(pdb) + C(cclass)', w).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(w.cluster)[0]})
        pe = w.groupby(['pdb', 'case']).out_A.mean().unstack()
        dif = pe[1] - pe[0]
        pp = w.groupby(['cluster', 'case']).out_A.mean().unstack().dropna()
        rows.append(dict(case_set=cs_lab, method=meth, n_entries=len(ents), n_papers=w[w.case == 1].cluster.nunique(),
                         n_case_chains=int(w.case.sum()), n_clean_chains=int((1 - w.case).sum()),
                         fe_est=mm.params['case'], fe_lo=mm.conf_int().loc['case', 0], fe_hi=mm.conf_int().loc['case', 1], fe_p=mm.pvalues['case'],
                         fe_se=mm.bse['case'], entries_case_higher=int((dif > 0).sum()), entry_median_diff=dif.median(),
                         entry_wilcoxon_p=stats.wilcoxon(dif).pvalue if len(dif) > 5 else np.nan,
                         paper_mean_diff=(pp[1] - pp[0]).mean(), paper_n=len(pp),
                         paper_wilcoxon_p=stats.wilcoxon(pp[1] - pp[0]).pvalue if len(pp) > 5 else np.nan))
    # clean V chains of case entries vs window controls
W = pd.DataFrame(rows)
for meth in ['EM', 'X-ray', 'all']:
    s_ = W[(W.case_set == 'strict') & (W.method == meth)].iloc[0]; r_ = W[(W.case_set == 'rest_case') & (W.method == meth)].iloc[0]
    if np.isfinite(s_.get('fe_se', np.nan)) and np.isfinite(r_.get('fe_se', np.nan)):
        W.loc[(W.case_set == 'strict') & (W.method == meth), 'het_p_vs_rest'] = het_z(s_.fe_est, s_.fe_se, r_.fe_est, r_.fe_se)
# neighbours (error-free V chains in strict case entries) vs window controls
nb = []
for meth in ['EM', 'X-ray', 'all']:
    c = CH[CH.out_A.notna()] if meth == 'all' else CH[CH.out_A.notna() & (CH.method == meth)]
    ents = set(c[c.grp == 'paper_strict'].pdb)
    w = c[(c.pdb.isin(ents) & ~c.err_any) | (c.grp == 'control')].copy(); w['case'] = w.pdb.isin(ents).astype(int)
    if w.case.sum() < 3: continue
    f = 'out_A ~ case + C(ybin_w) + res + C(cclass)' + (' + C(method)' if meth == 'all' else '')
    mm = smf.ols(f, w).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(w.cluster)[0]})
    nb.append(dict(case_set='neighbours of strict cases vs window controls', method=meth, n_case_chains=int(w.case.sum()),
                   n_clean_chains=int((1 - w.case).sum()), fe_est=mm.params['case'], fe_lo=mm.conf_int().loc['case', 0],
                   fe_hi=mm.conf_int().loc['case', 1], fe_p=mm.pvalues['case']))
W = pd.concat([W, pd.DataFrame(nb)])
W.to_csv(f'{TAB}/rama_within_entry.csv', index=False)
pd.set_option('display.width', 250)
print(W.to_string(index=False, float_format=lambda v: f'{v:.3g}'))
