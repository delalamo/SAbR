"""Verifier (confounding lens) for SN2_trunc_FR4_modesDE: re-tests with adjustment/stratification.
Reads tables/unit_features.csv.gz written by v01_features.py.
"""
import numpy as np, pandas as pd, warnings, statsmodels.formula.api as smf, statsmodels.api as sm
from scipy import stats
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings('ignore')
D_ = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/seq_novelty_SN2_trunc_FR4_modesDE_confounding/tables'
U = pd.read_csv(f'{D_}/unit_features.csv.gz', low_memory=False)
LOG = open(f'{D_}/v02_tests.log', 'w')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); LOG.write(s + '\n')
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 300); pd.set_option('display.max_colwidth', 60)

U['yb'] = pd.cut(U.dep_year, [0, 2018, 2019, 2020, 2021, 2022, 2023, 2100], labels=['<=2018', '2019', '2020', '2021', '2022', '2023', '>=2024']).astype(str)
U['per4'] = pd.cut(U.dep_year, [0, 2019, 2021, 2023, 2100], labels=['<=2019', '2020-21', '2022-23', '>=2024']).astype(str)
U['nonhm'] = (~U.q_species_all.isin(['human', 'mouse'])).astype(int)
U['vhh_i'] = U.vhh_u.fillna(0).astype(int); U['sars_i'] = U.sars2_u.astype(int)
U['modes_u'] = U.modes_u.fillna('')
U['trunc'] = U.seq_fr4_trunc.astype(int)
# corrected: a 'truncation' where the SEQRES continues >5 residues beyond the ANARCI V parse is an ANARCI parse stop, not a truncated SEQRES
U['spurious'] = (U.seq_fr4_trunc & (U.seq_tail_after_v > 5)).astype(int)
U['trunc_true'] = (U.seq_fr4_trunc & ~(U.seq_tail_after_v > 5)).astype(int)
# model-level C-terminal end inside FR4 (whatever the reason: SEQRES truncated or residues unbuilt)
U['model_end_fr4'] = (U.mod_last_anarci < 125).astype(float).where(U.mod_last_anarci.notna())
U['unbuilt_cterm'] = ((U.mod_last_anarci < 125) & (U.seq_last_imgt >= 125)).astype(float).where(U.mod_last_anarci.notna())
U['nt_trunc'] = U.seq_nt_trunc.astype(int)
U['paper_id'] = U.paper.fillna(U.pdb)
EM = U.method_u.eq('EM')

def case_mask(m):
    if m == 'ANY': return U.err_first_u
    if m == 'D_all': return U.err_first_u & U.modes_u.str.contains('Dplus|Dminus|D0')
    if m == 'Dplus': return U.err_first_u & U.modes_u.str.contains('Dplus')
    if m == 'E': return U.err_first_u & U.modes_u.str.contains('E_FR4bulge')
    if m == 'A': return U.err_first_u & U.modes_u.str.contains('A_FR1Nterm')
    if m == 'notDE': return U.err_first_u & ~U.modes_u.str.contains('Dplus|Dminus|D0|E_FR4bulge')
def sets(m, control='clean'):
    cases = EM & case_mask(m)
    if control == 'clean': ctrl = EM & U.clean
    else:  # every other EM unit without a first-occurrence error at this location (deepdive-style 'correct here')
        ctrl = EM & ~case_mask(m) & (U.err_first_u | U.clean)
    if m not in ('ANY', 'notDE'): ctrl = ctrl & U.ctype.isin(U[cases].ctype.unique())
    d = U[cases | ctrl].copy(); d['y'] = cases[cases | ctrl].astype(int)
    return d

def firth(y, X, maxit=200, tol=1e-8):
    """Firth-penalised logistic; returns beta, se, penalised loglik."""
    X = np.asarray(X, float); y = np.asarray(y, float); b = np.zeros(X.shape[1])
    def pll(b):
        p = 1 / (1 + np.exp(-X @ b)); W = p * (1 - p); I = X.T @ (X * W[:, None])
        sign, ld = np.linalg.slogdet(I)
        return np.sum(y * np.log(p + 1e-300) + (1 - y) * np.log(1 - p + 1e-300)) + 0.5 * ld
    for _ in range(maxit):
        p = 1 / (1 + np.exp(-X @ b)); W = p * (1 - p)
        I = X.T @ (X * W[:, None]); Ii = np.linalg.pinv(I)
        h = np.einsum('ij,jk,ik->i', X * np.sqrt(W)[:, None], Ii, X * np.sqrt(W)[:, None])
        U_ = X.T @ (y - p + h * (0.5 - p))
        step = Ii @ U_
        mx = np.max(np.abs(step))
        if mx > 5: step = step * 5 / mx
        b0 = pll(b); bn = b + step; k = 0
        while pll(bn) < b0 - 1e-10 and k < 30: step /= 2; bn = b + step; k += 1
        b = bn
        if np.max(np.abs(step)) < tol: break
    p = 1 / (1 + np.exp(-X @ b)); W = p * (1 - p); I = X.T @ (X * W[:, None]); se = np.sqrt(np.diag(np.linalg.pinv(I)))
    return b, se, pll(b)

def design(d, feat, covs):
    X = pd.get_dummies(d[covs + [feat]], columns=[c for c in covs if not pd.api.types.is_numeric_dtype(d[c])], drop_first=True, dtype=float)
    X = X.loc[:, X.std() > 0]
    X.insert(0, 'const', 1.0)
    return X

def firth_test(d, feat, covs):
    d = d.dropna(subset=[feat] + [c for c in covs])
    X = design(d, feat, covs)
    if feat not in X: return dict(OR=np.nan, lo=np.nan, hi=np.nan, p=np.nan)
    b, se, ll1 = firth(d.y, X)
    j = list(X.columns).index(feat)
    X0 = X.drop(columns=[feat]); _, _, ll0 = firth(d.y, X0)
    lr = 2 * (ll1 - ll0); p = stats.chi2.sf(max(lr, 0), 1)
    return dict(OR=np.exp(b[j]), lo=np.exp(b[j] - 1.96 * se[j]), hi=np.exp(b[j] + 1.96 * se[j]), p=p)

def exact_strat(d, feat, strat):
    """Exact conditional test of association stratified by `strat` (sum over strata of hypergeometric counts),
    two-sided by probability <= observed; plus Mantel-Haenszel OR."""
    d = d.dropna(subset=[feat])
    dists = []; obs = 0; tabs = []
    for _, g in d.groupby(strat):
        n1 = int(g.y.sum()); m = int(g[feat].sum()); N = len(g)
        if n1 == 0 or m == 0 or n1 == N or m == N: obs += int(((g.y == 1) & (g[feat] == 1)).sum()); continue
        lo, hi = max(0, n1 - (N - m)), min(n1, m)
        ks = np.arange(lo, hi + 1); pk = hypergeom.pmf(ks, N, m, n1)
        dists.append((ks, pk)); obs += int(((g.y == 1) & (g[feat] == 1)).sum())
        a = ((g.y == 1) & (g[feat] == 1)).sum(); b = ((g.y == 1) & (g[feat] == 0)).sum(); c = ((g.y == 0) & (g[feat] == 1)).sum(); e = ((g.y == 0) & (g[feat] == 0)).sum()
        tabs.append(np.array([[a, b], [c, e]], float))
    # convolution
    vals = {0: 1.0}; base = obs - sum(0 for _ in [])
    fixed = obs - 0
    tot = {0: 1.0}
    for ks, pk in dists:
        new = {}
        for v, pv in tot.items():
            for k, q in zip(ks, pk): new[v + k] = new.get(v + k, 0) + pv * q
        tot = new
    # observed contribution from informative strata only
    obs_inf = 0
    for _, g in d.groupby(strat):
        n1 = int(g.y.sum()); m = int(g[feat].sum()); N = len(g)
        if not (n1 == 0 or m == 0 or n1 == N or m == N): obs_inf += int(((g.y == 1) & (g[feat] == 1)).sum())
    po = tot.get(obs_inf, 0)
    p = sum(v for k, v in tot.items() if v <= po * (1 + 1e-7))
    exp_ = sum(k * v for k, v in tot.items())
    num = sum(t[0, 0] * t[1, 1] / t.sum() for t in tabs); den = sum(t[0, 1] * t[1, 0] / t.sum() for t in tabs)
    return dict(MH_OR=num / den if den > 0 else np.inf, p_exact_strat=p, obs=obs_inf, expected=exp_)

def cluster_logit(d, feat, covs):
    d = d.dropna(subset=[feat] + covs)
    f = 'y ~ ' + feat + ''.join(f' + C({c})' if not pd.api.types.is_numeric_dtype(d[c]) else f' + {c}' for c in covs if d[c].nunique() > 1)
    try:
        fit = smf.logit(f, data=d).fit(disp=0, maxiter=300, cov_type='cluster', cov_kwds={'groups': pd.factorize(d.paper_id)[0]})
        return dict(OR=np.exp(fit.params[feat]), lo=np.exp(fit.conf_int().loc[feat, 0]), hi=np.exp(fit.conf_int().loc[feat, 1]), p=fit.pvalues[feat])
    except Exception as e:
        return dict(OR=np.nan, lo=np.nan, hi=np.nan, p=np.nan)

def one_per_paper(d, feat, B=500, seed=1):
    rng = np.random.default_rng(seed); ors = []; ps = []; ncase = []
    for b in range(B):
        idx = d.groupby('paper_id').apply(lambda g: g.index[rng.integers(len(g))]).values
        s = d.loc[idx]
        t = pd.crosstab(s.y, s[feat]).reindex(index=[0, 1], columns=[0, 1], fill_value=0).values
        o, p = stats.fisher_exact(t); ors.append((t[1, 1] + .5) * (t[0, 0] + .5) / ((t[1, 0] + .5) * (t[0, 1] + .5))); ps.append(p); ncase.append(t[1, 1])
    return dict(OR_med=np.median(ors), p_med=np.median(ps), frac_p05=np.mean(np.array(ps) < 0.05), cases_exposed_med=np.median(ncase))

M2 = ['yb', 'res_med', 'ctype', 'vhh_i', 'nonhm', 'sars_i']
rows = []
def add(test, m, feat, d, res, note=''):
    r = dict(test=test, mode=m, feature=feat, n_err=int(d.y.sum()), n_ctrl=int((d.y == 0).sum()),
             err_exposed=int(((d.y == 1) & (d[feat] == 1)).sum()), ctrl_exposed=int(((d.y == 0) & (d[feat] == 1)).sum()), note=note)
    r['err_pct'] = 100 * r['err_exposed'] / max(r['n_err'], 1); r['ctrl_pct'] = 100 * r['ctrl_exposed'] / max(r['n_ctrl'], 1)
    r.update(res); rows.append(r); return r

P('=== 0. Reproduction (EM, first-occurrence error units vs clean units of the same chain types)')
for m in ['ANY', 'D_all', 'Dplus', 'E', 'A', 'notDE']:
    d = sets(m)
    t = pd.crosstab(d.y, d.trunc).reindex(index=[0, 1], columns=[0, 1], fill_value=0).values
    o, p = stats.fisher_exact(t[::-1, ::-1])
    add('fisher_raw', m, 'trunc', d, dict(OR=o, p=p))
    r = add('firth_M2', m, 'trunc', d, firth_test(d, 'trunc', M2))
    P(f'{m:6s} cases {t[1,1]}/{t[1].sum()} ({100*t[1,1]/t[1].sum():.1f}%)  ctrl {t[0,1]}/{t[0].sum()} ({100*t[0,1]/t[0].sum():.2f}%)  Fisher OR {o:.1f} p={p:.2g}  Firth-M2 OR {r["OR"]:.1f} ({r["lo"]:.1f}-{r["hi"]:.1f}) p={r["p"]:.2g}')

P('\n=== 1. Spurious truncations (SEQRES continues >5 residues after the ANARCI V parse) among EM truncated units')
E_ = U[EM & U.seq_fr4_trunc.astype(bool)]
P(E_[['pdb', 'chain', 'ctype', 'err_first_u', 'clean', 'modes_u', 'seq_last_imgt', 'mod_last_anarci', 'seq_len', 'seq_tail_after_v']][E_.spurious == 1].to_string())

P('\n=== 2. Corrected truncation (spurious ANARCI parse stops removed)')
for m in ['ANY', 'D_all', 'Dplus', 'E']:
    d = sets(m)
    r = add('fisher_corrected', m, 'trunc_true', d, dict(zip(['OR', 'p'], stats.fisher_exact(pd.crosstab(d.y, d.trunc_true).reindex(index=[1, 0], columns=[1, 0], fill_value=0).values))))
    r2 = add('firth_M2_corrected', m, 'trunc_true', d, firth_test(d, 'trunc_true', M2))
    P(f'{m:6s} {r["err_exposed"]}/{r["n_err"]} vs {r["ctrl_exposed"]}/{r["n_ctrl"]}  Fisher OR {r["OR"]:.1f} p={r["p"]:.2g}; Firth-M2 OR {r2["OR"]:.1f} ({r2["lo"]:.1f}-{r2["hi"]:.1f}) p={r2["p"]:.2g}')

P('\n=== 3. Period: truncation prevalence among clean EM units by deposition period, and year-stratified exact tests')
c = U[EM & U.clean]
P(c.groupby('per4').agg(n=('trunc', 'size'), trunc_pct=('trunc', lambda x: 100 * x.mean()), true_pct=('trunc_true', lambda x: 100 * x.mean())).round(2).to_string())
e = U[EM & U.err_first_u]
P('error units:'); P(e.groupby('per4').agg(n=('trunc', 'size'), trunc_pct=('trunc', lambda x: 100 * x.mean())).round(2).to_string())
for m in ['ANY', 'D_all', 'E']:
    for feat in ['trunc', 'trunc_true']:
        d = sets(m)
        for strat in ['yb', 'per4']:
            r = exact_strat(d, feat, strat)
            add(f'exact_strat_{strat}', m, feat, d, dict(OR=r['MH_OR'], p=r['p_exact_strat']), note=f'obs {r["obs"]} exp {r["expected"]:.2f}')
            P(f'{m:6s} {feat:10s} strat {strat:5s} MH OR {r["MH_OR"]:.1f}  exact stratified p={r["p_exact_strat"]:.2g}  obs {r["obs"]} vs expected {r["expected"]:.2f}')
        # year x chain type strata
        d['yc'] = d.yb + '_' + d.ctype
        r = exact_strat(d, feat, 'yc')
        add('exact_strat_year_ctype', m, feat, d, dict(OR=r['MH_OR'], p=r['p_exact_strat']), note=f'obs {r["obs"]} exp {r["expected"]:.2f}')
        P(f'{m:6s} {feat:10s} strat year x ctype MH OR {r["MH_OR"]:.1f}  exact stratified p={r["p_exact_strat"]:.2g}  obs {r["obs"]} vs expected {r["expected"]:.2f}')

P('\n=== 4. Paper clustering')
for m in ['ANY', 'D_all', 'E']:
    for feat in ['trunc', 'trunc_true']:
        d = sets(m)
        r = cluster_logit(d, feat, ['yb', 'res_med', 'ctype'])
        add('logit_M1_cluster_paper', m, feat, d, r)
        o = one_per_paper(d, feat, B=300)
        add('one_unit_per_paper', m, feat, d, dict(OR=o['OR_med'], p=o['p_med']), note=f'frac p<0.05 {o["frac_p05"]:.2f}; median exposed cases {o["cases_exposed_med"]}')
        P(f'{m:6s} {feat:10s} cluster-robust M1 OR {r["OR"]:.1f} ({r["lo"]:.1f}-{r["hi"]:.1f}) p={r["p"]:.2g} | one unit per paper (300 draws): median OR {o["OR_med"]:.1f} median p {o["p_med"]:.2g}, frac p<0.05 {o["frac_p05"]:.2f}')
    d = sets(m)
    P(f'   papers among exposed: cases {d[(d.y==1)&(d.trunc==1)].paper_id.nunique()} papers / {d[(d.y==1)&(d.trunc==1)].last_author.nunique()} labs; controls {d[(d.y==0)&(d.trunc==1)].paper_id.nunique()} papers / {d[(d.y==0)&(d.trunc==1)].last_author.nunique()} labs')

P('\n=== 5. Simpler known factor: the model itself ends inside FR4 (last modelled residue IMGT <125), whatever the SEQRES')
for m in ['ANY', 'D_all', 'Dplus', 'E']:
    d = sets(m).dropna(subset=['model_end_fr4'])
    t = pd.crosstab([d.model_end_fr4, d.trunc_true], d.y)
    P(f'--- {m}'); P(t.to_string())
    for feat in ['model_end_fr4', 'unbuilt_cterm']:
        r = add('firth_M2', m, feat, d, firth_test(d, feat, M2))
        P(f'{m:6s} {feat:14s} {r["err_exposed"]}/{r["n_err"]} ({r["err_pct"]:.1f}%) vs {r["ctrl_exposed"]}/{r["n_ctrl"]} ({r["ctrl_pct"]:.2f}%) Firth-M2 OR {r["OR"]:.1f} ({r["lo"]:.1f}-{r["hi"]:.1f}) p={r["p"]:.2g}')
    # does SEQRES truncation add anything beyond the model ending in FR4?
    r = add('firth_M2+model_end', m, 'trunc_true', d, firth_test(d, 'trunc_true', M2 + ['model_end_fr4']))
    P(f'{m:6s} trunc_true | M2 + model_end_fr4: OR {r["OR"]:.2f} ({r["lo"]:.2f}-{r["hi"]:.2f}) p={r["p"]:.2g}')
    dd = d[d.model_end_fr4 == 1]
    if dd.y.sum() > 0:
        t2 = pd.crosstab(dd.y, dd.trunc_true).reindex(index=[1, 0], columns=[1, 0], fill_value=0).values
        o, p = stats.fisher_exact(t2)
        add('fisher_within_model_end_fr4', m, 'trunc_true', dd, dict(OR=o, p=p))
        P(f'{m:6s} within model-ends-in-FR4 units: SEQRES-truncated {t2[0,0]}/{t2[0].sum()} errors vs {t2[1,0]}/{t2[1].sum()} controls; Fisher OR {o:.2f} p={p:.2g}')

P('\n=== 6. Error definition: do the D/E runs of truncated units reach the last modelled residue?')
for col, m in [('D_run_touches_cterm', 'D_all'), ('E_run_touches_cterm', 'E')]:
    d = sets(m); cs = d[d.y == 1]
    P(m, 'truncated cases, run reaches chain end:', cs[cs.trunc == 1][col].value_counts(dropna=False).to_dict(),
      '| non-truncated cases:', cs[cs.trunc == 0][col].value_counts(dropna=False).to_dict())
    cs2 = cs.copy(); cs2['touch'] = cs2[col].fillna(False).astype(bool)
    P('   non-chain-end D/E cases with truncated SEQRES:', int(((cs2.trunc == 1) & ~cs2.touch).sum()))
    # restrict cases to runs NOT reaching the chain end
    d2 = d[(d.y == 0) | ~d[col].fillna(False).astype(bool)]
    t = pd.crosstab(d2.y, d2.trunc).reindex(index=[1, 0], columns=[1, 0], fill_value=0).values
    o, p = stats.fisher_exact(t)
    add('fisher_cases_not_chain_end', m, 'trunc', d2, dict(OR=o, p=p))
    P(f'   cases whose run does not reach the chain end: {t[0,0]}/{t[0].sum()} truncated vs controls {t[1,0]}/{t[1].sum()}; Fisher p={p:.2g}')

P('\n=== 7. Specificity: is SEQRES N-terminal truncation (a marker of SEQRES = modelled residues) also enriched in D/E?')
for m in ['D_all', 'E']:
    d = sets(m)
    r = add('firth_M2', m, 'nt_trunc', d, firth_test(d, 'nt_trunc', M2))
    P(f'{m:6s} SEQRES starts after IMGT 1: {r["err_exposed"]}/{r["n_err"]} ({r["err_pct"]:.1f}%) vs {r["ctrl_pct"]:.1f}%  Firth-M2 OR {r["OR"]:.2f} ({r["lo"]:.2f}-{r["hi"]:.2f}) p={r["p"]:.2g}')
    r = add('firth_M2+nt_trunc', m, 'trunc_true', d, firth_test(d, 'trunc_true', M2 + ['nt_trunc']))
    P(f'{m:6s} trunc_true | M2 + nt_trunc: OR {r["OR"]:.1f} ({r["lo"]:.1f}-{r["hi"]:.1f}) p={r["p"]:.2g}')

P('\n=== 8. Alternative control: all other EM units without a first-occurrence error at this location (errors elsewhere allowed)')
for m in ['D_all', 'E']:
    d = sets(m, control='other')
    r = add('firth_M2_altctrl', m, 'trunc_true', d, firth_test(d, 'trunc_true', M2))
    P(f'{m:6s} {r["err_exposed"]}/{r["n_err"]} vs {r["ctrl_exposed"]}/{r["n_ctrl"]} ({r["ctrl_pct"]:.2f}%) Firth-M2 OR {r["OR"]:.1f} ({r["lo"]:.1f}-{r["hi"]:.1f}) p={r["p"]:.2g}')

P('\n=== 9. Leave-one-case-out (D_all, E; corrected truncation, Fisher)')
for m in ['D_all', 'E']:
    d = sets(m); ex = d[(d.y == 1) & (d.trunc_true == 1)]
    for i, r0 in ex.iterrows():
        d2 = d.drop(i)
        t = pd.crosstab(d2.y, d2.trunc_true).reindex(index=[1, 0], columns=[1, 0], fill_value=0).values
        o, p = stats.fisher_exact(t)
        P(f'{m:6s} drop {r0.pdb}_{r0.chain}: {t[0,0]}/{t[0].sum()} vs {t[1,0]}/{t[1].sum()}  OR {o:.1f} p={p:.2g}')

T = pd.DataFrame(rows)
ok = T.p.notna(); T.loc[ok, 'q_bh'] = multipletests(T.loc[ok, 'p'], method='fdr_bh')[1]
T.to_csv(f'{D_}/v02_tests.csv', index=False)
P('\n=== All tests (BH across this verifier\'s tests)')
P(T[['test', 'mode', 'feature', 'n_err', 'err_exposed', 'err_pct', 'n_ctrl', 'ctrl_exposed', 'ctrl_pct', 'OR', 'p', 'q_bh', 'note']].round(4).to_string())
LOG.close()
