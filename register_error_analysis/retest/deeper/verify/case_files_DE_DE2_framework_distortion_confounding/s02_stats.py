"""Adversarial confounding tests of DE2 (framework distortion in confirmed EM D/E error chains).
Outputs: tests_metric_variants.csv, tests_adjustment.csv, tests_strata.csv, tests_misc.csv (BH q across all p)."""
import warnings, numpy as np, pandas as pd, json
import statsmodels.formula.api as smf
from statsmodels.discrete.conditional_models import ConditionalLogit
from statsmodels.stats.multitest import multipletests
from scipy.stats import mannwhitneyu, spearmanr, wilcoxon
from s00_reproduce import load_sample, sets, D, OUT, MODES
warnings.filterwarnings('ignore')

X = load_sample()
F = pd.read_csv(f'{OUT}/features_recomputed.csv')
CH = pd.read_csv(f'{D}/chains.csv.gz')
cov = CH[['pdb', 'chain', 'species', 'vfam', 'v_id', 'L1_atyp', 'L2_atyp', 'L3_atyp', 'cdr3_len', 'sars2', 'lq_z',
          'irm', 'fit', 'entries_per_paper', 'last_author']]
X = X.merge(F, on=['pdb', 'chain'], how='left').merge(cov, on=['pdb', 'chain'], how='left')
X['dep_y'] = pd.to_datetime(X.deposit).dt.year + pd.to_datetime(X.deposit).dt.dayofyear / 365.25
X['mouse'] = (X.species == 'mouse').astype(int)
X['sars2'] = X.sars2.astype(float)
X['atyp'] = (X[['L1_atyp', 'L2_atyp', 'L3_atyp']].astype(float).sum(1) > 0).astype(int)
X['irm_pdb'] = X.irm.astype(str).str.contains('PDB|experimental', case=False).astype(int)
X['res2'] = X.res ** 2
e, mc, wp = sets(X)
print('irm levels', X.irm.value_counts().to_dict(), 'fit', X.fit.value_counts().to_dict())

rows = []


def z(s): return (s - s.mean()) / s.std()


def matched(var, e, c, extra='', label=''):
    d = pd.concat([e, c]).dropna(subset=[var]).copy()
    for col in [t.strip() for t in extra.replace('C(', '').replace(')', '').split('+') if t.strip()]:
        d = d.dropna(subset=[col])
    d['x'] = z(d[var])
    f = 'err ~ x + C(yb) + res + C(ct)' + (' + ' + extra if extra else '')
    out = dict(test=label or f'matched {var}', metric=var, n_err=int(d.err.sum()), n_ctl=int((d.err == 0).sum()),
               err_mean=d[d.err == 1][var].mean(), ctl_mean=d[d.err == 0][var].mean(),
               err_median=d[d.err == 1][var].median(), ctl_median=d[d.err == 0][var].median(), adjusted=f)
    try:
        m = smf.logit(f, data=d).fit(disp=0, cov_type='cluster', cov_kwds={'groups': pd.factorize(d.pp)[0]}, maxiter=200)
        out.update(OR_perSD=float(np.exp(m.params['x'])), lo=float(np.exp(m.conf_int().loc['x', 0])),
                   hi=float(np.exp(m.conf_int().loc['x', 1])), p=float(m.pvalues['x']))
        # difference in Angstrom (linear, cluster-robust)
        l = smf.ols(f'{var} ~ err + C(yb) + res + C(ct)' + (' + ' + extra if extra else ''), data=d).fit(
            cov_type='cluster', cov_kwds={'groups': pd.factorize(d.pp)[0]})
        out.update(diff_A=float(l.params['err']), diff_p=float(l.pvalues['err']))
    except Exception as ex:
        out.update(note=str(ex)[:60])
    return out


def within(var, e, c, covs=(), label=''):
    d = pd.concat([e, c]).dropna(subset=[var] + list(covs))
    d = d[d.pp.isin(d[d.err == 1].pp) & d.pp.isin(d[d.err == 0].pp)].copy()
    d['x'] = z(d[var])
    cols = ['x'] + list(covs)
    for cc in covs: d[cc] = z(d[cc])
    out = dict(test=label or f'within-paper {var}', metric=var, n_err=int(d.err.sum()), n_ctl=int((d.err == 0).sum()),
               n_papers=d.pp.nunique(), err_mean=d[d.err == 1][var].mean(), ctl_mean=d[d.err == 0][var].mean(),
               adjusted='conditional logit within paper' + (' + ' + ','.join(covs) if covs else ''))
    try:
        m = ConditionalLogit(d.err.values, d[cols].values, groups=pd.factorize(d.pp)[0]).fit(disp=0)
        out.update(OR_perSD=float(np.exp(m.params[0])), p=float(m.pvalues[0]))
    except Exception as ex:
        out.update(note=str(ex)[:60])
    # paired paper-level difference (error mean - control mean per paper), Wilcoxon
    g = d.groupby(['pp', 'err'])[var].mean().unstack()
    diff = (g[1] - g[0]).dropna()
    out['papers_err_higher'] = int((diff > 0).sum()); out['paper_median_diff'] = float(diff.median())
    out['paper_wilcoxon_p'] = float(wilcoxon(diff).pvalue) if len(diff) >= 5 else np.nan
    return out


# ---------- 1. metric variants ----------
metrics = ['fw_orig', 'fw_core', 'fw_far', 'fw_near', 'fw_med', 'fw_frac_gt2', 'fw_panel_min', 'fw_panel_med',
           'fw_seqref', 'dev_fr1', 'dev_fr2', 'dev_fr3a', 'dev_fr3b']
for v in metrics:
    rows.append(dict(block='metric', **matched(v, e, mc)))
    rows.append(dict(block='metric', **within(v, e, wp)))

# ---------- 2. adjustment for confounders (matched controls, fw_orig and fw_core) ----------
adj = {'+ Q-score (framework, EM local)': 'q_fw', '+ map-model fit lq_z': 'lq_z', '+ unbuilt V residues': 'n_unbuilt_v',
       '+ resolution^2': 'res2', '+ deposition date (continuous)': 'dep_y', '+ SARS-CoV-2': 'sars2', '+ mouse': 'mouse',
       '+ V identity': 'v_id', '+ atypical CDR length': 'atyp', '+ CDR3 length': 'cdr3_len',
       '+ starting model type': 'C(irm)', '+ fitting protocol': 'C(fit)', '+ seq-matched ref identity': 'seqref_fw_identity',
       '+ entries per paper': 'entries_per_paper',
       'full: Q + lq_z + unbuilt + dep + SARS2 + mouse + v_id + atyp + irm':
           'q_fw + lq_z + n_unbuilt_v + dep_y + sars2 + mouse + v_id + atyp + C(irm)'}
for v in ['fw_orig', 'fw_core', 'fw_seqref']:
    rows.append(dict(block='adjust', **matched(v, e, mc, '', f'{v}: base (year bin, res, chain type)')))
    for lab, ex in adj.items():
        rows.append(dict(block='adjust', **matched(v, e, mc, ex, f'{v}: {lab}')))
    for cv in [('q_fw',), ('res',), ('q_fw', 'res', 'n_unbuilt_v'), ('lq_z',)]:
        rows.append(dict(block='adjust', **within(v, e, wp, cv, f'{v}: within paper + {"+".join(cv)}')))

# ---------- 3. strata ----------
def strat(lab, me, mm, mw):
    rows.append(dict(block='strata', stratum=lab, **matched('fw_orig', me, mm, '', f'stratum {lab}: matched')))
    if len(mw) and me.pp.isin(mw.pp).sum() >= 4:
        rows.append(dict(block='strata', stratum=lab, **within('fw_orig', me, mw, (), f'stratum {lab}: within paper')))

for lab, f in {'<=2020': lambda d: d.year <= 2020, '2021': lambda d: d.year == 2021, '>=2022': lambda d: d.year >= 2022,
               'SARS-CoV-2': lambda d: d.sars2 == 1, 'non-SARS-CoV-2': lambda d: d.sars2 == 0,
               'res<3.5': lambda d: d.res < 3.5, 'res>=3.5': lambda d: d.res >= 3.5,
               'heavy (non-VHH)': lambda d: d.ct == 'H', 'light (K/L)': lambda d: d.ct.isin(['K', 'L']),
               'VHH': lambda d: d.ct == 'VHH', 'human': lambda d: d.species == 'human', 'mouse': lambda d: d.species == 'mouse',
               'Q_fw>=median': lambda d: d.q_fw >= X.q_fw.median(), 'Q_fw<median': lambda d: d.q_fw < X.q_fw.median(),
               'no unbuilt V residues': lambda d: d.n_unbuilt_v == 0}.items():
    me, mm, mw = e[f(e)], mc[f(mc)], wp[f(wp)]
    if len(me) >= 5:
        try: strat(lab, me, mm, mw)
        except Exception as ex: print('stratum fail', lab, ex)

# ---------- 4. one unit per paper (resampling) ----------
rng = np.random.default_rng(1)
ors, ps = [], []
for b in range(300):
    ee = e.groupby('pp').sample(1, random_state=int(rng.integers(1e9)))
    cc = mc.groupby('pp').sample(1, random_state=int(rng.integers(1e9)))
    cc = cc[~cc.pp.isin(ee.pp)]
    d = pd.concat([ee, cc]).copy(); d['x'] = z(d.fw_orig)
    try:
        m = smf.logit('err ~ x + C(yb) + res + C(ct)', data=d).fit(disp=0)
        ors.append(np.exp(m.params['x'])); ps.append(m.pvalues['x'])
    except Exception: pass
rows.append(dict(block='misc', test='one chain per paper (300 resamples), matched, fw_orig', metric='fw_orig',
                 n_err=int(e.pp.nunique()), n_ctl=int(mc.pp.nunique()), OR_perSD=float(np.median(ors)),
                 lo=float(np.percentile(ors, 2.5)), hi=float(np.percentile(ors, 97.5)), p=float(np.median(ps)),
                 note=f'fraction p<0.05: {np.mean(np.array(ps) < 0.05):.2f}'))

# ---------- 5. error-definition / selection checks ----------
ea = X[(X.group == 'error') & X['mode'].isin(MODES)]
cf, nc = ea[ea.confirm != 'not confirmed'], ea[ea.confirm == 'not confirmed']
rows.append(dict(block='misc', test='confirmed vs not-confirmed errors (selection by superposition test)', metric='fw_orig',
                 n_err=len(cf), n_ctl=len(nc), err_mean=cf.fw_orig.mean(), ctl_mean=nc.fw_orig.mean(),
                 p=float(mannwhitneyu(cf.fw_orig, nc.fw_orig).pvalue), adjusted='Mann-Whitney'))
rows.append(dict(block='misc', test='not-confirmed errors vs matched clean', **{k: v for k, v in matched('fw_orig', nc, mc).items() if k != 'test'}))
rows.append(dict(block='misc', test='all D/E errors (incl. not confirmed) vs matched', **{k: v for k, v in matched('fw_orig', ea, mc).items() if k != 'test'}))
cl = pd.concat([mc, wp]).drop_duplicates(['pdb', 'chain'])
r_, p_ = spearmanr(cl.n_disagree, cl.fw_orig)
rows.append(dict(block='misc', test='clean controls: sub-run SAbR/ANARCI disagreements vs fw_orig (Spearman)', metric='fw_orig',
                 n_ctl=len(cl), err_mean=float(r_), p=float(p_), adjusted='rho in err_mean column'))
for nm, var in [('Q-score framework', 'q_fw'), ('unbuilt V residues', 'n_unbuilt_v'), ('lq_z', 'lq_z'), ('res', 'res')]:
    r_, p_ = spearmanr(cl[var], cl.fw_orig, nan_policy='omit')
    rows.append(dict(block='misc', test=f'clean controls: fw_orig vs {nm} (Spearman rho)', metric='fw_orig',
                     n_ctl=int(cl[var].notna().sum()), err_mean=float(r_), p=float(p_), adjusted='rho in err_mean column'))
    rows.append(dict(block='misc', test=f'errors vs matched: {nm}', metric=var, n_err=int(e[var].notna().sum()),
                     n_ctl=int(mc[var].notna().sum()), err_mean=e[var].mean(), ctl_mean=mc[var].mean(),
                     p=float(mannwhitneyu(e[var].dropna(), mc[var].dropna()).pvalue), adjusted='Mann-Whitney'))
# leave-one-paper-out within paper
lo = []
for p in e.pp.unique():
    o = within('fw_orig', e[e.pp != p], wp[wp.pp != p])
    lo.append(o.get('OR_perSD', np.nan))
lo = np.array(lo); lo = lo[np.isfinite(lo)]
rows.append(dict(block='misc', test='within paper, leave-one-paper-out OR range', metric='fw_orig',
                 OR_perSD=float(np.median(lo)), lo=float(lo.min()), hi=float(lo.max())))

T = pd.DataFrame(rows)
ok = T.p.notna() & ~T.test.str.contains('Spearman|confirmed vs not|leave-one')
T.loc[ok, 'q'] = multipletests(T.loc[ok, 'p'], method='fdr_bh')[1]
T.to_csv(f'{OUT}/tests_all.csv', index=False)
pd.set_option('display.width', 260); pd.set_option('display.max_rows', 500); pd.set_option('display.max_colwidth', 70)
cols = ['block', 'test', 'n_err', 'n_ctl', 'n_papers', 'err_mean', 'ctl_mean', 'OR_perSD', 'lo', 'hi', 'p', 'q', 'diff_A',
        'diff_p', 'papers_err_higher', 'paper_median_diff', 'paper_wilcoxon_p', 'note']
print(T[[c for c in cols if c in T]].round(4).to_string())
