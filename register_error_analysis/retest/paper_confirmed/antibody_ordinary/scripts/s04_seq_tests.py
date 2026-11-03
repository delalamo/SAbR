"""Sequence / germline / species / CDR-length-rarity / representation tests at the antibody x paper unit level.

Comparisons (rows of tables/seq_tests.csv):
  strict      paper_strict units vs window control units (PRIMARY)
  incl        paper_incl units vs window control units (sensitivity, copies included)
  rest_full   rest units vs clean units of all years (recomputed full-set reference, ledger artifacts and amyloid removed)
  rest_win    rest units released in the paper window (errors conkit missed) vs window control units
  het_case    case-only contrast: paper_strict (y=1) vs rest_win (y=0), same era -> does conkit-confirmed differ?
Each comparison is run for all methods, EM and X-ray.
Models:
  M1 = logistic GEE (independence, clustered by paper): y ~ f + C(year bin) + resolution + C(chain type) [+ C(method)]
  M2 = M1 + nanobody + non-human/mouse species + SARS-CoV-2 (the covariate equal to the feature is dropped)
  PW = M1 covariates, one vote per paper per group (weights 1/n), cluster-robust SE   (paper-clustered / one per paper)
  crude = Fisher exact (binary, Haldane CI) or Mann-Whitney (continuous)
Continuous features are standardised by the SD of the window controls (OR per SD).
Heterogeneity: z-test of the M1 log ORs strict vs rest_full, and the case-only het_case model.
Power: minimum OR detectable at 80% power (two-sided 0.05) for n strict units vs controls at the control prevalence."""
import sys, os, numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(__file__))
from aoutil import TAB, DEEP, logit_or, fisher, z_het, mde_or, f_or

X = pd.read_csv(f'{TAB}/units_frame.csv.gz', low_memory=False)
X['res'] = X.res_med
FEATS = {
    # germline novelty
    'f_first_V': 'first-time V gene in the PDB', 'f_first_J': 'first-time J gene', 'f_first_VJ': 'first-time V-J pair',
    'f_first_L123': 'first-time CDR1/2/3 length combination', 'f_log_prior_V': 'log10 prior PDB entries with same V gene',
    'f_nn_any_V': 'identity to nearest earlier PDB V domain (any)', 'f_nn_diff_V': 'identity to nearest earlier different antibody',
    'f_prior_same_ab': 'earlier PDB structure of the same antibody',
    'f_v_id_hm': 'germline V identity', 'f_low_vid85': 'germline V identity <85%', 'f_mut_total': 'somatic mutations (total)',
    'f_humanised_sig': 'humanisation signature',
    'f_rare_canonical_any': 'rare residue at a canonical position', 'f_rare_vernier_any': 'rare residue at a Vernier position',
    'f_rare_interface_any': 'rare residue at a VH-VL interface position',
    # species
    'f_nonhuman_mouse': 'non-human/mouse germline species', 'f_nanobody': 'nanobody (VHH)', 'f_rabbit': 'rabbit',
    'f_bovine': 'bovine', 'f_alpaca': 'alpaca/llama germline call',
    # literature
    'f_rare_cdr_len5': 'any CDR length <5% in the X-ray <=3 A reference', 'f_has_hq_xray': 'any X-ray <=3 A structure of the same V domain',
}
SPECIES = {'f_nonhuman_mouse', 'f_nanobody', 'f_rabbit', 'f_bovine', 'f_alpaca'}
BIN = {f for f in FEATS if set(X[f].dropna().unique()) <= {0.0, 1.0}}
ctrl_sd = X[X.control][list(FEATS)].std()

COMP = {'strict': ('paper_strict', 'control', 'yb_w'), 'incl': ('paper_incl', 'control', 'yb_w'),
        'rest_full': ('rest', 'clean_all', 'yb_f'), 'rest_win': ('rest_win', 'control', 'yb_w'),
        'het_case': ('paper_strict', 'rest_win', 'yb_w')}


def frame(comp, meth):
    a, b, yb = COMP[comp]
    d = X[(X[a] | X[b]) & ~X.any_amyloid].copy()
    d['y'] = d[a].astype(int)
    d = d[~(d[a] & d[b])] if a != 'paper_strict' or b != 'rest_win' else d
    if meth != 'all':
        d = d[d.method == meth]
    d['yb'] = d[yb]
    return d


rows = []
for f, lab in FEATS.items():
    for comp in COMP:
        for meth in ['all', 'EM', 'X-ray']:
            d = frame(comp, meth).dropna(subset=[f, 'res'])
            isb = f in BIN
            d['x'] = d[f] if isb else (d[f] - X.loc[X.control, f].mean()) / ctrl_sd[f]
            x1, x0 = d.loc[d.y == 1, f], d.loc[d.y == 0, f]
            r = dict(feature=f, label=lab, comparison=comp, method=meth, binary=isb, n1=len(x1), n0=len(x0),
                     papers1=d.loc[d.y == 1, 'paper'].nunique(), v1=x1.mean(), v0=x0.mean())
            if len(x1) < 3:
                rows.append(r); continue
            if isb:
                a1, a0 = int(x1.sum()), int(x0.sum()); r['k1'], r['k0'] = a1, a0
                fz = fisher(a1, len(x1), a0, len(x0)); r.update({f'crude_{k}': v for k, v in fz.items()})
                r['mde_OR80'] = mde_or(len(x1), x0.mean(), len(x0)) if 0 < x0.mean() < 1 else np.nan
                r['mde_OR80_low'] = mde_or(len(x1), x0.mean(), len(x0), lower=True) if 0 < x0.mean() < 1 else np.nan
                if a1 + a0 < 3 or (len(d) - a1 - a0) < 3 or a1 in (0, len(x1)) or a0 in (0, len(x0)):
                    r['sep'] = True; rows.append(r); continue   # zero cell: crude Haldane/Fisher only
            else:
                r['crude_p'] = stats.mannwhitneyu(x1, x0).pvalue; r['med1'], r['med0'] = x1.median(), x0.median()
            cov1 = ['C(yb)', 'res', 'C(ctype)'] + (['C(method)'] if meth == 'all' else [])
            cov2 = cov1 + [c for c, ff in [('vhh', 'f_nanobody'), ('nonhm', 'f_nonhuman_mouse'), ('sars', None)] if ff != f]
            if f in SPECIES:
                cov2 = cov1 + ['sars']
            m1 = logit_or(d, 'x', cov1, groups='paper'); m2 = logit_or(d, 'x', cov2, groups='paper')
            d['w'] = 1.0 / d.groupby(['paper', 'y']).y.transform('size')
            pw = logit_or(d, 'x', cov1, groups='paper', weights='w')
            # M3 (seq_novelty): first PDB structure of the antibody, human/mouse, non-nanobody units only
            d3 = d[(d.n_prior_same_ab == 0) & (d.nonhm == 0) & (d.vhh == 0)]
            m3 = {}
            if d3.y.sum() >= 5 and d3.x.nunique() > 1 and (not isb or (0 < d3[d3.y == 1].x.sum() < d3.y.sum() and 0 < d3[d3.y == 0].x.sum())):
                m3 = logit_or(d3, 'x', cov1 + ['sars'], groups='paper'); r['n1_M3'] = int(d3.y.sum())
            for nm, m in [('M1', m1), ('M2', m2), ('PW', pw), ('M3', m3)]:
                r.update({f'{nm}_{k}': v for k, v in m.items() if k in ('OR', 'lo', 'hi', 'p', 'b', 'se', 'err')})
            rows.append(r)
T = pd.DataFrame(rows)
# heterogeneity: strict vs rest_full (M1, same method stratum)
het = []
for (f, meth), g in T.groupby(['feature', 'method']):
    s = g[g.comparison == 'strict']; rf = g[g.comparison == 'rest_full']
    if len(s) and len(rf):
        a = dict(b=s.M1_b.iat[0], se=s.M1_se.iat[0]) if 'M1_b' in s and pd.notna(s.M1_b.iat[0]) else None
        b = dict(b=rf.M1_b.iat[0], se=rf.M1_se.iat[0]) if pd.notna(rf.M1_b.iat[0]) else (dict(b=rf.crude_b.iat[0], se=rf.crude_se.iat[0]) if 'crude_b' in rf and pd.notna(rf.crude_b.iat[0]) else None)
        if a is None and 'crude_b' in s and pd.notna(s.crude_b.iat[0]):
            a = dict(b=s.crude_b.iat[0], se=s.crude_se.iat[0])
        het.append(dict(feature=f, method=meth, het_z_p=z_het(a, b) if a and b else np.nan))
T = T.merge(pd.DataFrame(het), on=['feature', 'method'], how='left')
T.to_csv(f'{TAB}/seq_tests.csv', index=False)

# ---- compact print
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 500); pd.set_option('display.max_colwidth', 40)


def show(r):
    if r.binary:
        base = f"{int(r.get('k1', 0))}/{r.n1} ({100*r.v1:.1f}%) vs {int(r.get('k0', 0))}/{r.n0} ({100*r.v0:.1f}%)"
    else:
        base = f"mean {r.v1:.3f} vs {r.v0:.3f} (n {r.n1}/{r.n0})"
    m1 = f_or({k[3:]: r[k] for k in r.index if k.startswith('M1_')}) if pd.notna(r.get('M1_OR', np.nan)) else (
        'crude ' + f_or({k[6:]: r[k] for k in r.index if k.startswith('crude_')}) if pd.notna(r.get('crude_OR', np.nan)) else 'n/a')
    pw = f_or({k[3:]: r[k] for k in r.index if k.startswith('PW_')}) if pd.notna(r.get('PW_OR', np.nan)) else ''
    m2 = f_or({k[3:]: r[k] for k in r.index if k.startswith('M2_')}) if pd.notna(r.get('M2_OR', np.nan)) else ''
    m3 = f_or({k[3:]: r[k] for k in r.index if k.startswith('M3_')}) if pd.notna(r.get('M3_OR', np.nan)) else ''
    return base, m1, m2, pw, m3


out = []
for r in T.itertuples():
    b, m1, m2, pw, m3 = show(pd.Series(r._asdict()))
    out.append(dict(feature=r.feature, comp=r.comparison, meth=r.method, counts=b, M1=m1, M2=m2, PW=pw, M3=m3,
                    mde=round(r.mde_OR80, 2) if pd.notna(getattr(r, 'mde_OR80', np.nan)) else '', het=round(r.het_z_p, 3) if pd.notna(r.het_z_p) else ''))
O = pd.DataFrame(out)
O.to_csv(f'{TAB}/seq_tests_compact.csv', index=False)
with open(f'{TAB}/seq_tests_compact.txt', 'w') as fh:
    for f in FEATS:
        fh.write(f'\n=== {f}: {FEATS[f]}\n')
        fh.write(O[O.feature == f].drop(columns='feature').to_string(index=False) + '\n')
print(open(f'{TAB}/seq_tests_compact.txt').read())
