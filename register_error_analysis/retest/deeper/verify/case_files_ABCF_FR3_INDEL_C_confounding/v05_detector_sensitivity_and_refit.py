"""(1) Detector sensitivity: per chain, SAbR/ANARCI FR3 disagreement (any, >=3) for FR3-atypical vs typical chains, in the
high-resolution validation set, the 3-5 A X-ray scan and the EM scan (fully modelled, contiguous FR3 only).
(2) Re-fit of the FR3 indel -> C association after re-classifying the 10 FR3-atypical C units with the independent-
structure evidence of v04 (and investigator s09/s10):
   artifact (reproduced in independent structures or deterministic pattern): 9bop, 9e89 (investigator), 8c8t
     (85A-D insertion: 31/31 other-paper chains disagree; sibling chain 8c8t R already classed as artifact by the
     pipeline), 7m3l (identical 90-95 -1 disagreement in 3gk8 at 2.0 A and 7m3n at 2.4 A)
   probable artifact: 9yfu (same VHH: identical disagreement in 8tb7 at 2.94 A; none in 9jrt / 9jro)
   copy: 8xlk (identical V and identical 93-95 run as earlier 7te9, other lab; copy rule missed it because SEQRES
     carries a signal peptide)
   model-specific (kept): 7te9 (7te4 2.46 A agrees), 7y0o (7y0n agrees), 9dm7, 9ml5
Output: tables/detector_sensitivity.csv, tables/refit.csv
"""
import numpy as np, pandas as pd, statsmodels.formula.api as smf, warnings
from scipy.stats import fisher_exact
warnings.filterwarnings('ignore')
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/case_files_ABCF_FR3_INDEL_C_confounding'
P = pd.read_csv(f'{OUT}/tables/fr3_pattern_chains.csv.gz')
P = P[P.contiguous & P.spanned].copy()
normlen = lambda pat, ct: pat == '()' or (ct == 'K' and pat == '()')  # norm gaps already removed in v02
P['atyp'] = P.pattern != '()'
# lambda (81,82) and ()+81A, ()+82A are length variants within the lambda norm {36, 38}; treat lambda (81, 82) as typical
P.loc[(P.ct == 'L') & (P.pattern == '(81, 82)'), 'atyp'] = False
P['grp'] = np.where(P.set == 'hires', 'hires <3.0 A (any method)', np.where(P.method == 'X-ray', 'X-ray 3-5 A scan', np.where(P.method == 'EM', 'EM 3-5 A scan', 'other')))
P['dis_any'] = P.n_dis >= 1
P['dis_12'] = P.n_dis.between(1, 2)
P['dis3'] = P.n_dis >= 3
rows = []
for (g, a), d in P.groupby(['grp', 'atyp']):
    rows.append(dict(group=g, fr3_atypical=a, n=len(d), dis_any=d.dis_any.mean(), dis_1_2=d.dis_12.mean(), dis_ge3=d.dis3.mean()))
S = pd.DataFrame(rows)
S.to_csv(f'{OUT}/tables/detector_sensitivity.csv', index=False)
pd.set_option('display.width', 250)
print(S.round(4).to_string())
for g in ['hires <3.0 A (any method)', 'X-ray 3-5 A scan', 'EM 3-5 A scan']:
    d = P[P.grp == g]
    t = [[(d.atyp & d.dis_12).sum(), (d.atyp & ~d.dis_12).sum()], [(~d.atyp & d.dis_12).sum(), (~d.atyp & ~d.dis_12).sum()]]
    print(g, '1-2 residue FR3 disagreement atyp vs typ: OR %.1f p %.2g' % fisher_exact(t))

# ---------- (2) refit
U = pd.read_csv(f'{OUT}/tables/units_em_flags.csv.gz')
U['first_modes'] = U.first_modes.fillna('')
U['y'] = U.first_modes.str.contains('C_FR3').astype(int)
art = {'9bop', '9e89', '8c8t', '7m3l'}; prob = {'9yfu'}; copy = {'8xlk'}
pdbs = U.set_index('unit').pdb.to_dict()
rows = []


def fit(d, name):
    ne, nc = int(d.y.sum()), int((d.y == 0).sum())
    a = int(d[d.y == 1].fr3_atyp.sum()); c = int(d[d.y == 0].fr3_atyp.sum())
    orc, pf = fisher_exact([[a, ne - a], [c, nc - c]])
    r = dict(test=name, n_error=ne, n_control=nc, error_value=f'{a/ne:.1%} ({a}/{ne})', control_value=f'{c/nc:.2%} ({c}/{nc})', OR_crude=orc, p_fisher=pf)
    try:
        if min(a, ne - a, c, nc - c) == 0: raise ValueError('zero cell: Fisher only')
        m = smf.logit('y ~ fr3_atyp + ybin + res + cclass', data=d).fit(disp=0, cov_type='cluster', cov_kwds={'groups': pd.factorize(d.paper)[0]})
        r.update(OR_adj=np.exp(m.params['fr3_atyp']), lo=np.exp(m.conf_int().loc['fr3_atyp', 0]), hi=np.exp(m.conf_int().loc['fr3_atyp', 1]), p_adj=m.pvalues['fr3_atyp'])
    except Exception as ex:
        r.update(note=str(ex)[:80])
    rows.append(r)


def frame(drop, ctrl='clean'):
    d = U[~(U.y.astype(bool) & U.pdb.isin(drop))].copy()
    d = d[(d.y == 1) | (~d.err_any.astype(bool))] if ctrl == 'clean' else d
    return d[d.cclass.isin(set(d[d.y == 1].cclass))]
fit(frame(set()), 'S0 as investigator (no removals)')
fit(frame({'9bop', '9e89'}), 'S1 investigator removals (9bop, 9e89)')
fit(frame(art), 'S2 remove all 4 verified artifacts')
fit(frame(art | prob), 'S3 remove 4 artifacts + probable artifact 9yfu')
fit(frame(art | prob | copy), 'S4 S3 + treat 8xlk as copy of 7te9')
fit(frame(art | prob | copy, ctrl='all'), 'S5 S4 with control = all EM units without a first-occurrence C run')
# S6: S4 and remove from controls / exposure the deterministic-pattern units that can never be clean (balance arms)
d = frame(art | prob | copy, ctrl='all')
d = d[~((d.y == 0) & (d.conv_prone == 1))]
fit(d, 'S6 S5 but convention-prone patterns removed from the control arm')
# S7: unit-level period restriction for the residual (genuine-looking) cases
d = frame(art | prob | copy)
fit(d[d.ybin.isin(['2020-21'])], 'S7a S4 restricted to 2020-21 (peak-error period)')
fit(d[d.ybin.isin(['2022-23', '>=2024'])], 'S7b S4 restricted to 2022+')
R = pd.DataFrame(rows)
R.to_csv(f'{OUT}/tables/refit.csv', index=False)
print(R.round(4).to_string())
print('residual exposed C units:', sorted(set(frame(art | prob | copy).query('y==1 and fr3_atyp==1').pdb)))

# ---------- (3) pattern-matched: is P(>=3 FR3 disagreement | same indel pattern) higher in EM 3-5 A than in high-res / X-ray?
from statsmodels.stats.contingency_tables import StratifiedTable
A = P[P.atyp].copy()
A = A.drop_duplicates(['pdb', 'pattern'])  # one chain per entry and pattern
A['em'] = A.grp == 'EM 3-5 A scan'
A['ref'] = A.grp.isin(['hires <3.0 A (any method)', 'X-ray 3-5 A scan'])
A = A[A.em | A.ref]
tabs, info = [], []
for pat, g in A.groupby('pattern'):
    t = np.array([[(g.em & g.dis3).sum(), (g.em & ~g.dis3).sum()], [(g.ref & g.dis3).sum(), (g.ref & ~g.dis3).sum()]], float)
    info.append(dict(pattern=pat, em_n=int(g.em.sum()), em_dis3=int((g.em & g.dis3).sum()), ref_n=int(g.ref.sum()), ref_dis3=int((g.ref & g.dis3).sum())))
    if t[0].sum() > 0 and t[1].sum() > 0 and t[:, 0].sum() > 0 and t[:, 1].sum() > 0:
        tabs.append(t + 0.5 * (t == 0).any())
I = pd.DataFrame(info).sort_values('em_n', ascending=False)
I.to_csv(f'{OUT}/tables/pattern_matched_em_vs_ref.csv', index=False)
print(I.head(25).to_string())
st = StratifiedTable(tabs)
print('pattern-matched MH OR (EM vs high-res/X-ray) for >=3 FR3 disagreement: %.2f (%.2f-%.2f) p=%.3g, informative strata=%d' % (
    st.oddsratio_pooled, *st.oddsratio_pooled_confint(), st.test_null_odds().pvalue, len(tabs)))

# ---------- (4) strictest: only the two long, clearly model-specific register shifts count (9dm7: 72-92 +1 next to
# unbuilt residues, template FR3 2 residues longer; 9ml5: 85-95 -1, opposite to the 81-83 +1 seen in all high-res
# rabbit (84,) structures); 7te9 (3-residue run at the gap; 7te4 chain H itself has 1 residue off) and 7y0o (sibling
# 7y0n has 1 residue off; run is a CDR3/FR4 junction slip) removed as uncertain
rows = []
fit(frame(art | prob | copy | {'7te9', '7y0o'}), 'S8 only 9dm7 and 9ml5 count; uncertain 7te9/7y0o removed')
fit(frame(art | prob | copy | {'7te9', '7y0o'}, ctrl='all'), 'S8b S8 with control = all EM units without a first-occurrence C run')
R8 = pd.DataFrame(rows)
print(R8.round(4).to_string())
pd.concat([R, R8]).to_csv(f'{OUT}/tables/refit.csv', index=False)
