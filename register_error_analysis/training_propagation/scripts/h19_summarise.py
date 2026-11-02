"""Summarise the full ImmuneBuilder/IgFold run.
usage: h19_summarise.py readout_full.csv outdir
Unit = (deposited chain, error window). Antibody-level = unique (Fv sequence, chain type, IMGT window), so that
copies of one antibody in one or many entries count once (any copy exposed -> exposed; call = majority).

Outcomes per predictor:
  copies   ladder: >=50% of the window residues sit on the same residue as in the deposit
  det      the prediction itself shows a SAbR-vs-ANARCI run overlapping the window
           (same sign of shift as the deposit, when the deposit's shift is defined)
Exposure (was this deposited chain in the predictor's training data?):
  ImmuneBuilder (ABodyBuilder2/NanoBodyBuilder2): released <= 2021-07-31, <= 3.5 A, no unmodelled residues inside
                 either V domain (ABodyBuilder2's stated filters; NanoBodyBuilder2 assumed the same)
  IgFold:        released before 2021-07-01 (its benchmark starts then; its own filters are not documented)
"""
import sys, os, numpy as np, pandas as pd
from scipy.stats import fisher_exact
t = pd.read_csv(sys.argv[1]); O = sys.argv[2]; os.makedirs(O, exist_ok=True)
t = t[t.predictor.notna()].copy()
t['rel'] = pd.to_datetime(t.release)
gaps0 = (t.v_internal_missing.fillna(0) == 0) & (t.partner_internal_missing.fillna(0) == 0)
t['exposed'] = np.where(t.predictor == 'immunebuilder', (t.rel <= '2021-07-31') & (t.res <= 3.5) & gaps0, t.rel < '2021-07-01')
scorable = ['copies error', 'canonical', 'shifted, not copied', 'unclear']
t['ladder_ok'] = t.ladder_call.isin(scorable)
t['copies'] = t.ladder_call == 'copies error'
def same_sign(r):
    if not r.pred_run: return False
    if pd.isna(r.delta) or pd.isna(r.pred_run_shift): return True
    return any(np.sign(float(x)) == np.sign(r.delta) for x in str(r.pred_run_shift).split(';') if x)
t['pred_run'] = t.pred_run.fillna(False).astype(bool)
t['det'] = t.apply(same_sign, axis=1)
def category(r):
    if r.source == 'template_family': return f'template family {int(r.family)}'
    if r.fr_insertion is True or str(r.fr_insertion) == 'True': return 'framework insertion vs germline (detector artifact)'
    if r['kind'] == 'shift in FR3 80–86 (DE loop)': return 'FR3 80–86 DE loop (detector artifact)'
    if pd.isna(r.delta) or r['kind'] == 'no framework shift measurable': return 'no measurable shift'
    pos = range(int(r.imgt_lo), int(r.imgt_hi) + 1)
    fr = sum(not (27 <= q <= 38 or 56 <= q <= 65 or 105 <= q <= 117) for q in pos)
    return 'framework register shift' if fr >= 0.5 * len(pos) else 'mostly within a CDR'
t['category'] = t.apply(category, axis=1)
t['abkey'] = t.fv_id + '|' + t.chain_type.astype(str) + '|' + t.imgt_lo.astype(str) + '-' + t.imgt_hi.astype(str)

def tab(d, by):
    g = d.groupby(by)
    out = pd.DataFrame(dict(units=g.size(), antibodies=g.abkey.nunique(), ladder_scorable=g.ladder_ok.sum(),
                            copies=g.copies.sum(), det_in_prediction=g.det.sum(), both=g.apply(lambda x: (x.copies & x.det).sum())))
    out['pct_copies_of_scorable'] = (100 * out.copies / out.ladder_scorable.replace(0, np.nan)).round(1)
    return out
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 60)
log = open(f'{O}/full_run_summary.log', 'w')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); log.write(s + '\n')
P('Units (chain x error window) scored:', t.groupby('predictor').size().to_dict())
P('Ladder calls:'); P(t.groupby(['predictor', 'ladder_call']).size().unstack(0).to_string())
T1 = tab(t, ['predictor', 'category']); P('\nBy category (unit level):'); P(T1.to_string()); T1.to_csv(f'{O}/full_by_category.csv')
# antibody level
A = t.groupby(['predictor', 'abkey']).agg(category=('category', 'first'), exposed=('exposed', 'any'), units=('pdb', 'size'),
                                          entries=('pdb', 'nunique'), copies=('copies', 'mean'), det=('det', 'mean'),
                                          scorable=('ladder_ok', 'any'), first_release=('rel', 'min')).reset_index()
A['copies'] = A.copies >= 0.5; A['det'] = A.det >= 0.5
real = A[A.category == 'framework register shift']
P('\nAntibody level, framework register shifts only (detector artifacts and CDR-only runs excluded):')
rows = []
for p, g in real.groupby('predictor'):
    s = g[g.scorable]
    for ex, h in s.groupby('exposed'):
        rows.append(dict(predictor=p, in_training=ex, antibodies=len(h), copies=int(h.copies.sum()), det=int(h.det.sum()),
                         pct_copies=round(100 * h.copies.mean(), 1)))
    e, u = s[s.exposed], s[~s.exposed]
    if len(e) and len(u):
        orr, pv = fisher_exact([[e.copies.sum(), len(e) - e.copies.sum()], [u.copies.sum(), len(u) - u.copies.sum()]])
        P(f'  {p}: copies {e.copies.sum()}/{len(e)} in training vs {u.copies.sum()}/{len(u)} not; Fisher p={pv:.3g}')
T2 = pd.DataFrame(rows); P(T2.to_string(index=False)); T2.to_csv(f'{O}/full_exposure.csv', index=False)
# template families
tf = t[t.source == 'template_family']
if len(tf):
    T3 = tf.groupby(['family', 'template_call', 'predictor']).agg(units=('pdb', 'size'), entries=('pdb', 'nunique'), dep_run=('dep_run', 'sum'),
                                                                   copies=('copies', 'sum'), det=('det', 'sum'),
                                                                   calls=('ladder_call', lambda x: dict(x.value_counts()))).reset_index()
    P('\nProblematic-template families:'); P(T3.to_string(index=False)); T3.to_csv(f'{O}/full_template_families.csv', index=False)
# every copy, for inspection
C = t[t.copies | t.det][['predictor', 'category', 'pdb', 'chain', 'chain_type', 'mode', 'fv_id', 'release', 'res', 'exposed', 'loc', 'imgt_lo', 'imgt_hi',
                         'delta', 'dep_run_shift', 'pred_run_shift', 'ladder', 'fw_rmsd', 'ladder_call']]
P(f'\nUnits where a prediction copies the error (ladder) or carries a run in the window: {len(C)}')
P(C.sort_values(['category', 'pdb']).to_string(index=False)); C.to_csv(f'{O}/full_copies.csv', index=False)
A.to_csv(f'{O}/full_antibody_level.csv', index=False)
