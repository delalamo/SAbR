"""Mode distribution and CDR3 flank asymmetry for paper-confirmed errors vs the full first-occurrence set.
 (1) Units (antibody x paper x mode) per mode: paper_strict, paper_incl, full first-occurrence (strict-ledger removed),
     rest, rest in the paper window (conkit-missed), rest after it. Share of CDR3/FR4 modes (D+, D-, D0, E).
 (2) Every Table S1 error (all 131 on scanned V chains) placed in IMGT by its residue range midpoint (ANARCI numbering
     of that residue), by confirmation status: does SAbR/ANARCI confirmation itself select regions?
 (3) junction README flank test: runs overlapping the CDR3 N-flank (IMGT 96-104) vs C-flank (118-128), excluding
     origins 1a/4/3 as in junction/scripts/mechanism.py, and the sign of the shift on the C-flank.
Output: ../tables/mode_distribution.csv, flank_asymmetry_paper.csv, s1_region_by_status.csv, modes_flanks.log"""
import json, numpy as np, pandas as pd
from scipy.stats import fisher_exact
from lhutil import *
log = open(f'{O}/modes_flanks.log', 'w')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n')
pd.set_option('display.width', 250)
C = chains()
R = case_runs()
R = R.drop(columns=['paper']).merge(C[['pdb', 'chain', 'paper', 'vseq', 'in_paper_window', 'unit']], on=['pdb', 'chain'], how='left')   # paper key as in chains.csv
R['mm'] = R['mode'].map(MODEMAP).fillna(R['mode'])
R['DE'] = R.mm.isin(['D+', 'D-', 'D0', 'E'])
paper_units = set(R[R.pset.isin(['strict', 'loose', 'copy', 'near_copy', 'artifact'])].unit)
fo = R[~R.copied_or_artifact.astype(bool) & ~R.flag.isin(STRICT_LEDGER)]
S = {'paper_strict': R[R.pset.eq('strict')], 'paper_incl': R[R.pset.isin(['strict', 'loose', 'copy', 'near_copy'])], 'full_first': fo,
     'rest': fo[~fo.unit.isin(paper_units)]}
S['rest_win'] = S['rest'][S['rest'].in_paper_window]; S['rest_post'] = S['rest'][~S['rest'].in_paper_window]
rows = []
for k, d in S.items():
    u = d.drop_duplicates(['mm', 'paper', 'vseq'])
    vc = u.mm.value_counts()
    row = dict(set=k, units=len(u), papers=u.paper.nunique(), **{f'n_{m}': int(vc.get(m, 0)) for m in ['A', 'F', 'B', 'C', 'D+', 'D-', 'D0', 'E', 'other']})
    row['CDR3FR4_units'] = int(u.DE.sum()); row['CDR3FR4_pct'] = 100 * u.DE.mean()
    pp = u.groupby('paper').DE.any(); row['CDR3FR4_papers_pct'] = 100 * pp.mean()
    row['EM_units'] = int(u.method.eq('EM').sum())
    rows.append(row)
M = pd.DataFrame(rows)
P('== (1) mode distribution (units = antibody x paper x mode)'); P(M.round(1).to_string(index=False))
def fx(a, b, col='DE'):
    ua = S[a].drop_duplicates(['mm', 'paper', 'vseq']); ub = S[b].drop_duplicates(['mm', 'paper', 'vseq'])
    t = [[ua[col].sum(), (~ua[col]).sum()], [ub[col].sum(), (~ub[col]).sum()]]
    OR, p = fisher_exact(t); lo, hi = Table2x2(np.array(t, float) + 0.5).oddsratio_confint()
    return f'{a} {ua[col].sum()}/{len(ua)} vs {b} {ub[col].sum()}/{len(ub)}: OR {OR:.2f} ({lo:.2f}-{hi:.2f}) p={p:.3g}'
for a, b in [('paper_strict', 'rest'), ('paper_strict', 'rest_win'), ('paper_incl', 'rest'), ('paper_strict', 'rest_post')]: P('CDR3/FR4 share', fx(a, b))
# per-paper version
for k in ['paper_strict', 'rest']:
    d = S[k]; pp = d.groupby('paper').DE.any(); P(f'papers with any CDR3/FR4 unit, {k}: {pp.sum()}/{len(pp)} ({100*pp.mean():.0f}%)')
# chi-square of the full mode distribution, paper_strict vs rest (7 modes; small cells -> Monte Carlo)
from scipy.stats import chi2_contingency
mm = ['A', 'F', 'B', 'C', 'D+', 'D-', 'E']
a = M.set_index('set').loc['paper_strict', [f'n_{m}' for m in mm]].values.astype(int); b = M.set_index('set').loc['rest', [f'n_{m}' for m in mm]].values.astype(int)
obs = chi2_contingency(np.array([a, b]))[0]
rng = np.random.default_rng(0); pool = np.repeat(np.arange(7), a + b); n1 = a.sum(); cnt = 0
for _ in range(20000):
    rng.shuffle(pool); x = np.bincount(pool[:n1], minlength=7); y = np.bincount(pool[n1:], minlength=7)
    with np.errstate(all='ignore'):
        e = np.array([x, y]); exp = e.sum(0) * e.sum(1)[:, None] / e.sum(); st = np.nansum((e - exp) ** 2 / np.where(exp > 0, exp, np.nan))
    cnt += st >= obs
P(f'mode distribution paper_strict vs rest (7 modes, permutation chi-square): p={cnt/20000:.3f}')
M.to_csv(f'{O}/mode_distribution.csv', index=False)

# (2) Table S1 errors placed in IMGT
REC = {}
for fn in ['new.jsonl', 'snapshot_rest.jsonl', 'validation.jsonl']:
    for line in open(f'{SP}/sabr/out/{fn}'):
        r = json.loads(line)
        if r.get('status') == 'ok' and r.get('residues'): REC[(r['pdb'], r['chain'])] = r['residues']
E = pd.read_csv(f'{SP}/paperconf/paper_errors_annotated.csv')
rr = E.s1_range.str.extract(r'(-?\d+)\s*-\s*(-?\d+)').astype(int)
E['rs'], E['re'] = rr[0], rr[1]
def imgt_span(r):
    v = [x[4] for x in REC[(r.pdb, r.chain)] if r.rs <= x[0] <= r.re and x[4] is not None]
    return (min(v), max(v)) if v else (np.nan, np.nan)
E[['s1_imgt0', 's1_imgt1']] = E.apply(lambda r: pd.Series(imgt_span(r)), axis=1)
def region(a, b):
    if pd.isna(a): return 'unmapped'
    if b >= 105: return 'touches CDR3/FR4 (>=105)'
    if a <= 26: return 'FR1/CDR1 start (<=26)'
    return 'CDR1-FR3 (27-104)'
E['region'] = [region(a, b) for a, b in zip(E.s1_imgt0, E.s1_imgt1)]
E['grp'] = np.where(E.status.eq('not confirmed by SAbR/ANARCI'), 'not confirmed', 'confirmed (any status)')
X = pd.crosstab(E.region, E.grp, margins=True); P('\n== (2) Table S1 V-region errors on scanned chains, by IMGT region of the S1 range and confirmation'); P(X.to_string())
t = pd.crosstab(E.region.eq('touches CDR3/FR4 (>=105)'), E.grp)
P('Fisher CDR3/FR4 vs rest, confirmed vs not confirmed: p=%.3g' % fisher_exact(t.values)[1])
E[['pdb', 'chain', 's1_range', 's1_shift', 'status', 's1_imgt0', 's1_imgt1', 'region', 'mode']].to_csv(f'{O}/s1_region_by_status.csv', index=False)

# (3) flank asymmetry
flank = {'N-flank (96-104)': (96, 104), 'C-flank (118-128)': (118, 128)}
rows = []
for k in ['paper_strict', 'paper_incl', 'full_first', 'rest', 'rest_win']:
    for excl, lab in [(['1a', '4', '3'], 'excl. 1a/4/3 (junction README)'), (['1a', '4'], 'excl. 1a/4 only')]:
        d = S[k] if k not in ('full_first', 'rest', 'rest_win') else S[k]
        d = d[~d.cls.astype(str).isin(excl)]
        for side, (a, b) in flank.items():
            s = d[(d.imgt0 <= b) & (d.imgt1 >= a)]
            ev = s.drop_duplicates(['paper', 'seqkey', 'imgt0', 'delta'])
            sg = s.delta.dropna(); se = ev.delta.dropna()
            rows.append(dict(set=k, rule=lab, side=side, runs=len(s), events=len(ev), plus_runs=int((sg > 0).sum()), minus_runs=int((sg < 0).sum()),
                             plus_events=int((se > 0).sum()), minus_events=int((se < 0).sum())))
FL = pd.DataFrame(rows); FL['pct_plus_runs'] = 100 * FL.plus_runs / (FL.plus_runs + FL.minus_runs); FL['pct_plus_events'] = 100 * FL.plus_events / (FL.plus_events + FL.minus_events)
P('\n== (3) CDR3 flank asymmetry'); P(FL.round(1).to_string(index=False))
P('junction README (all origins excl. 1a/4/3, all years): CDR3 N-flank 12 runs / 9 events; C-flank 171 runs / 80 events, +149/-22 runs (87% +1)')
for rule in FL.rule.unique():
    a = FL[(FL.set == 'paper_strict') & (FL.rule == rule) & (FL.side.str.startswith('C'))].iloc[0]
    b = FL[(FL.set == 'rest') & (FL.rule == rule) & (FL.side.str.startswith('C'))].iloc[0]
    t = [[a.plus_events, a.minus_events], [b.plus_events, b.minus_events]]
    P(f'{rule}: C-flank +1 share (events) paper_strict {a.plus_events}/{a.plus_events+a.minus_events} vs rest {b.plus_events}/{b.plus_events+b.minus_events}: Fisher p={fisher_exact(t)[1]:.3g}')
FL.to_csv(f'{O}/flank_asymmetry_paper.csv', index=False)
log.close()
