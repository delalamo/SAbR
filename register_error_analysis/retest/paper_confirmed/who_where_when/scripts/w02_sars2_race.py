"""The early SARS-CoV-2 race (2020-21) in the paper window: where do the paper-confirmed errors sit in the order of
SARS-CoV-2 antibody cryo-EM entries, and is the SARS-CoV-2 share of paper cases different from window controls and from
the in-window errors that are not paper-confirmed (rest_w)?  Reuses verify/antigen_competition_no_competitive_pressure_confounding
entries_frame.csv.gz (same ordering as its v03_race_sars2.py: deposition date of bound EM SARS-CoV-2 entries)."""
import sys; sys.path.insert(0, '.')
from ww_common import *
from scipy.stats import fisher_exact
E = entry_flags()
m = pd.read_csv(f'{D}/verify/antigen_competition_no_competitive_pressure_confounding/entries_frame.csv.gz').drop(columns=['clean']).merge(
    E[['pdb', 'case', 'case_i', 'ctrl', 'gen', 'rest', 'rest_w', 'win_all', 'clean', 'win']], on='pdb')
m['dep'] = pd.to_datetime(m.deposit)
em = m[(m.method == 'EM') & m.keep_first].copy()
s = em[em.sars2.astype(bool)].sort_values('dep').reset_index(drop=True); s['rank'] = np.arange(1, len(s) + 1)
s['status'] = np.select([s.case, s.case_i, s.rest, s.gen, s.clean], ['paper case', 'paper copy', 'rest (not paper-confirmed)', 'gen', 'clean'], 'other run')
print('SARS-CoV-2 EM entries (keep_first):', len(s), '| first deposit', s.dep.min().date())
print('ranks of paper-strict case entries:', s.loc[s.case, 'rank'].tolist())
print('ranks of rest (not paper-confirmed) error entries <= 2022-04:', s.loc[s.rest & s.win, 'rank'].tolist())
rows = []
for lo, hi in [(1, 25), (26, 50), (51, 100), (101, 200), (201, 400), (401, 10 ** 6)]:
    t = s[(s['rank'] >= lo) & (s['rank'] <= hi)]
    rows.append(dict(ranks=f'{lo}-{hi if hi < 10**6 else "end"}', first_dep=str(t.dep.min().date()), last_dep=str(t.dep.max().date()), n=len(t),
                     paper_case_entries=int(t.case.sum()), paper_case_papers=t.loc[t.case, 'paper'].nunique(), paper_incl_copies=int(t.case_i.sum()),
                     rest_errors=int(t.rest.sum()), rest_errors_in_window=int((t.rest & t.win).sum()), any_gen=int(t.gen.sum())))
R = pd.DataFrame(rows); print(R.to_string(index=False)); R.to_csv(f'{T}/w02_sars2_rank_bins.csv', index=False)
first50 = s[s['rank'] <= 50]
print('\nfirst 50: paper-case entries by paper:', first50[first50.case].groupby('paper').pdb.apply(lambda x: ' '.join(x)).to_dict())
print('first 50: rest error entries by paper:', first50[first50.rest].groupby('paper').pdb.apply(lambda x: ' '.join(x)).to_dict())

# quarter time course in the window, SARS2 vs non-SARS2 (EM, bound or fiducial-only)
em['q'] = em.dep.dt.to_period('Q').astype(str)
w = em[(em.dep >= '2019-10-01') & (em.dep < '2022-04-01')]
Q = w.groupby(['q', 'sars2']).agg(n=('pdb', 'size'), case=('case', 'sum'), case_i=('case_i', 'sum'), rest_w=('rest_w', 'sum'), gen=('gen', 'sum'), ctrl=('ctrl', 'sum')).reset_index()
Q['case_rate_%'] = (100 * Q.case / Q.n).round(1); Q['gen_rate_%'] = (100 * Q.gen / Q.n).round(1)
print('\n', Q.to_string(index=False)); Q.to_csv(f'{T}/w02_quarter_timecourse.csv', index=False)

# share SARS2: paper cases vs controls vs rest_w (window, EM, deposited 2020+ so SARS2 is possible)
x = em[(em.dep >= '2020-01-01')]
out = []
for g, ctl in [('case', 'ctrl'), ('case_i', 'ctrl'), ('rest_w', 'ctrl')]:
    a = x[x[g]]; c = x[x[ctl] & ~x.gen & ~x.case_i]
    tab = [[int(a.sars2.sum()), int((~a.sars2.astype(bool)).sum())], [int(c.sars2.sum()), int((~c.sars2.astype(bool)).sum())]]
    orr, p = fisher_exact(tab); out.append(dict(group=g, n=len(a), sars2_pct=100 * a.sars2.mean(), ctrl_n=len(c), ctrl_sars2_pct=100 * c.sars2.mean(), OR=orr, p=p))
# case-only: paper case vs rest_w
a = x[x.case]; b = x[x.rest_w]
tab = [[int(a.sars2.sum()), len(a) - int(a.sars2.sum())], [int(b.sars2.sum()), len(b) - int(b.sars2.sum())]]
orr, p = fisher_exact(tab); out.append(dict(group='case vs rest_w (case-only, entries)', n=len(a), sars2_pct=100 * a.sars2.mean(), ctrl_n=len(b), ctrl_sars2_pct=100 * b.sars2.mean(), OR=orr, p=p))
ap = a.groupby('paper').sars2.max(); bp = b.groupby('paper').sars2.max()
tab = [[int(ap.sum()), len(ap) - int(ap.sum())], [int(bp.sum()), len(bp) - int(bp.sum())]]
orr, p = fisher_exact(tab); out.append(dict(group='case vs rest_w (case-only, papers)', n=len(ap), sars2_pct=100 * ap.mean(), ctrl_n=len(bp), ctrl_sars2_pct=100 * bp.mean(), OR=orr, p=p))
O = pd.DataFrame(out); print('\nSARS-CoV-2 share, EM entries deposited 2020+ in the window (Fisher):\n', O.round(3).to_string(index=False)); O.to_csv(f'{T}/w02_sars2_share.csv', index=False)
print('\npaper-case SARS2 papers (entries):', a[a.sars2.astype(bool)].groupby('paper').pdb.apply(list).to_dict())
s[['rank', 'pdb', 'deposit', 'paper', 'status', 'labs_acc_1m']].to_csv(f'{T}/w02_sars2_entry_order.csv', index=False)
