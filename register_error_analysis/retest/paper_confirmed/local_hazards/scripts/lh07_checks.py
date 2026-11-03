"""Small follow-up checks for notes.md.
 (1) Model ends inside FR4 (last built IMGT < 125) for CDR3/FR4-mode units (the factor that absorbed FR4 SEQRES truncation).
 (2) Era composition of paper_strict vs conkit-missed in-window errors (AlphaFold2 training cut-off ~2018-05).
 (3) Paper concentration: units per paper in the paper_strict set.
Output: ../tables/checks.log"""
import pandas as pd
from scipy.stats import fisher_exact
O = '/home/user/SAbR/register_error_analysis/retest/paper_confirmed/local_hazards/tables'
log = open(f'{O}/checks.log', 'w')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); log.write(s + '\n')
U = pd.read_pickle(f'{O}/analysis_units.pkl.gz')
dd = lambda u: u.sort_values(['pdb', 'chain']).drop_duplicates(['mode', 'paper', 'vseq'])
P('(1) CDR3/FR4-mode units whose model ends inside FR4 (last built IMGT < 125); controls at the D+ window')
for meth in ['EM', 'all']:
    out = {}
    for cs in ['paper_strict', 'rest', 'rest_win', 'ctl_paper', 'ctl_full']:
        u = U[(U.cset == cs) & U['mode'].isin(['D+', 'D-', 'E'] if not cs.startswith('ctl') else ['D+'])]
        if meth == 'EM': u = u[u.EM]
        u = dd(u); out[cs] = (int((u.last_imgt < 125).sum()), len(u))
    P(meth, {k: f'{a}/{n} ({100*a/n:.1f}%)' for k, (a, n) in out.items()})
    for b in ['ctl_paper', 'rest']:
        a, c = out['paper_strict'], out[b]
        P(f'   paper_strict vs {b}: Fisher p={fisher_exact([[a[0], a[1]-a[0]], [c[0], c[1]-c[0]]])[1]:.3g}')
P('\n(2) release before 2018-05 (AlphaFold2 training cut-off) among in-window error units')
for cs in ['paper_strict', 'paper_incl', 'rest_win']:
    u = dd(U[U.cset == cs]); pre = (u.release.astype(str) < '2018-05-01')
    P(f'{cs}: {int(pre.sum())}/{len(u)} units released before 2018-05; by year {u.year.value_counts().sort_index().to_dict()}')
P('\n(3) paper_strict units (antibody x paper x mode) per paper')
u = dd(U[U.cset == 'paper_strict'])
P(u.groupby('paper').agg(units=('pdb', 'size'), modes=('mode', lambda s: ','.join(sorted(set(s)))), entries=('pdb', lambda s: ','.join(sorted(set(s))))).sort_values('units', ascending=False).to_string())
log.close()
