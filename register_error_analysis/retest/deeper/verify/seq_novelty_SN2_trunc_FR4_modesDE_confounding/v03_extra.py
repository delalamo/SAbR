"""Extra checks: (a) A-mode FR4-truncation association adjusted for SEQRES N-terminal truncation (non-specificity);
(b) one unit per paper that keeps the case units (cases: one per paper; controls: one random unit per paper);
(c) deepdive local map fit / unbuilt residues for the truncated D/E cases vs other D/E cases."""
import numpy as np, pandas as pd, warnings
from scipy import stats
warnings.filterwarnings('ignore')
import importlib.util, sys
D_ = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/seq_novelty_SN2_trunc_FR4_modesDE_confounding'
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
src = open(f'{D_}/v02_tests.py').read().split("M2 = ['yb'")[0]   # reuse definitions only
exec(src.replace("LOG = open(f'{D_}/v02_tests.log', 'w')", "LOG = open(f'{D_}/v03_extra.log', 'w')"))
M2 = ['yb', 'res_med', 'ctype', 'vhh_i', 'nonhm', 'sars_i']
P('=== (a) A-mode: FR4 truncation with and without adjustment for SEQRES N-terminal truncation')
for m in ['A', 'notDE']:
    d = sets(m)
    r0 = firth_test(d, 'trunc_true', M2); r1 = firth_test(d, 'trunc_true', M2 + ['nt_trunc'])
    P(f'{m:6s} trunc_true M2 OR {r0["OR"]:.1f} p={r0["p"]:.2g} | + nt_trunc OR {r1["OR"]:.1f} ({r1["lo"]:.1f}-{r1["hi"]:.1f}) p={r1["p"]:.2g}')
    r2 = firth_test(d, 'trunc_true', M2 + ['model_end_fr4'])
    P(f'{m:6s} trunc_true | M2 + model_end_fr4 OR {r2["OR"]:.2f} ({r2["lo"]:.2f}-{r2["hi"]:.2f}) p={r2["p"]:.2g}')
    P(f'   co-occurrence in controls: P(nt_trunc | FR4 trunc) = {d[(d.y==0)&(d.trunc_true==1)].nt_trunc.mean():.2f} vs {d[(d.y==0)&(d.trunc_true==0)].nt_trunc.mean():.3f}')

P('\n=== (b) one unit per paper, cases kept (one per paper), one random control per paper, 2000 draws')
rng = np.random.default_rng(7)
for m in ['D_all', 'E', 'ANY']:
    d = sets(m)
    ps = []; ors = []
    for b in range(2000):
        s = d.sample(frac=1, random_state=int(rng.integers(1e9)))
        cs = s[s.y == 1].drop_duplicates('paper_id'); ct = s[(s.y == 0) & ~s.paper_id.isin(cs.paper_id)].drop_duplicates('paper_id')
        t = np.array([[((cs.trunc_true == 1)).sum(), (cs.trunc_true == 0).sum()], [(ct.trunc_true == 1).sum(), (ct.trunc_true == 0).sum()]])
        o, p = stats.fisher_exact(t); ps.append(p); ors.append(o)
    P(f'{m:6s} cases {len(cs)} papers, controls {len(ct)} papers: median OR {np.median(ors):.1f}, median p {np.median(ps):.2g}, frac p<0.05 {np.mean(np.array(ps)<0.05):.2f}')

P('\n=== (c) deepdive window features for D+/D-/E cases: SEQRES-truncated vs not')
DD = pd.read_pickle(f'{SP}/unexpl/deepdive_units.pkl')
DD = DD[(DD.method == 'EM') & DD['mode'].isin(['D+', 'D-', 'E'])]
tr = U[['pdb', 'chain', 'trunc', 'trunc_true', 'mod_last_anarci', 'seq_last_imgt']]
DD = DD.merge(tr, on=['pdb', 'chain'], how='left')
DD['model_end_fr4'] = DD.mod_last_anarci < 125
g = DD.groupby(['mode', 'case', 'model_end_fr4']).agg(n=('pdb', 'size'), unmod_near_any=('unmod_near', lambda x: (x > 0).mean()),
                                                    win_fit_pct=('win_fit_pct', 'median'), trunc=('trunc', 'mean'))
P(g.round(3).to_string())
P(DD[DD.case & (DD.trunc == 1)][['mode', 'pdb', 'chain', 'year', 'res', 'unmod_near', 'win_fit_pct', 'mod_last_anarci', 'seq_last_imgt', 'irm', 'fit']].to_string())
LOG.close()
