"""Simpler-factor check: do the claimed artifact runs sit in poorly fitted / unbuilt neighbourhoods like genuine errors?
If they are numbering conventions on correctly built models, their local map fit should look like the rest of the
domain (percentile ~0.5), unlike genuine C/F register errors (prior finding: errors sit at poor local density).
For every first-occurrence C_FR3 / F_CDR1 run: mean within-chain percentile (V-domain residues) of the per-residue
Q-score (EM) or RSCC (X-ray) over the run window (local_all.jsonl), and unbuilt residues within +-3 of the window
(sabr/out records). Mann-Whitney / Fisher, claimed-artifact vs other runs, one row per unit (paper|V region).
Output: tables/v06_local_fit.csv
"""
import json, numpy as np, pandas as pd
from scipy.stats import mannwhitneyu, fisher_exact
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/case_files_ABCF_ARTIFACT_INDEL_confounding'
RF = pd.read_csv(f'{SP}/deeper/runs_first_occurrence.csv')
amy = set(pd.read_csv('/home/user/SAbR/register_error_analysis/retest/deeper/pdb_meta/tables/amyloid_fibril_chains.csv').pdb.str.lower())
RF = RF[RF['mode'].isin(['C_FR3', 'F_CDR1']) & ~RF.pdb.isin(amy)].copy()
RF['unit'] = RF.paper.astype(str) + '|' + RF.seqkey.astype(str)
ART = pd.read_csv('/home/user/SAbR/register_error_analysis/retest/deeper/case_files_ABCF/tables/artifact_runs.csv')
artk = set(zip(ART.pdb, ART.chain, ART.imgt0, ART.imgt1))
RF['claimed'] = [(p, c, a, b) in artk for p, c, a, b in zip(RF.pdb, RF.chain, RF.imgt0, RF.imgt1)]
need = set(zip(RF.pdb, RF.chain))
rec = {}
for fn in ['validation.jsonl', 'new.jsonl', 'snapshot_rest.jsonl']:
    for line in open(f'{SP}/sabr/out/{fn}'):
        x = json.loads(line)
        k = (x['pdb'], x['chain'])
        if k in need and k not in rec and x.get('residues'): rec[k] = x['residues']
fit = {}
with open(f'{SP}/retest/local_all.jsonl') as fh:
    for line in fh:
        if line[9:13] not in {p for p, c in need}: continue
        x = json.loads(line)
        k = (x['pdb'], x['chain'])
        if k not in need: continue
        f = x.get('feats') or {}
        key = 'Q_SCORE' if 'Q_SCORE' in f else ('RSCC' if 'RSCC' in f else None)
        if key is None: continue
        am = x['auth_map']
        fit[k] = (key, {am[int(i) - 1]: v for i, v in f[key].items() if v is not None and int(i) - 1 < len(am)})
rows = []
for r in RF.itertuples():
    k = (r.pdb, r.chain)
    R = rec.get(k); F = fit.get(k)
    d = dict(pdb=r.pdb, chain=r.chain, unit=r.unit, mode=r.mode, method=r.method, imgt0=r.imgt0, imgt1=r.imgt1, claimed=r.claimed)
    if R:
        V = [q for q in R if q[4] is not None]
        si = [q[3] for q in V if q[3] is not None]
        win = [q for q in V if r.imgt0 - 3 <= q[4] <= r.imgt1 + 3 and q[3] is not None]
        if win:
            w = sorted(q[3] for q in win)
            d['unbuilt_near'] = (w[-1] - w[0] + 1) - len(set(w))
        if F:
            key, m = F
            vals = {str(q[0]) + (q[1] or ''): m.get(str(q[0]) + (q[1] or '')) for q in V}
            allv = np.array([v for v in vals.values() if v is not None])
            wv = [vals.get(str(q[0]) + (q[1] or '')) for q in V if r.imgt0 <= q[4] <= r.imgt1]
            wv = [v for v in wv if v is not None]
            if len(allv) > 20 and wv:
                d['fit_metric'] = key; d['win_pct'] = float(np.mean([(allv < v).mean() for v in wv]))
    rows.append(d)
D = pd.DataFrame(rows)
D.to_csv(f'{OUT}/tables/v06_local_fit.csv', index=False)
U = D.groupby('unit').agg(claimed=('claimed', 'max'), win_pct=('win_pct', 'mean'), unbuilt=('unbuilt_near', 'max'), method=('method', 'first')).reset_index()
a = U[U.claimed]; b = U[~U.claimed]
print(U[U.claimed].to_string())
print('window fit percentile, claimed vs other (units):', a.win_pct.median(), len(a.win_pct.dropna()), b.win_pct.median(), len(b.win_pct.dropna()),
      mannwhitneyu(a.win_pct.dropna(), b.win_pct.dropna()))
t = [[int((a.unbuilt > 0).sum()), int((a.unbuilt == 0).sum())], [int((b.unbuilt > 0).sum()), int((b.unbuilt == 0).sum())]]
print('unbuilt within +-3 of window, claimed vs other:', t, fisher_exact(t))
for meth in ['EM', 'X-ray']:
    aa = a[a.method == meth].win_pct.dropna(); bb = b[b.method == meth].win_pct.dropna()
    if len(aa) and len(bb): print(meth, aa.median(), len(aa), bb.median(), len(bb), mannwhitneyu(aa, bb).pvalue)
