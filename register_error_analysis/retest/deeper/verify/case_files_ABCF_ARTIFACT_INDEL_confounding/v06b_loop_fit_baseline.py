"""Baseline for v06: within-chain fit percentile of the IMGT 81-83 window (FR3 C''-D loop) in CLEAN scan chains
(no disagreement run anywhere: chains.csv err_any False), by method, vs the claimed-artifact chains.
Shows whether a low percentile at 81-83 is just the loop's normal behaviour.
Output: tables/v06b_loop_fit_baseline.csv
"""
import json, random, numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/case_files_ABCF_ARTIFACT_INDEL_confounding'
C = pd.read_csv(f'{SP}/deeper/chains.csv.gz', low_memory=False, usecols=['pdb', 'chain', 'ctype', 'method', 'err_any', 'unit'])
ART = pd.read_csv('/home/user/SAbR/register_error_analysis/retest/deeper/case_files_ABCF/tables/artifact_runs.csv')
art = set(zip(ART[ART['mode'] == 'C_FR3'].pdb, ART[ART['mode'] == 'C_FR3'].chain))  # FR3 artifact chains only (run window = 81-83)
clean = C[(C.err_any == False) & (C.ctype == 'H')].drop_duplicates('unit')
random.seed(1)
samp = set(zip(clean.pdb, clean.chain)) if len(clean) < 1500 else set(random.sample(list(zip(clean.pdb, clean.chain)), 1500))
need = samp | art
meth = dict(zip(zip(C.pdb, C.chain), C.method))
rec = {}
for fn in ['validation.jsonl', 'new.jsonl', 'snapshot_rest.jsonl']:
    for line in open(f'{SP}/sabr/out/{fn}'):
        x = json.loads(line); k = (x['pdb'], x['chain'])
        if k in need and k not in rec and x.get('residues'): rec[k] = x['residues']
pdbs = {p for p, c in need}
rows = []
with open(f'{SP}/retest/local_all.jsonl') as fh:
    for line in fh:
        if line[9:13] not in pdbs: continue
        x = json.loads(line); k = (x['pdb'], x['chain'])
        if k not in need or k not in rec: continue
        f = x.get('feats') or {}
        key = 'Q_SCORE' if 'Q_SCORE' in f else ('RSCC' if 'RSCC' in f else None)
        if key is None: continue
        am = x['auth_map']; m = {am[int(i) - 1]: v for i, v in f[key].items() if v is not None and int(i) - 1 < len(am)}
        V = [q for q in rec[k] if q[4] is not None]
        vals = [(q[4], m.get(str(q[0]) + (q[1] or ''))) for q in V]
        allv = np.array([v for _, v in vals if v is not None])
        wv = [v for p, v in vals if v is not None and 81 <= p <= 83]
        if len(allv) > 20 and wv:
            rows.append(dict(pdb=k[0], chain=k[1], method=meth.get(k), artifact=k in art, pct_81_83=float(np.mean([(allv < v).mean() for v in wv]))))
D = pd.DataFrame(rows); D.to_csv(f'{OUT}/tables/v06b_loop_fit_baseline.csv', index=False)
for mth, g in D.groupby('method'):
    a = g[g.artifact].pct_81_83; b = g[~g.artifact].pct_81_83
    print(mth, 'artifact chains', round(a.median(), 3), len(a), '| clean chains', round(b.median(), 3), len(b), 'IQR', b.quantile([.25, .75]).round(3).tolist(),
          mannwhitneyu(a, b).pvalue if len(a) else '')
