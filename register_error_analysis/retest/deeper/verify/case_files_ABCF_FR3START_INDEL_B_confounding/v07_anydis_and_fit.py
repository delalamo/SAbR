"""(a) Does the 66-75 indel produce SAbR/ANARCI disagreement of ANY length (>=1, >=2, >=3 residues in IMGT 55-80) regardless of
method/resolution? (chains with a fully modelled 60-78 window; feature from the modelled residues).
(b) Local map fit of the run residues: Q-score (EM) percentile within the chain, for first-occurrence EM B runs with vs without the
SEQRES feature. Artifacts should show ordinary fit; genuine misthreading usually poor fit.
Output: tables/v07_anydis.csv, tables/v07_fit_B_runs.csv"""
import json, numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/case_files_ABCF_FR3START_INDEL_B_confounding'
D = pd.read_pickle(f'{OUT}/tables/sabr_window_all.pkl'); D = D[D.contiguous]
D['dis1'] = (D.n_dis_55_80 >= 1).astype(int); D['dis2'] = (D.n_dis_55_80 >= 2).astype(int)
A = D.groupby(['grp', 'feat']).agg(n=('dis1', 'size'), any1=('dis1', 'mean'), ge2=('dis2', 'mean'), ge3=('dis3', 'mean')).reset_index()
print(A.round(3).to_string()); A.to_csv(f'{OUT}/tables/v07_anydis.csv', index=False)
# (b) local fit
CH = pd.read_pickle(f'{OUT}/tables/chain_features.pkl').drop_duplicates(['pdb', 'chain']).set_index(['pdb', 'chain'])
RF = pd.read_csv(f'{SP}/deeper/runs_first_occurrence.csv')
RF = RF[(RF['mode'] == 'B_CDR2FR3') & (RF.method == 'EM')]
need = set(zip(RF.pdb, RF.chain))
recs = {}
for fn in ['new', 'snapshot_rest']:
    for line in open(f'{SP}/sabr/out/{fn}.jsonl'):
        x = json.loads(line)
        if (x['pdb'], x['chain']) in need and x.get('residues'): recs[(x['pdb'], x['chain'])] = x['residues']
loc = {}
for line in open(f'{SP}/retest/local_all.jsonl'):
    x = json.loads(line)
    if (x['pdb'], x['chain']) in need: loc[(x['pdb'], x['chain'])] = x
rows = []
for r in RF.itertuples():
    rec, L = recs.get((r.pdb, r.chain)), loc.get((r.pdb, r.chain))
    if rec is None or L is None or 'Q_SCORE' not in L['feats']: continue
    q = L['feats']['Q_SCORE']; amap = L['auth_map']
    qa = {}
    for k, v in q.items():
        i = int(k) - 1
        if 0 <= i < len(amap): qa[str(amap[i])] = v
    allq = np.array([v for v in qa.values() if v is not None], float)
    vq = [qa.get(f'{x[0]}{x[1] or ""}') for x in rec if x[4] is not None]
    vq = np.array([v for v in vq if v is not None], float)
    run = [qa.get(f'{x[0]}{x[1] or ""}') for x in rec if x[4] is not None and r.imgt0 <= x[4] <= r.imgt1]
    run = np.array([v for v in run if v is not None], float)
    if len(run) == 0 or len(vq) < 20: continue
    rows.append(dict(pdb=r.pdb, chain=r.chain, imgt0=r.imgt0, imgt1=r.imgt1, feat=int(np.nan_to_num(CH.fr3start_indel.get((r.pdb, r.chain), 0))),
                     q_run=run.mean(), q_V=np.median(vq), pct_in_V=(vq < run.mean()).mean() * 100, origin=r.origin))
F = pd.DataFrame(rows).drop_duplicates(['pdb', 'chain'])
F.to_csv(f'{OUT}/tables/v07_fit_B_runs.csv', index=False)
pd.set_option('display.width', 250)
print(F.sort_values('feat').to_string())
for f in [0, 1]: print('feat', f, 'n', (F.feat == f).sum(), 'median pct of run Q within V', F[F.feat == f].pct_in_V.median().round(1))
print(mannwhitneyu(F[F.feat == 1].pct_in_V, F[F.feat == 0].pct_in_V))
