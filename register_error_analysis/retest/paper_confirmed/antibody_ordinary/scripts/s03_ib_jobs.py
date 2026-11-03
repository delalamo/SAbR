"""ImmuneBuilder job list for the paper-window structural comparison (run with struct_unusual/scripts/s07_ib_run.py,
same protocol as the deeper dive: model 0 unrefined + ensemble error per residue).
Cases: every unique (Hseq, Lseq) of the case Fvs and every unique case VHH sequence (re-run to get ensemble error).
Controls: random 70 EM + 70 X-ray control Fvs and all 80 control VHHs of tables/fv_abangle_window.csv (seed 5).
Output: scratchpad/paperord/ib_jobs_{0,1}.json (two halves for two processes), tables/ib_jobs.csv"""
import sys, os, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from aoutil import SP, TAB, CACHE

A = pd.read_csv(f'{TAB}/fv_abangle_window.csv', dtype={'H': str, 'L': str})
d = pd.read_csv(f'{SP}/deeper/chains.csv.gz', usecols=['pdb', 'chain', 'vseq'], dtype={'chain': str})
seq = dict(zip(zip(d.pdb, d.chain), d.vseq))
A['Hseq'] = [seq.get((p, h)) for p, h in zip(A.pdb, A.H)]
A['Lseq'] = [seq.get((p, l)) if isinstance(l, str) else '' for p, l in zip(A.pdb, A.L)]
rng = np.random.default_rng(5)
jobs = []
cs = A[A.role == 'case'].drop_duplicates(['Hseq', 'Lseq'])
for r in cs.itertuples():
    fid = f'x_{r.pdb}_{r.H}' + (f'_{r.L}' if r.kind == 'Fv' else '')
    jobs.append(dict(fv_id=fid, Hseq=r.Hseq, Lseq=r.Lseq if r.kind == 'Fv' else '', mode='fv' if r.kind == 'Fv' else 'nanobody', grp='case'))
ct = A[(A.role == 'control') & (A.kind == 'Fv')]
ct = pd.concat([g.iloc[rng.permutation(len(g))[:70]] for _, g in ct.groupby('method')])
for r in ct.itertuples():
    jobs.append(dict(fv_id=f'c_{r.pdb}_{r.H}_{r.L}', Hseq=r.Hseq, Lseq=r.Lseq, mode='fv', grp='control'))
for r in A[(A.role == 'control') & (A.kind == 'VHH')].itertuples():
    jobs.append(dict(fv_id=f'c_{r.pdb}_{r.H}', Hseq=r.Hseq, Lseq='', mode='nanobody', grp='control'))
jobs = [j for j in jobs if isinstance(j['Hseq'], str) and (j['mode'] == 'nanobody' or isinstance(j['Lseq'], str))]
J = pd.DataFrame(jobs); J.to_csv(f'{TAB}/ib_jobs.csv', index=False)
order = rng.permutation(len(jobs))
for k in range(2):
    json.dump([jobs[i] for i in order[k::2]], open(f'{CACHE}/ib_jobs_{k}.json', 'w'))
print(J.groupby(['grp', 'mode']).size())
