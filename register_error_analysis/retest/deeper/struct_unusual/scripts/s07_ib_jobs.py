"""Build the ImmuneBuilder job list (the run used ib_jobs_trim.json: same list with a random 60 of the 108 case Fvs, to save compute): control Fvs (random 120), all control VHH, then case Fvs/VHH (re-run only to
obtain ensemble error estimates; model 0 unrefined is deterministic and matches the existing case predictions)."""
import json, numpy as np, pandas as pd
from common import SP, C, TAB
d = pd.read_csv(f'{SP}/deeper/chains.csv.gz', dtype={'chain': str}, usecols=['pdb', 'chain', 'vseq'])
seq = dict(zip(zip(d.pdb, d.chain), d.vseq))
fv = pd.read_csv(f'{TAB}/fv_abangle.csv', dtype={'H': str, 'L': str})
s = pd.read_csv(f'{TAB}/sample.csv', dtype={'chain': str})
rng = np.random.default_rng(7)
jobs = []
ctrl = fv[~fv.fv_case & ~fv.H_any & ~fv.L_any]
ctrl = ctrl.iloc[rng.permutation(len(ctrl))[:120]]
for _, f in ctrl.iterrows():
    jobs.append(dict(fv_id=f'c_{f.pdb}_{f.H}_{f.L}', Hseq=seq[(f.pdb, f.H)], Lseq=seq[(f.pdb, f.L)], mode='fv', grp='control'))
for _, u in s[(s.role_cc == 'control') & (s.cclass == 'VHH')].iterrows():
    jobs.append(dict(fv_id=f'c_{u.pdb}_{u.chain}', Hseq=seq[(u.pdb, u.chain)], Lseq='', mode='nanobody', grp='control'))
for _, u in s[(s.role_cc == 'case') & (s.cclass == 'VHH')].iterrows():
    jobs.append(dict(fv_id=f'x_{u.pdb}_{u.chain}', Hseq=seq[(u.pdb, u.chain)], Lseq='', mode='nanobody', grp='case'))
for _, f in fv[fv.fv_case].iterrows():
    jobs.append(dict(fv_id=f'x_{f.pdb}_{f.H}_{f.L}', Hseq=seq[(f.pdb, f.H)], Lseq=seq[(f.pdb, f.L)], mode='fv', grp='case'))
jobs = [j for j in jobs if isinstance(j['Hseq'], str) and (j['mode'] == 'nanobody' or isinstance(j['Lseq'], str))]
json.dump(jobs, open(f'{C}/ib_jobs.json', 'w'))
print(pd.DataFrame(jobs).groupby(['grp', 'mode']).size())
