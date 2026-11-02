# Shared unit construction for lineage.py / lineage_perm.py (exec'd; expects SP defined).
# Builds U: one row per antibody-in-paper (paper x V-region sequence) with germline V/J (species|chain type|gene), lab
# (last author), deposition/release, EM flag, period, and y_<class> flags; R: classified runs.
import json, numpy as np, pandas as pd
M = pd.read_pickle(f'{SP}/unexpl/master4.pkl'); G = json.load(open(f'{SP}/unexpl/germline.json'))
# master4.seq is the full chain SEQRES (V + constant). Antibody identity and similarity must use the V region only.
import os
VF = f'{SP}/unexpl/vregion.json'
if os.path.exists(VF): VR = json.load(open(VF))
else:
    from anarci import run_anarci
    VR = {}; useq = sorted(set(M.seq))
    for k in range(0, len(useq), 1000):
        chunk = [(str(i), q) for i, q in enumerate(useq[k:k + 1000])]
        _, _, det, _ = run_anarci(chunk, scheme='imgt', ncpu=4)
        for (n, q), d in zip(chunk, det):
            if d: VR[q] = q[d[0]['query_start']:d[0]['query_end'] + 1]
    json.dump(VR, open(VF, 'w'))
M['vseq_full'] = M.seq; M['seq'] = M.seq.map(VR).fillna(M.seq)
M['j_gene'] = M.vseq_full.map(lambda s: (G.get(s) or {}).get('j_gene'))
gene = lambda g: g.split('*')[0] if isinstance(g, str) else None
M['V'] = M.species.fillna('?') + '|' + M.ctype + '|' + M.v_gene.map(gene).fillna('?')
M['J'] = M.species.fillna('?') + '|' + M.ctype + '|' + M.j_gene.map(gene).fillna('?')
M['lab'] = M.last_author.fillna('').str.lower().str.replace('[^a-z]', '', regex=True)
M['paper'] = M.cite_key.fillna(M.pdb)
M['dep'] = pd.to_datetime(M.deposit); M['rel'] = pd.to_datetime(M.release)
M['EM'] = M.method.eq('EM'); M['period'] = pd.cut(M.year, [0, 2017, 2019, 2021, 2030], labels=['<=2017', '2018-19', '2020-21', '>=2022']).astype(str)
R = pd.read_csv(f'{SP}/unexpl/runs_origin_v2.csv')
R['cls'] = R.origin.str.split(' ').str[0].map({'5': 'junction', '6': 'bulge', '7': 'cdr12', '8': 'cdr12', '3': 'gap'})
R = R[R.cls.notna()].copy()
W = pd.read_csv(f'{SP}/unexpl/w118_burial.csv')
wout = set((p, c) for p, c, l, n in zip(W.pdb, W.chain, W.label, W.ca_within_10A) if l.startswith('CDR3') and n < 14)
J2 = R[(R.cls == 'junction').values & np.array([k in wout for k in zip(R.pdb, R.chain)], dtype=bool)].copy(); J2['cls'] = 'junction_w118out'
R = pd.concat([R, J2], ignore_index=True)
U = M.groupby(['paper', 'seq']).agg(pdbs=('pdb', lambda x: sorted(set(x))), dep=('dep', 'min'), rel=('rel', 'min'), lab=('lab', 'first'),
                                     V=('V', 'first'), J=('J', 'first'), ctype=('ctype', 'first'), EM=('EM', 'any'), period=('period', 'first'),
                                     year=('year', 'min')).reset_index()
key = dict(zip(zip(M.pdb, M.chain), zip(M.paper, M.seq)))
R['unit'] = [key.get(k) for k in zip(R.pdb, R.chain)]; R = R[R.unit.notna()]
uidx = {u: i for i, u in enumerate(zip(U.paper, U.seq))}
for c in ['junction', 'junction_w118out', 'bulge', 'cdr12', 'gap']:
    U[f'y_{c}'] = False; U.loc[[uidx[u] for u in R[R.cls == c].unit], f'y_{c}'] = True
print('units', len(U), {c: int(U[f'y_{c}'].sum()) for c in ['junction', 'junction_w118out', 'bulge', 'cdr12', 'gap']}, flush=True)
