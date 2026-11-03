"""Label run residues using the project's own run table (runs_all.csv: i0..i1), checking whether i0/i1 index the
SAbR record list or seq_index. Adds columns pinrun, pm3, pm10, pchain_run to res.pkl -> res2.pkl"""
import json, numpy as np, pandas as pd
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
OUTC = f'{SP}/deeper/cache/verify_struct_unusual_rest'
S = pd.read_csv('/home/user/SAbR/register_error_analysis/retest/deeper/struct_unusual/tables/sample.csv', dtype={'chain': str})
P = set(S.pdb)
ru = pd.read_csv(f'{SP}/deeper/runs_all.csv', dtype={'chain': str})
ru = ru[ru.pdb.isin(P)]
keys = set(zip(ru.pdb, ru.chain))
sab = {}
for fn in ['validation', 'snapshot_rest', 'new']:
    with open(f'{SP}/sabr/out/{fn}.jsonl') as f:
        for line in f:
            if line[9:13] in P:
                r = json.loads(line)
                if (r['pdb'], r['chain']) in keys and 'residues' in r:
                    sab[(r['pdb'], r['chain'])] = r['residues']
ok_idx = ok_si = 0
lab = {}
for r in ru.itertuples():
    res = sab[(r.pdb, r.chain)]
    dis = lambda z: not (z[4] == z[6] and (z[5] or '') == (z[7] or ''))
    a = [dis(res[k]) for k in range(int(r.i0), int(r.i1) + 1) if k < len(res)]
    si = {z[3]: z for z in res}
    b = [dis(si[k]) for k in range(int(r.i0), int(r.i1) + 1) if k in si]
    ok_idx += np.mean(a) if a else 0; ok_si += np.mean(b) if b else 0
    n = len(res)
    for k in range(n):
        z = res[k]; key = (r.pdb, r.chain, z[0], z[1] or '')
        cur = lab.get(key, [False, False, False])
        dist = 0 if r.i0 <= k <= r.i1 else min(abs(k - r.i0), abs(k - r.i1))
        lab[key] = [cur[0] or dist == 0, cur[1] or dist <= 3, cur[2] or dist <= 10]
print('runs', len(ru), 'mean disagreement fraction if i0/i1 = record index', round(ok_idx / len(ru), 3), '; if seq_index', round(ok_si / len(ru), 3))
R = pd.read_pickle(f'{OUTC}/res.pkl')
L = [lab.get((p, c, n, i), [False, False, False]) for p, c, n, i in zip(R.pdb, R.chain, R.resnum, R.ic)]
R[['pinrun', 'pm3', 'pm10']] = pd.DataFrame(L, index=R.index)
R['pchain_run'] = [(p, c) in keys for p, c in zip(R.pdb, R.chain)]
R['pchain_run_first'] = False
rf = pd.read_csv(f'{SP}/deeper/runs_first_occurrence.csv', dtype={'chain': str})
kf = set(zip(rf.pdb, rf.chain))
R['pchain_run_first'] = [(p, c) in kf for p, c in zip(R.pdb, R.chain)]
R.to_pickle(f'{OUTC}/res2.pkl')
print(R[R.isV].groupby(R.pdb.isin(set(S[S.role_cc == 'case'].pdb)))[['pinrun', 'pm3', 'pm10', 'pchain_run']].mean())
