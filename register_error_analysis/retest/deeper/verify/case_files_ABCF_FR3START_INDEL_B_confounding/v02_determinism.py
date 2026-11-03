"""Is an ANARCI indel at IMGT 66-75 a deterministic SAbR-vs-ANARCI numbering disagreement (artifact), independent of model quality?
For every chain SAbR renumbered (new + snapshot_rest + validation; all 3-5 A scan chains, de-duplicated), take the ANARCI
numbering of the modelled residues, flag an indel at 66-75 (missing positions other than 73, or insertion codes), and count residues in
IMGT 55-80 where SAbR != ANARCI. Also record whether the window 60-78 is fully modelled (no unbuilt residues).
Output: tables/determinism_by_set.csv, tables/feature_chains_all_sets.csv"""
import json, numpy as np, pandas as pd
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/case_files_ABCF_FR3START_INDEL_B_confounding'
C = pd.read_pickle(f'{OUT}/tables/chain_features.pkl')
meta = C.drop_duplicates(['pdb', 'chain']).set_index(['pdb', 'chain'])
rows = []
for fn in ['validation', 'new', 'snapshot_rest']:
    for line in open(f'{SP}/sabr/out/{fn}.jsonl'):
        x = json.loads(line); rec = x.get('residues')
        if not rec or x.get('status') != 'ok': continue
        ct = x.get('chain_type') or 'H'
        an = [r for r in rec if r[4] is not None]
        if not an: continue
        pos = {r[4] for r in an if not r[5]}
        if not (any(p <= 62 for p in pos) and any(p >= 80 for p in pos)): continue
        miss = tuple(k for k in range(66, 76) if k not in pos and k != 73)
        ins = tuple(f'{r[4]}{r[5]}' for r in an if 66 <= r[4] <= 75 and r[5])
        w = [r for r in an if 60 <= r[4] <= 78]
        si = [r[3] for r in w]
        contiguous = len(si) > 0 and (max(si) - min(si) + 1 == len(si))
        dis = [r for r in an if 55 <= r[4] <= 80 and (r[4], r[5] or '') != (r[6], (r[7] or '') if r[6] is not None else '')]
        rows.append(dict(set=fn, pdb=x['pdb'], chain=x['chain'], ct=ct, feat=int(bool(miss) or bool(ins)), miss66=miss, ins66=ins,
                         contiguous=contiguous, n_dis_55_80=len(dis), dis3=int(len(dis) >= 3),
                         dis_detail=' '.join(f'{r[2][:3]}{r[4]}{r[5]}/{r[6]}{r[7] or ""}' for r in dis)[:200]))
D = pd.DataFrame(rows)
D['method'] = [meta.method.get((p, c), 'X-ray' if s == 'validation' else None) for p, c, s in zip(D.pdb, D.chain, D.set)]
D['res'] = [meta.res.get((p, c), np.nan) for p, c in zip(D.pdb, D.chain)]
D['err_first'] = [meta.err_first.get((p, c), np.nan) for p, c in zip(D.pdb, D.chain)]
D['B_first'] = [meta.B_first.get((p, c), np.nan) for p, c in zip(D.pdb, D.chain)]
# NOTE: validation.jsonl is NOT a <3 A set (most of its entries are 3-5 A scan entries); group by the chains.csv method instead and
# de-duplicate (pdb, chain), preferring the scan files. Chains absent from chains.csv are dropped.
D['meth'] = [meta.method.get((p, c)) for p, c in zip(D.pdb, D.chain)]
D = D[D.meth.notna()].sort_values('set', key=lambda s: s.map({'new': 0, 'snapshot_rest': 1, 'validation': 2})).drop_duplicates(['pdb', 'chain'])
D['grp'] = np.where(D.meth == 'EM', 'EM 3-5 A', 'X-ray 3-5 A')
G = D.groupby(['grp', 'feat']).agg(n=('dis3', 'size'), dis3=('dis3', 'sum'), rate=('dis3', 'mean')).reset_index()
print(G.to_string())
G.to_csv(f'{OUT}/tables/determinism_by_set.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 120)
F = D[D.feat == 1].sort_values(['grp', 'pdb'])
print(F[['grp', 'pdb', 'chain', 'ct', 'res', 'miss66', 'ins66', 'contiguous', 'n_dis_55_80', 'B_first', 'dis_detail']].to_string())
F.to_csv(f'{OUT}/tables/feature_chains_all_sets.csv', index=False)
D.to_pickle(f'{OUT}/tables/sabr_window_all.pkl')
