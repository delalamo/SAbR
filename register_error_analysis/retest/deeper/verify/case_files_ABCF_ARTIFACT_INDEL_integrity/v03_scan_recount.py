"""V03: recount the claim at scan level, independently of s10.
(1) units per mode (paper|seqkey) and artifact-explained units/runs;
(2) determinism inside the 3-5 A scan: every scan heavy chain whose SEQRES ANARCI pattern is one of the claimed
    patterns, with the window fully built, should carry a disagreement run in the window (and none for gap 81);
(3) other first-occurrence C/F runs lying in the DE window 79-86 or CDR1 27-35 that the rule leaves unexplained.
Output: tables/scan_pattern_chains.csv, printed tallies."""
import json, pickle, pandas as pd, numpy as np
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/case_files_ABCF_ARTIFACT_INDEL_integrity'
AN = pickle.load(open(f'{SP}/deeper/cache/seq_novelty/anarci_num.pkl', 'rb'))
C = pd.read_csv(f'{SP}/deeper/chains.csv.gz', low_memory=False)
amy = set(pd.read_csv('/home/user/SAbR/register_error_analysis/retest/deeper/pdb_meta/tables/amyloid_fibril_chains.csv').pdb.str.lower())
RF = pd.read_csv(f'{SP}/deeper/runs_first_occurrence.csv'); RA = pd.read_csv(f'{SP}/deeper/runs_all.csv')
print('first-occurrence runs', len(RF), 'after amyloid removal', (~RF.pdb.isin(amy)).sum())
RF = RF[~RF.pdb.isin(amy)].copy(); RF['unit'] = RF.paper.astype(str) + '|' + RF.seqkey.astype(str)
# SAbR per-residue records of the scan
S = {}
for fn in ['new', 'snapshot_rest', 'validation']:
    for l in open(f'{SP}/sabr/out/{fn}.jsonl'):
        x = json.loads(l)
        if x.get('status') == 'ok': S.setdefault((x['pdb'], x['chain']), x['residues'])
def pattern(seq):
    for d in AN.get(seq) or []:
        if d[0] == 'H':
            pos = {int(p) for p, i, a in d[4] if str(i).strip() == ''}
            ins = [p for p, i, a in d[4] if str(i).strip() != '' and 74 <= p <= 100]
            m = set(range(min(pos), 105)) - pos
            fr1 = tuple(sorted(k for k in m if 11 <= k <= 26)); fr3 = tuple(sorted(k for k in m if 74 <= k <= 100))
            return fr1, fr3, bool(ins)
    return None
PATS = {'FR3:84': ('fr3', (84,), (79, 86)), 'FR3:80+84': ('fr3', (80, 84), (79, 86)), 'FR3:84+85': ('fr3', (84, 85), (79, 87)),
        'FR1:26': ('fr1', (26,), (24, 37)), 'FR3:81 (control)': ('fr3', (81,), (79, 86)), 'FR3:85': ('fr3', (85,), (79, 86)),
        'FR3:96': ('fr3', (96,), (90, 98))}
H = C[(C.ctype == 'H') & ~C.pdb.isin(amy)].copy()
rows = []
for r in H.itertuples():
    p = pattern(r.seq)
    if p is None: continue
    fr1, fr3, ins = p
    for name, (reg, gaps, (lo, hi)) in PATS.items():
        cur = fr1 if reg == 'fr1' else fr3
        if cur != gaps or (reg == 'fr3' and ins): continue
        rec = S.get((r.pdb, r.chain))
        if rec is None: rows.append(dict(pattern=name, pdb=r.pdb, chain=r.chain, status='no SAbR record')); continue
        w = [x for x in rec if x[4] is not None and lo <= x[4] <= hi]
        si = [x[3] for x in w]
        built = len(w) > 0 and (max(si) - min(si) + 1 == len(si)) and min(x[4] for x in w) <= lo + 1 and max(x[4] for x in w) >= hi - 1
        dis = [x for x in w if (x[4], x[5] or '') != (x[6], x[7] or '')]
        ra = RA[(RA.pdb == r.pdb) & (RA.chain == r.chain) & (RA.imgt0 <= hi) & (RA.imgt1 >= lo)]
        rf = RF[(RF.pdb == r.pdb) & (RF.chain == r.chain) & (RF.imgt0 <= hi) & (RF.imgt1 >= lo)]
        rows.append(dict(pattern=name, pdb=r.pdb, chain=r.chain, method=r.method, res=r.res, year=r.year, v_gene=r.v_gene, v_id=round(r.v_id, 2),
                         species=r.species, built=built, n_dis=len(dis), dis=' '.join(f'{x[4]}/{x[6]}' for x in dis),
                         run_any=';'.join(f'{a}-{b}({o[:2]})' for a, b, o in zip(ra.imgt0, ra.imgt1, ra.origin)),
                         run_first=';'.join(f'{a}-{b}' for a, b in zip(rf.imgt0, rf.imgt1)), err_first=r.err_first))
P = pd.DataFrame(rows); P.to_csv(f'{OUT}/tables/scan_pattern_chains.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200)
print(P.drop(columns=['v_id']).to_string())
print('\nDeterminism among fully built windows (chain has >=3 disagreeing residues in window):')
B = P[P.built == True]
print(B.groupby('pattern').agg(chains=('pdb', 'size'), entries=('pdb', 'nunique'), dis3=('n_dis', lambda s: int((s >= 3).sum()))).to_string())

# (1) unit recount with an independent artifact rule: run window inside the pattern window and chain carries the pattern
A = P[P.pattern.isin(['FR3:84', 'FR3:80+84', 'FR3:84+85', 'FR1:26'])][['pdb', 'chain', 'pattern']]
win = {'FR3:84': (79, 85), 'FR3:80+84': (79, 85), 'FR3:84+85': (79, 86), 'FR1:26': (24, 37)}
M = RF.merge(A, on=['pdb', 'chain'], how='left')
M['artifact'] = [isinstance(p, str) and win[p][0] <= a and b <= win[p][1] for p, a, b in zip(M.pattern, M.imgt0, M.imgt1)]
T = M.groupby('mode').agg(units=('unit', 'nunique'), runs=('unit', 'size'),
                          art_units=('unit', lambda u: M.loc[u.index][M.loc[u.index].artifact].unit.nunique()),
                          art_runs=('artifact', 'sum'),
                          xray_units=('unit', lambda u: M.loc[u.index][M.loc[u.index].method == 'X-ray'].unit.nunique()),
                          art_xray_units=('unit', lambda u: M.loc[u.index][M.loc[u.index].artifact & (M.loc[u.index].method == 'X-ray')].unit.nunique()))
print('\nUnit recount:\n', T.to_string())
print(M[M.artifact].drop_duplicates('unit')[['pdb', 'chain', 'mode', 'imgt0', 'imgt1', 'delta', 'method', 'last_author', 'pattern']].to_string())
# (3) first-occurrence C runs inside the DE window that are not flagged
de = M[(M['mode'] == 'C_FR3') & (M.imgt0 >= 79) & (M.imgt1 <= 86) & ~M.artifact]
print('\nUnflagged first-occurrence C runs inside IMGT 79-86:\n', de[['pdb', 'chain', 'imgt0', 'imgt1', 'delta', 'method', 'res', 'last_author', 'unit']].to_string())
