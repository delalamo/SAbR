"""PDB-wide census of the claimed artifact indel patterns (verification of ARTIFACT_INDEL).

For every heavy-chain V domain in the PDB-wide ANARCI cache (seq_novelty/anarci_num.pkl, 52k SEQRES sequences mapped
to PDB chains via cand_seqs.json), record the ANARCI-absent IMGT positions in FR3 (74-100) and FR1 (11-26), exactly as
s10_artifact_summary.py does. Then list, per pattern, the PDB entries/chains carrying it with method and resolution,
and the number of DISTINCT V sequences (antibodies) among high-resolution X-ray entries.
Outputs: tables/v01_pattern_census.csv (chain-level), tables/v01_pattern_summary.csv
"""
import pickle, json, re, collections, pandas as pd, numpy as np
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/case_files_ABCF_ARTIFACT_INDEL_confounding'
AN = pickle.load(open(f'{SP}/deeper/cache/seq_novelty/anarci_num.pkl', 'rb'))
CS = json.load(open(f'{SP}/deeper/cache/seq_novelty/cand_seqs.json'))
# entry method/resolution from entries.idx
meth, res = {}, {}
for line in open('/tmp/supp/pdbwide/entries.idx', errors='replace'):
    f = line.rstrip('\n').split('\t')
    if len(f) < 8 or len(f[0]) != 4: continue
    p = f[0].lower()
    m = f[7].strip() or 'X-RAY DIFFRACTION'
    meth[p] = m
    try: res[p] = float(f[6].split(',')[0])
    except: res[p] = np.nan
PATS = {'FR3_84': ('FR3', {84}), 'FR3_84_85': ('FR3', {84, 85}), 'FR3_80_84': ('FR3', {80, 84}),
        'FR1_26': ('FR1', {26}), 'FR3_81': ('FR3', {81}), 'FR3_96': ('FR3', {96}), 'FR3_82': ('FR3', {82}), 'FR3_85': ('FR3', {85})}
rows = []
for seq, doms in AN.items():
    if not doms: continue
    for d in doms:
        if d[0] != 'H': continue
        pos = {int(p) for p, i, a in d[4] if str(i).strip() == ''}
        if not pos: continue
        m = set(range(min(pos), 105)) - pos
        fr3 = frozenset(k for k in m if 74 <= k <= 100); fr1 = frozenset(k for k in m if 11 <= k <= 26)
        vseq = ''.join(a for p, i, a in d[4] if a != '-')
        hit = [n for n, (r, g) in PATS.items() if (fr3 if r == 'FR3' else fr1) == g]
        if not hit: continue
        for pc in CS.get(seq, []):
            p, c = pc.split('_', 1)
            rows.append(dict(pattern=hit[0], pdb=p.lower(), chain=c, method=meth.get(p.lower(), '?'), res=res.get(p.lower(), np.nan),
                             vseq=vseq, first_imgt=min(pos)))
        break
D = pd.DataFrame(rows)
D.to_csv(f'{OUT}/tables/v01_pattern_census.csv', index=False)
D['xr_hi'] = D.method.str.startswith('X-RAY') & (D.res < 2.6)
S = D.groupby('pattern').agg(chains=('pdb', 'size'), entries=('pdb', 'nunique'), distinct_V=('vseq', 'nunique'),
                             EM_entries=('pdb', lambda s: s[D.loc[s.index, 'method'].str.contains('ELECTRON')].nunique()),
                             xr_hi_entries=('pdb', lambda s: s[D.loc[s.index, 'xr_hi']].nunique()),
                             xr_hi_distinct_V=('vseq', lambda s: s[D.loc[s.index, 'xr_hi']].nunique()))
print(S)
S.to_csv(f'{OUT}/tables/v01_pattern_summary.csv')
