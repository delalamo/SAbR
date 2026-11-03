"""PDB-wide census of ANARCI indel patterns at IMGT 66-75 (SEQRES-based; anarci_num.pkl covers 52k antibody-like SEQRES
sequences mapped to PDB chains via cand_seqs.json). For each pattern carried by a B error unit, list the high-resolution
X-ray entries (<2.6 A) with the SAME chain type and SAME pattern (distinct V sequences first), as candidates for SAbR.
Output: tables/v04_pdbwide_patterns.csv (chain-level), tables/v04_hires_candidates.json"""
import pickle, json, numpy as np, pandas as pd
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/case_files_ABCF_FR3START_INDEL_B_confounding'
AN = pickle.load(open(f'{SP}/deeper/cache/seq_novelty/anarci_num.pkl', 'rb'))
CS = json.load(open(f'{SP}/deeper/cache/seq_novelty/cand_seqs.json'))
meth, res, dep = {}, {}, {}
for line in open('/tmp/supp/pdbwide/entries.idx', errors='replace'):
    f = line.rstrip('\n').split('\t')
    if len(f) < 8 or len(f[0]) != 4: continue
    p = f[0].lower(); meth[p] = f[7].strip() or 'X-RAY DIFFRACTION'
    try: res[p] = float(f[6].split(',')[0])
    except: res[p] = np.nan
    dep[p] = f[2]
rows = []
for seq, doms in AN.items():
    for d in doms or []:
        ct = d[0]
        if ct not in 'HKL': continue
        nm = [(int(p), str(i).strip(), a) for p, i, a in d[4]]
        pos = {p for p, i, a in nm if i == ''}
        if not pos or min(pos) > 60 or max(pos) < 85: continue
        miss = tuple(k for k in range(66, 76) if k not in pos and k != 73)
        ins = tuple(f'{p}{i}' for p, i, a in nm if 66 <= p <= 75 and i)
        if not (miss or ins): continue
        vseq = ''.join(a for p, i, a in nm)
        for pc in CS.get(seq, []):
            p, c = pc.split('_', 1); p = p.lower()
            rows.append(dict(pdb=p, chain=c, ct=ct, miss66=miss, ins66=ins, pattern=f'{ct}:{miss}+{ins}', vseq=vseq, seq=seq,
                             method=meth.get(p, '?'), res=res.get(p, np.nan), dep=dep.get(p)))
D = pd.DataFrame(rows)
D.to_pickle(f'{OUT}/tables/v04_pdbwide_patterns.pkl')
D.drop(columns=['seq']).to_csv(f'{OUT}/tables/v04_pdbwide_patterns.csv', index=False)
D['xr_hi'] = D.method.str.startswith('X-RAY') & (D.res < 2.6)
S = D.groupby('pattern').agg(chains=('pdb', 'size'), entries=('pdb', 'nunique'), distinct_V=('vseq', 'nunique'),
                             EM=('pdb', lambda s: s[D.loc[s.index, 'method'].str.contains('ELECTRON')].nunique()),
                             xr_hi=('pdb', lambda s: s[D.loc[s.index, 'xr_hi']].nunique()),
                             xr_hi_V=('vseq', lambda s: s[D.loc[s.index, 'xr_hi']].nunique())).sort_values('chains', ascending=False)
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 300)
print(S.head(60).to_string())
# patterns of the B error units and feature-carrying clean controls
U = pd.read_csv(f'{OUT}/tables/em_units_with_feature.csv')
UV = pd.read_pickle(f'{OUT}/tables/units_em.pkl').set_index('unit').vseq
U['vseq'] = U.unit.map(UV)
cand = {}
for r in U.itertuples():
    ct = {'H': 'H', 'VHH': 'H', 'K': 'K', 'L': 'L'}[r.cclass]
    pat = f'{ct}:{r.miss66}+{r.ins66}'
    sub = D[(D.pattern == pat) & D.xr_hi].sort_values('res')
    same_v = sub[sub.vseq.str[:100] == str(r.vseq)[:100]]
    picks = []
    for x in sub.drop_duplicates('vseq').itertuples():
        picks.append((x.pdb, x.chain, x.res, x.ct))
    cand[r.pdb + '_' + r.chain] = dict(pattern=pat, outcome=r.outcome, n_hires_entries=int(sub.pdb.nunique()),
                                     n_hires_V=int(sub.vseq.nunique()), picks=picks[:6])
    print(r.pdb, r.chain, r.outcome, pat, 'hires entries', sub.pdb.nunique(), 'distinct V', sub.vseq.nunique(), picks[:6])
json.dump(cand, open(f'{OUT}/tables/v04_hires_candidates.json', 'w'), indent=1, default=str)
