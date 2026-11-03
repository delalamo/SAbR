"""Build per-chain local-sequence features (CDR3/FR4) for every V chain, and the D+/D-/E error-chain table.

Outputs (scratch cache): chains_local.pkl ; tables/error_chains_DE.csv
CDR3 = ANARCI IMGT 105-117, FR4 = 118-128 (from the SAbR/ANARCI per-residue records).
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, '/home/user/SAbR/register_error_analysis/retest/deeper/case_files_DE/scripts')
from common import *

C = chains(); R = runs(); RA = runs_all()
rec = sabr_records()
rows = []
for k, res in rec.items():
    v = [x for x in res if x[4] is not None]
    if not v: continue
    cdr3 = ''.join(three for three in [])
    aa3 = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}
    c3 = ''.join(aa3.get(x[2], 'X') for x in v if 105 <= x[4] <= 117)
    f4 = ''.join(aa3.get(x[2], 'X') for x in v if 118 <= x[4] <= 128)
    a118 = [x for x in v if x[4] == 118 and not x[5]]
    a117 = [x for x in v if x[4] == 117 and not x[5]]
    a116 = [x for x in v if x[4] == 116 and not x[5]]
    a119 = [x for x in v if x[4] == 119 and not x[5]]
    # built residues in model: seq_index gaps between consecutive residues (unbuilt) in 100-128 window
    idx = [(x[3], x[4]) for x in v]
    gaps_cdr3 = 0
    for (i0, n0), (i1, n1) in zip(idx, idx[1:]):
        if n0 is not None and 100 <= n0 <= 128 and i1 - i0 > 1: gaps_cdr3 += i1 - i0 - 1
    last = max(x[4] for x in v)
    rows.append(dict(pdb=k[0], chain=k[1], cdr3_seq=c3, cdr3_n=len(c3), fr4_seq=f4,
                     aa116=aa3.get(a116[0][2], 'X') if a116 else '-', aa117=aa3.get(a117[0][2], 'X') if a117 else '-',
                     aa118=aa3.get(a118[0][2], 'X') if a118 else '-', aa119=aa3.get(a119[0][2], 'X') if a119 else '-',
                     unbuilt_100_128=gaps_cdr3, last_imgt=last))
L = pd.DataFrame(rows)
L.to_pickle(f'{CACHE}/chains_local.pkl')
X = C.merge(L, on=['pdb', 'chain'], how='left')
E = R[R['mode'].isin(MODES)].groupby(['pdb', 'chain', 'mode']).agg(imgt0=('imgt0', 'min'), imgt1=('imgt1', 'max'),
        delta=('delta', 'first'), nruns=('imgt0', 'size')).reset_index()
E = E.merge(X, on=['pdb', 'chain'], how='left')
# does the partner chain in the same entry carry any first-occurrence error?
errpdbchains = R.groupby('pdb').chain.apply(set).to_dict()
modes_by_chain = R.groupby(['pdb', 'chain'])['mode'].apply(lambda s: ';'.join(sorted(set(s)))).to_dict()
E['other_err_chains_in_entry'] = [';'.join(sorted(f'{c}:{modes_by_chain[(p, c)]}' for c in errpdbchains.get(p, set()) - {ch}))[:200] for p, ch in zip(E.pdb, E.chain)]
E.to_csv(f'{OUT}/tables/error_chains_DE.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 300); pd.set_option('display.max_colwidth', 30)
print(E[['pdb', 'chain', 'mode', 'ctype', 'method', 'year', 'imgt0', 'imgt1', 'delta', 'cdr3_n', 'cdr3_seq', 'fr4_seq', 'aa117', 'aa118', 'unbuilt_100_128', 'j_gene', 'species', 'vhh', 'other_err_chains_in_entry']].drop_duplicates(['pdb', 'mode', 'ctype']).to_string())
