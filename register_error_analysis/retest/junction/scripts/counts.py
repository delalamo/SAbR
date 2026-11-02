"""How many CDR3/FR4 errors are there? (a) AlphaFold-independent 3-5 A scan; (b) the 2024 study's Table S1."""
import pandas as pd, numpy as np, json
from anarci import anarci
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
R = pd.read_csv(f'{SP}/unexpl/runs_origin_v2.csv'); M = pd.read_pickle(f'{SP}/unexpl/master4.pkl')
R['cls'] = R.origin.str.split(' ').str[0]
for name, sel in [('CDR3-FR4 junction (W/F118 displaced)', R.cls == '5'), ('FR4 bulge after W/F-G (IMGT 120)', R.cls == '6'),
                  ('any run touching IMGT 105-128 (all origins)', (R.imgt1 >= 105))]:
    s = R[sel]
    print(f'{name}: {len(s)} runs, {s[["pdb","chain"]].drop_duplicates().shape[0]} chains, {s.pdb.nunique()} entries, '
          f'{s.drop_duplicates(["paper","seqkey"]).shape[0]} antibody-in-paper events; EM {100*(s.method=="EM").mean():.0f}%')
print('scan denominator:', len(M), 'single-domain V chains,', M.pdb.nunique(), 'entries')
# 2024 study (Table S1) antibody errors mapped to IMGT
E = pd.read_csv('/home/user/SAbR/register_error_analysis/template_propagation/tables/first_instance_errors.csv')
em = pd.read_pickle('/tmp/tmpl/errors_seqmapped.pkl')[['PDB_CHAIN_ID', 'RESIDUE_RANGE', 'seq']]
E['PDB_CHAIN_ID'] = E.pdb + ':' + E.chain
E = E.merge(em, left_on=['PDB_CHAIN_ID', 'residue_range'], right_on=['PDB_CHAIN_ID', 'RESIDUE_RANGE'], how='left')
reg = []
for r in E.itertuples():
    if pd.isna(r.s0) or not isinstance(r.seq, str): reg.append('unmapped'); continue
    n, a, _ = anarci([('q', r.seq)], scheme='imgt', assign_germline=False)
    if not n[0]: reg.append('no V'); continue
    lab = {}; i = a[0][0]['query_start']
    for (p, ic), aa in n[0][0][0]:
        if aa != '-': lab[i] = p; i += 1
    pos = [lab[k] for k in range(int(r.s0), int(r.s1) + 1) if k in lab]
    if not pos: reg.append('constant domain'); continue
    lo, hi = min(pos), max(pos)
    reg.append('CDR3/FR4 (IMGT >=105)' if hi >= 105 else 'FR1-FR3/CDR1-2')
E['region'] = reg
print('\n2024 study (Table S1) antibody errors by region:'); print(E.region.value_counts().to_string())
print('entries with a CDR3/FR4 error:', E[E.region.str.startswith('CDR3')].pdb.nunique(), 'of', E.pdb.nunique())
