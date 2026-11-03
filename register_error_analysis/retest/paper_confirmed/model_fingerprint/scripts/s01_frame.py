"""Chain-level analysis frame for the model-fingerprint re-test.

Joins deeper/chains.csv.gz + paperconf/chain_flags.csv.gz + pdb_meta/tables/chain_cis_sheet.csv.gz (wwPDB cis-peptide
and PROMOTIF sheet annotation mapped to ANARCI IMGT positions, all 15,752 non-amyloid V chains, EM and X-ray).

Groups (column grp):
  paper_strict  chain carries a Table S1 error confirmed by a SAbR/ANARCI run, first occurrence   (PRIMARY CASES)
  paper_loose   the 1 loose-ledger-doubt case (paper_case_first minus strict)
  paper_copy    Table S1 error confirmed, but the run is a copy / near copy                     (added for SENSITIVITY)
  rest_case     deeper first-occurrence error chain (err_first) that is none of the above       (HETEROGENEITY comparator)
  control       paper_control: in window, entry not in Table S1, no SAbR/ANARCI run on chain    (PRIMARY CONTROLS)
  clean_other   no run of any origin (err_any==0) but not a paper control (outside window, or in a Table S1 entry)
  other         remaining chains (copied/artifact-only runs, etc.)
For paper_copy chains (err_first False) the sheet-away fraction is recomputed excluding runs_all positions +-5, because
chain_cis_sheet only excluded first-occurrence runs.
Output: tables/chain_frame.csv.gz"""
import json, collections, sys, numpy as np, pandas as pd
sys.path.insert(0, '/home/user/SAbR/register_error_analysis/retest/paper_confirmed/model_fingerprint/scripts')
from common import SP, DEEP, TAB, ybin, ybin_win

C = pd.read_csv(f'{SP}/deeper/chains.csv.gz', dtype={'chain': str})
F = pd.read_csv(f'{SP}/paperconf/chain_flags.csv.gz', dtype={'chain': str})
S = pd.read_csv(f'{SP}/paperconf/paper_errors_annotated.csv', dtype={'chain': str})
CS = pd.read_csv(f'{DEEP}/pdb_meta/tables/chain_cis_sheet.csv.gz', dtype={'chain': str})
C = C.drop(columns=['seq']).merge(F, on=['pdb', 'chain'], how='left')
C['amyloid'] = C.title.fillna('').str.contains(r'amyloid(?!-beta)', case=False)
keep = ['pdb', 'chain', 'v_sheet_away', 'const_sheet', 'n_const_modeled', 'v_cis_n', 'const_cis_n', 'v_cis_pos',
        'v_pro_pos', 'const_pro_n', 'const_cis_pos']
C = C.merge(CS[keep], on=['pdb', 'chain'], how='left')
C['err_first'] = C.err_first.astype(bool); C['err_any'] = C.err_any.astype(bool)

g = np.full(len(C), 'other', dtype=object)
g[(C.err_any == False).values] = 'clean_other'
g[C.paper_control.values] = 'control'
g[(C.err_first & ~C.paper_case_incl_copies & ~C.paper_case_first).values] = 'rest_case'
g[(C.paper_case_incl_copies & ~C.paper_case_first).values] = 'paper_copy'
g[(C.paper_case_first & ~C.paper_case_strict).values] = 'paper_loose'
g[C.paper_case_strict.values] = 'paper_strict'
C['grp'] = g
C['rest_in_window'] = (C.grp == 'rest_case') & C.in_paper_window

# modes: deeper first-occurrence modes + the mode of the run that confirms the Table S1 error (covers copies)
pm = S[S.status.isin(['confirmed first occurrence', 'confirmed, loose-ledger doubt', 'copy of an earlier model',
                      'near copy (ledger)'])].groupby(['pdb', 'chain'])['mode'].apply(lambda s: ';'.join(sorted(set(s.dropna()))))
C = C.merge(pm.rename('paper_modes').reset_index(), on=['pdb', 'chain'], how='left')
allm = C.err_modes.fillna('') + ';' + C.paper_modes.fillna('')
C['modeA'] = allm.str.contains('A_FR1') & (C.grp != 'control') & (C.grp != 'clean_other')
C['modeDE'] = allm.str.contains('Dplus|Dminus|E_FR4') & (C.grp != 'control') & (C.grp != 'clean_other')
C['cluster'] = C.paper.fillna(C.pdb)
C['ybin'] = ybin(C.year.values); C['ybin_w'] = ybin_win(C.year.values)

pro = C.v_pro_pos.fillna('').str.split(';'); cisp = C.v_cis_pos.fillna('').str.split(';')
C['pro8'] = [('8' in a) for a in pro]; C['cis8'] = [('8PRO' in a) for a in cisp]
C['pro115'] = [('115' in a) for a in pro]; C['cis115'] = [('115PRO' in a) for a in cisp]
C['const_cis_any'] = (C.const_cis_n > 0).astype(float).where((C.n_const_modeled > 60) & (C.const_pro_n > 0))
C['has_cs'] = C.v_cis_pos.notna() | C.v_pro_pos.notna() | C.v_sheet_away.notna()

# ---- recompute sheet-away for paper_copy chains (exclude runs_all +-5, as 08_cis_sheet.py did for first-occurrence runs)
RA = pd.read_csv(f'{SP}/deeper/runs_all.csv', dtype={'chain': str})
cp = set(zip(C[C.grp == 'paper_copy'].pdb, C[C.grp == 'paper_copy'].chain))
rs = collections.defaultdict(list)
for r in RA.itertuples():
    if (r.pdb, r.chain) in cp: rs[(r.pdb, r.chain)].append((r.i0, r.i1))
imgt = {}
for fn in ['new', 'snapshot_rest', 'validation']:
    for l in open(f'{SP}/sabr/out/{fn}.jsonl'):
        if l[9:13] not in {p for p, _ in cp}: continue
        r = json.loads(l)
        k = (r['pdb'], r['chain'])
        if k in cp and k not in imgt:
            imgt[k] = {x[3]: x[4] for x in r.get('residues') or [] if x[3] is not None}
feat = {}
for l in open(f'{SP}/deeper/cache/pdb_meta/cis_sheet.jsonl'):
    x = json.loads(l); k = (x['rcsb_id'].split('.')[0].lower(), x['auth'])
    if k in cp: feat[k] = x['feats']
new = {}
for k in cp:
    m = imgt.get(k); fs = feat.get(k)
    if m is None or fs is None: continue
    sheet = set()
    for f in fs:
        if f['type'] == 'SHEET':
            for p in f['feature_positions']:
                sheet.update(range(p['beg_seq_id'] - 1, (p.get('end_seq_id') or p['beg_seq_id'])))
    near = set()
    for i0, i1 in rs.get(k, []): near.update(range(int(i0) - 5, int(i1) + 6))
    vres = [i for i, a in m.items() if a is not None and i not in near]
    new[k] = np.mean([i in sheet for i in vres]) if len(vres) > 40 else np.nan
C['v_sheet_away'] = [new.get(k, v) for k, v in zip(zip(C.pdb, C.chain), C.v_sheet_away)]
print('sheet-away recomputed for', len(new), 'paper_copy chains')

C.drop(columns=['title', 'chain_desc']).to_csv(f'{TAB}/chain_frame.csv.gz', index=False)
na = C[~C.amyloid]
print(na.groupby(['grp', 'method']).agg(chains=('pdb', 'size'), entries=('pdb', 'nunique'), papers=('cluster', 'nunique'),
                                         with_cis_sheet=('has_cs', 'sum')).to_string())
print('rest_case in window:', int(na.rest_in_window.sum()))
