"""Context of each case and control: the same antibody elsewhere in the PDB, the same antigen at the same time,
and the antibody's germline / sequence novelty.

  * same antibody: every PDB chain whose SEQRES contains the case V region (exact match of the V region, or of
    its first 95 residues when the SEQRES is truncated) -> entries, dates, last author, resolution, method; before/after
    the case; same or other lab; and whether those entries are in the 3-5 A scan with or without an error
  * same antigen: PDB-wide antibody entries (../antigen_competition) sharing a UniProt accession with the case,
    deposited within +-6 months; how many labs; how many of them carry a first-occurrence error in the 3-5 A scan;
    whether the case was the first antibody structure of that antigen
  * germline / novelty (../seq_novelty chain_novelty): V/J gene, identity, nearest prior V excluding the same
    antibody, prior entries with the V gene, CDR lengths, rare residues in the mode window, hallmarks, engineering
Output: tables/context.csv
"""
import json, gzip, re, numpy as np, pandas as pd
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/case_files_ABCF'
DE = '/home/user/SAbR/register_error_analysis/retest/deeper'
L = pd.read_csv(f'{OUT}/tables/local_sites.csv')
C = pd.read_csv(f'{SP}/deeper/chains.csv.gz', low_memory=False)
E = pd.read_csv(f'{SP}/deeper/entries.csv')
A = pd.read_csv(f'{DE}/antigen_competition/tables/pdbwide_ab_entries_antigen.csv.gz')
NOV = pd.read_csv(f'{DE}/seq_novelty/tables/chain_novelty.csv.gz', low_memory=False)
# PDB-wide index
idx = {}
for line in open('/tmp/supp/pdbwide/entries.idx', encoding='latin-1'):
    p = line.rstrip('\n').split('\t')
    if len(p) < 7 or len(p[0]) != 4: continue
    auth = [a.strip() for a in re.split(r',\s(?=[A-Z][^,]*,)', p[5])] if p[5] else []
    last = p[5].split('., ')[-1].strip() if p[5] else ''
    idx[p[0].lower()] = dict(date=p[2], authors=p[5], res=p[6], exp=(p[7] if len(p) > 7 else 'X-RAY'), last=last)
dep = {}
for line in open('/tmp/supp/pdbwide/cite_all.jsonl'):
    e = json.loads(line); dep[e['rcsb_id'].lower()] = (e.get('rcsb_accession_info') or {}).get('deposit_date', '')[:10]
targets = {}
for r in L.itertuples():
    v = C[(C.pdb == r.pdb) & (C.chain == r.chain)]
    if len(v): targets[(r.pdb, r.chain)] = v.vseq.iloc[0]
probes = {k: (v if len(v) <= 95 else v[:95]) for k, v in targets.items()}
hits = {k: set() for k in targets}
cur = None
with gzip.open('/tmp/supp/pdbwide/pdb_seqres.txt.gz', 'rt') as fh:
    for line in fh:
        if line.startswith('>'): cur = line[1:5]; is_prot = 'mol:protein' in line; continue
        if not is_prot or len(line) < 90: continue
        s = line.strip()
        for k, pr in probes.items():
            if pr in s: hits[k].add(cur)
def surname(x): return re.split(r'[ ,]', str(x).strip())[0].lower() if isinstance(x, str) else ''
scan_err = dict(zip(E.pdb, E.err_first)); scan_any = dict(zip(E.pdb, E.err_any))
rows = []
for r in L.itertuples():
    k = (r.pdb, r.chain)
    d0 = pd.Timestamp(dep.get(r.pdb) or '2100-01-01')
    lab = surname(E.set_index('pdb').last_author.get(r.pdb))
    row = dict(pdb=r.pdb, chain=r.chain, group=r.group, mode=r.mode, case_pdb=r.case_pdb)
    same = sorted(hits.get(k, set()) - {r.pdb})
    info = []
    for p in same:
        dd = pd.Timestamp(dep.get(p) or '2100-01-01'); ii = idx.get(p, {})
        info.append(dict(p=p, dt=dd, before=dd < d0, samelab=surname(ii.get('last')) == lab or lab in ii.get('authors', '').lower(),
                         res=ii.get('res'), exp=ii.get('exp', ''), scan=p in scan_err, err=scan_err.get(p), anyerr=scan_any.get(p)))
    I = pd.DataFrame(info)
    if len(I):
        row['same_ab_n'] = len(I); row['same_ab_before'] = int(I.before.sum()); row['same_ab_after'] = int((~I.before).sum())
        row['same_ab_other_lab'] = int((~I.samelab).sum())
        hr = pd.to_numeric(I.res, errors='coerce')
        row['same_ab_before_hires'] = int(((hr < 3.0) & I.before).sum())
        row['same_ab_after_hires'] = int(((hr < 3.0) & ~I.before).sum())
        row['same_ab_after_inscan_clean'] = int(((~I.before) & I.scan & (I.anyerr == False)).sum())
        row['same_ab_after_inscan_err'] = int(((~I.before) & I.scan & (I.anyerr == True)).sum())
        row['same_ab_list'] = ';'.join(f"{x.p}({x.dt.date()},{'same' if x.samelab else 'other'},{x.res},{'E' if x.anyerr is True else ('c' if x.anyerr is False else '-')})" for x in I.sort_values('dt').itertuples())[:600]
    else:
        row['same_ab_n'] = 0
    # same antigen
    a = A[A.pdb == r.pdb]
    if len(a) and isinstance(a.tgt_keys.iloc[0], str):
        keys = set(a.tgt_keys.iloc[0].split(';'))
        A2 = A[A.tgt_keys.fillna('').map(lambda s: bool(keys & set(s.split(';'))))].copy()
        A2['dt'] = pd.to_datetime(A2.date)
        w = A2[((A2.dt - d0).abs().dt.days <= 183) & (A2.pdb != r.pdb)]
        row['ag_name'] = a.tgt_name.iloc[0]; row['ag_cls'] = a.tgt_cls.iloc[0]; row['ab_status'] = a.ab_status.iloc[0]
        row['ag_6m_entries'] = len(w); row['ag_6m_labs_other'] = w[w.lab != a.lab.iloc[0]].lab.nunique()
        ws = w[w.pdb.isin(scan_err)]
        row['ag_6m_inscan'] = len(ws); row['ag_6m_inscan_err'] = int(ws.pdb.map(scan_err).sum())
        wo = ws[ws.lab != a.lab.iloc[0]]
        row['ag_6m_otherlab_inscan'] = len(wo); row['ag_6m_otherlab_inscan_err'] = int(wo.pdb.map(scan_err).sum())
        row['ag_first_ab_structure'] = a.first_acc.iloc[0]; row['ag_age_days'] = a.age_days_acc.iloc[0]
        row['ag_prev12_otherlab'] = a.prev12_otherlab_acc.iloc[0]
    # novelty
    n = NOV[(NOV.pdb == r.pdb) & (NOV.chain == r.chain)]
    if len(n):
        n = n.iloc[0]
        for c in ['species', 'v_gene', 'j_gene', 'v_id', 'L1', 'L2', 'L3', 'L1_atyp', 'L2_atyp', 'L3_atyp', 'vhh', 'role', 'aux_reagent',
                  'nn_diff_V', 'nn_diff_FW', 'nn_diff_CDR3', 'n_prior_same_ab', 'gprior_V', 'gprior_VJ', 'gprior_L123', 'q_v_gene_all',
                  'mut_FR1', 'mut_CDR1', 'mut_FR2', 'mut_CDR2', 'mut_FR3', 'cdr1_len_vs_gl', 'cdr2_len_vs_gl', 'rare_win_A', 'rareins_win_A',
                  'rare_win_F', 'rareins_win_F', 'rare_win_B', 'rareins_win_B', 'rare_win_C', 'rareins_win_C', 'rare_V', 'hallmark',
                  'humanised_sig', 'scfv_linker', 'engineered_multidomain', 'irm', 'fit', 'lq_z', 'sw_Coot', 'sw_Phenix', 'sw_ISOLDE',
                  'sw_Rosetta', 'title', 'chain_desc', 'last_author', 'year', 'res', 'method', 'journal', 'doi']:
            row[c] = n.get(c)
    rows.append(row)
X = pd.DataFrame(rows); X.to_csv(f'{OUT}/tables/context.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 40)
print(X[['group', 'mode', 'pdb', 'chain', 'same_ab_n', 'same_ab_before', 'same_ab_after', 'same_ab_before_hires', 'same_ab_after_inscan_clean',
         'ag_6m_entries', 'ag_6m_labs_other', 'ag_6m_otherlab_inscan', 'ag_6m_otherlab_inscan_err', 'ag_first_ab_structure', 'v_gene', 'v_id', 'nn_diff_V', 'gprior_V']].to_string())
