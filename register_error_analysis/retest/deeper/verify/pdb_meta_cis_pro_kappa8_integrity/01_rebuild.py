"""Independent rebuild of the kappa cis-Pro8 frame from shared data.
- IMGT positions: ANARCI (sequence-based) labels stored in the SAbR scan output (seq_index = label_seq_id-1).
- cis peptides: RCSB CIS-PEPTIDE instance features (cached by pdb_meta in cis_sheet.jsonl); end_seq_id = residue
  whose preceding peptide bond is cis.
Extra columns not in the investigator's frame: whether IMGT 7 (the residue before Pro8) is modelled, how many of
IMGT 1-10 are modelled, any cis peptide anywhere in the V chain, number of chains / papers.
Output: frame.csv.gz (all EM+X-ray kappa chains with Pro modelled at IMGT 8)."""
import json, collections, numpy as np, pandas as pd
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/pdb_meta_cis_pro_kappa8_integrity'
ch = pd.read_csv(f'{SP}/deeper/chains.csv.gz')
ch['amyloid'] = ch.title.fillna('').str.contains(r'amyloid(?!-beta)', case=False)
K = ch[ch.ctype == 'K'].copy()
key = set(zip(K.pdb, K.chain))
res = {}
for fn in ['new', 'snapshot_rest', 'validation']:
    for l in open(f'{SP}/sabr/out/{fn}.jsonl'):
        r = json.loads(l)
        k = (r['pdb'], r['chain'])
        if k not in key or k in res or r.get('status') != 'ok':
            continue
        res[k] = r['residues']
feat = {}
for l in open(f'{SP}/deeper/cache/pdb_meta/cis_sheet.jsonl'):
    x = json.loads(l)
    feat[(x['rcsb_id'].split('.')[0].lower(), x['auth'])] = x['feats']
rows = []
for _, c in K.iterrows():
    k = (c.pdb, c.chain)
    rr = res.get(k)
    fs = feat.get(k)
    o = {'pdb': c.pdb, 'chain': c.chain, 'have_res': rr is not None, 'have_feat': fs is not None}
    if rr is not None:
        by_imgt = {}
        for an, ic, rn, si, aimgt, ains, simgt, sins in rr:
            if aimgt is not None and not ains:
                by_imgt[aimgt] = (si, rn)
        cis_end = set()
        cis_all = []
        for f in (fs or []):
            if f['type'] != 'CIS-PEPTIDE':
                continue
            for p in f['feature_positions']:
                e = p.get('end_seq_id') or p['beg_seq_id']
                cis_end.add(e - 1)
                cis_all.append((p['beg_seq_id'], p.get('end_seq_id')))
        p8 = by_imgt.get(8)
        o['pro8'] = p8 is not None and p8[1] == 'PRO'
        o['res7_modelled'] = 7 in by_imgt
        o['res7_name'] = by_imgt.get(7, (None, None))[1]
        o['n_modelled_1_10'] = sum(1 for i in range(1, 11) if i in by_imgt)
        o['cis8'] = bool(p8 is not None and p8[0] in cis_end)
        # also catch cis annotated on a neighbouring bond (off-by-one check)
        o['cis_7or9'] = any(by_imgt.get(i, (None,))[0] in cis_end for i in (7, 9) if i in by_imgt)
        p115 = by_imgt.get(115)
        o['pro115'] = p115 is not None and p115[1] == 'PRO'
        o['cis115'] = bool(p115 is not None and p115[0] in cis_end)
        vmax = max([si for an, ic, rn, si, aimgt, ains, simgt, sins in rr if aimgt is not None and si is not None] or [0])
        o['n_cis_chain'] = len(cis_all)
        o['n_cis_V'] = sum(1 for e in cis_end if e <= vmax)
        o['first_beg_end_gap'] = None if not cis_all else (cis_all[0][1] - cis_all[0][0] if cis_all[0][1] else None)
    rows.append(o)
F = pd.DataFrame(rows)
D = K.merge(F, on=['pdb', 'chain'])
D.drop(columns=['seq', 'vseq']).to_csv(f'{OUT}/frame.csv.gz', index=False)
print('kappa chains', len(D), 'with residues', D.have_res.sum(), 'with feats', D.have_feat.sum())
print('beg/end gap of cis features:', collections.Counter(F.first_beg_end_gap.dropna()))
