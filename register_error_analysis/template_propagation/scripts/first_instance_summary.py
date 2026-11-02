"""Summarise first_instance_errors.csv: how much of the 2024 study's antibody errors is inherited from an earlier
(first) structure of the same antibody, split by how the earlier model reached the flagged entry.

Source entry = the earliest earlier structure of the same V region that carries the same register (for origin 3 this is the
antibody's first instance). Relation of the flagged entry to the source:
  same paper   : same primary-citation DOI, or same last author and deposited <= 30 days apart (DOI missing)
  same lab     : same last author, different paper
  other lab    : different last author
Last author is parsed from entries.idx author lists ("Surname, I., Surname2, I2.").
Independent error events collapse symmetry copies: one event per (entry, V region, overlapping segment).
"""
import json, re, pandas as pd, numpy as np
T = '/home/user/SAbR/register_error_analysis/template_propagation/tables'
PW = '/tmp/supp/pdbwide'
E = pd.read_csv(f'{T}/first_instance_errors.csv')
last, doi, rel, dep = {}, {}, {}, {}
for line in open(f'{PW}/entries.idx', encoding='latin-1'):
    p = line.rstrip('\n').split('\t')
    if len(p) < 6 or len(p[0]) != 4: continue
    pairs = re.findall(r'\s*([^,]+),\s*([^,]+?\.)(?=,|$)', p[5])
    last[p[0].lower()] = (pairs[-1][0].strip() + ', ' + pairs[-1][1].strip()) if pairs else p[5].split(',')[0]
for line in open(f'{PW}/cite_all.jsonl'):
    e = json.loads(line); i = e['rcsb_id'].lower()
    c = e.get('rcsb_primary_citation') or {}; doi[i] = c.get('pdbx_database_id_DOI')
    rel[i] = (e.get('rcsb_accession_info') or {}).get('initial_release_date', '')[:10]
    dep[i] = (e.get('rcsb_accession_info') or {}).get('deposit_date', '')[:10]
E['source'] = np.where(E.origin.str.startswith('3'), E.first_instance, np.where(E.origin.str.startswith('4'), E.earliest_same_prior, None))
def relation(r):
    if not isinstance(r.source, str): return None
    a, b = r.pdb, r.source
    if doi.get(a) and doi.get(a) == doi.get(b): return 'same paper'
    days = abs((pd.Timestamp(dep.get(a) or r.flagged_dep) - pd.Timestamp(dep.get(b) or r.flagged_dep)).days)
    norm = lambda x: re.sub('[^a-z]', '', (x or '').lower())
    if norm(last.get(a)) == norm(last.get(b)): return 'same paper' if (not doi.get(a) or not doi.get(b)) and days <= 30 else 'same lab, later paper'
    return 'other lab'
E['relation'] = E.apply(relation, axis=1)
E['flagged_last'] = E.pdb.map(last); E['source_last'] = E.source.map(last)
E['source_released'] = E.source.map(rel); E['source_public_before_deposit'] = E.source_released < pd.to_datetime(E.flagged_dep).dt.strftime('%Y-%m-%d')
def bucket(r):
    o = r.origin[0]
    if o in '34': return f'inherited ({r.relation})'
    if o == '1': return 'first instance, later recycled'
    if o == '2': return 'first instance, not recycled'
    if o == '5': return 'new error (earlier structures correct)'
    return 'undetermined / unmapped'
E['bucket'] = E.apply(bucket, axis=1)
# independent events: collapse chains of one entry carrying the same V region and an overlapping segment
E = E.sort_values(['pdb', 's0']); ev = []; key = None
for r in E.itertuples():
    k = (r.pdb, r.v_len, r.s0 // 10)
    ev.append(k)
E['event'] = [f'{a}|{b}|{c}' for a, b, c in ev]
order = ['inherited (same paper)', 'inherited (same lab, later paper)', 'inherited (other lab)', 'first instance, later recycled',
         'first instance, not recycled', 'new error (earlier structures correct)', 'undetermined / unmapped']
rows = {}
for unit, g in (('errors (Table S1 rows)', E), ('chains', E.drop_duplicates(['pdb', 'chain'])), ('independent error events', E.drop_duplicates('event')),
                ('entries', E.assign(rank=E.bucket.map({b: i for i, b in enumerate(order)})).sort_values('rank').drop_duplicates('pdb'))):
    vc = g.bucket.value_counts().reindex(order).fillna(0).astype(int); rows[unit] = vc
    rows[unit + ' %'] = (100 * vc / len(g)).round(1)
S = pd.DataFrame(rows); pd.set_option('display.width', 250)
print(S.to_string())
S.to_csv(f'{T}/first_instance_summary.csv')
cols = ['pdb', 'chain', 'residue_range', 'flagged_dep', 'flagged_last', 'source', 'source_last', 'source_released', 'source_public_before_deposit', 'relation', 'bucket']
print(E[E.bucket.str.startswith('inherited')][cols].sort_values(['source', 'pdb']).to_string(index=False))
print(E[E.bucket == 'first instance, later recycled'][['pdb', 'chain', 'residue_range', 'flagged_dep', 'flagged_last', 'n_later', 'later_same']].to_string(index=False))
E.to_csv(f'{T}/first_instance_errors.csv', index=False)
