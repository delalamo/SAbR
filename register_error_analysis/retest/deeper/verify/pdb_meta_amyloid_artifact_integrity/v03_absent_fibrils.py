"""3-5 A cryo-EM AL light-chain fibrils in the whole PDB (entries.idx), whether each is in the analysis set, and for
those SAbR processed but which are absent from entries.csv (6HUD, 8CPE), the SAbR-vs-ANARCI disagreement runs
(>=3 consecutive residues) recomputed from the per-residue records.
(Note: the 'sabr_s' field in the SAbR output is not a fold-similarity score - its distribution is unrelated to the
fold - so it is not used here.)"""
import json, re, pandas as pd
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
ent = pd.read_csv(f'{SP}/deeper/entries.csv')
inset = set(ent.pdb)
rows = []
for l in open('/tmp/supp/pdbwide/entries.idx', encoding='latin-1'):
    p = l.rstrip('\n').split('\t')
    if len(p) < 8 or len(p[0]) != 4 or 'ELECTRON MICROSCOPY' not in p[7]:
        continue
    t = p[3]
    if re.search(r'(?i)amyloid|fibril', t) and re.search(r'(?i)light chain|\bAL\b|lambda|kappa|immunoglobulin|AL\d|myeloma|IGLV|IGKV|FOR0', t) and not re.search(r'(?i)synuclein|ATTR|transthyretin', t):
        try:
            r = float(p[6])
        except ValueError:
            r = None
        rows.append(dict(pdb=p[0].lower(), res=r, deposit=p[2], title=t, in_scope_3_5A=(r is not None and 3 <= r <= 5), in_analysis_set=p[0].lower() in inset))
A = pd.DataFrame(rows).sort_values('pdb')
pd.set_option('display.width', 220); pd.set_option('display.max_colwidth', 90)
print(A.to_string(index=False))
print('3-5 A AL fibrils:', A.in_scope_3_5A.sum(), ' in analysis set:', (A.in_scope_3_5A & A.in_analysis_set).sum())

def runs(res):
    out, cur = [], []
    for r in res + [None]:
        dis = r is not None and r[4] is not None and r[6] is not None and (r[4], r[5] or '') != (r[6], r[7] or '')
        if dis:
            cur.append(r)
        else:
            if len(cur) >= 3:
                out.append((cur[0][4], cur[-1][4], len(cur), cur[0][6] - cur[0][4]))
            cur = []
    return out
missing = set(A[A.in_scope_3_5A & ~A.in_analysis_set].pdb)
seen = set()
for fn in ['new', 'snapshot_rest', 'validation']:
    for l in open(f'{SP}/sabr/out/{fn}.jsonl'):
        x = json.loads(l)
        if x['pdb'] in missing and (x['pdb'], x['chain']) not in seen:
            seen.add((x['pdb'], x['chain']))
            R = x['residues']; rr = runs(R)
            print(fn, x['pdb'], x['chain'], 'n_res', len(R), 'runs (imgt0, imgt1, length, delta at start):', rr,
                  ' residues in runs %d/%d' % (sum(r[2] for r in rr), len(R)))
print('processed by SAbR:', sorted({p for p, _ in seen}), ' never processed:', sorted(missing - {p for p, _ in seen}))
