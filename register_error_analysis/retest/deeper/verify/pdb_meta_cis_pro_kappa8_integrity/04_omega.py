"""Recompute the omega angle of the IMGT7-IMGT8 (Ser/Thr7-Pro8) peptide bond directly from coordinates (RCSB
ModelServer atoms, model 1, first altloc) and compare with the wwPDB CIS-PEPTIDE annotation used by pdb_meta.
cis: |omega| < 30 deg. Also counts all cis peptides in the chain from coordinates (annotation completeness)."""
import os, json, numpy as np, pandas as pd
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/pdb_meta_cis_pro_kappa8_integrity'
C = f'{SP}/deeper/cache/verify_cis8/coords'
S = pd.read_csv(f'{OUT}/coord_sample.csv')
F = pd.read_csv(f'{OUT}/frame.csv.gz')
keys = set(zip(S.pdb, S.chain))
seqidx8 = {}
for fn in ['new', 'snapshot_rest', 'validation']:
    for l in open(f'{SP}/sabr/out/{fn}.jsonl'):
        r = json.loads(l)
        k = (r['pdb'], r['chain'])
        if k not in keys or k in seqidx8 or r.get('status') != 'ok':
            continue
        d = {x[4]: x[3] for x in r['residues'] if x[4] is not None and not x[5]}
        seqidx8[k] = d


def dihed(p0, p1, p2, p3):
    b0 = p0 - p1; b1 = p2 - p1; b2 = p3 - p2
    b1 /= np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1; w = b2 - np.dot(b2, b1) * b1
    x = np.dot(v, w); y = np.dot(np.cross(b1, v), w)
    return np.degrees(np.arctan2(y, x))


def parse(fn):
    cols = []; atoms = {}
    for line in open(fn):
        if line.startswith('_atom_site.'):
            cols.append(line.strip().split('.', 1)[1]); continue
        if line.startswith('ATOM') or line.startswith('HETATM'):
            t = line.split()
            if len(t) != len(cols):
                continue
            a = dict(zip(cols, t))
            if a.get('pdbx_PDB_model_num', '1') != '1' or a['label_seq_id'] in ('.', '?'):
                continue
            k = (int(a['label_seq_id']), a['label_atom_id'])
            if k in atoms:
                continue
            atoms[k] = np.array([float(a['Cartn_x']), float(a['Cartn_y']), float(a['Cartn_z'])])
    return atoms


rows = []
for p, c in zip(S.pdb, S.chain):
    fn = f'{C}/{p}_{c}.cif'
    if not os.path.exists(fn) or (p, c) not in seqidx8:
        continue
    at = parse(fn)
    sids = sorted({k[0] for k in at})
    om = {}
    for s in sids:
        need = [(s - 1, 'CA'), (s - 1, 'C'), (s, 'N'), (s, 'CA')]
        if all(n in at for n in need):
            d = np.linalg.norm(at[(s - 1, 'C')] - at[(s, 'N')])
            if d < 2.0:
                om[s] = dihed(*[at[n].copy() for n in need])
    i8 = seqidx8[(p, c)].get(8)
    o8 = om.get(i8 + 1) if i8 is not None else None   # label_seq_id = seq_index + 1
    rows.append(dict(pdb=p, chain=c, omega8=o8, cis8_coord=(o8 is not None and abs(o8) < 30),
                     n_cis_coord=sum(1 for v in om.values() if abs(v) < 30), n_bonds=len(om)))
O = pd.DataFrame(rows).merge(F[['pdb', 'chain', 'method', 'err_first', 'cis8', 'n_cis_chain', 'year', 'res', 'paper']], on=['pdb', 'chain'])
O.to_csv(f'{OUT}/omega_check.csv', index=False)
print(len(O), 'chains with coordinates')
print(pd.crosstab([O.method, O.err_first], [O.cis8, O.cis8_coord], margins=True))
print('disagreements annotation vs coordinates:')
print(O[O.cis8 != O.cis8_coord][['pdb', 'chain', 'method', 'err_first', 'omega8', 'cis8', 'n_cis_chain', 'n_cis_coord']].to_string(index=False))
print('omega8 distribution for annotated-trans chains (|omega|):')
t = O[~O.cis8.astype(bool)]
print(t.groupby('err_first').omega8.apply(lambda s: s.abs().describe()).unstack().round(1))
print('total cis peptides: annotated vs coordinates', O.n_cis_chain.sum(), O.n_cis_coord.sum(), 'chains where coord>annot', (O.n_cis_coord > O.n_cis_chain).sum())
