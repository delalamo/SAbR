"""Independent (SAbR-free) structural check that the AL fibril V chains are not Ig-folded.
For each V chain: C-alpha coordinates of the V-region residues (author numbers from the SAbR per-residue records),
principal-axis spread (sqrt of the smallest eigenvalue = 'thickness'), radius of gyration, and the fraction of
residue pairs >= 10 apart in sequence that are within 6 A (long-range contact density).
A folded Ig V domain is a compact two-sheet sandwich; an amyloid layer is a flat, extended single-chain sheet.
Compared: fibril chains vs clean EM lambda/kappa/heavy V chains vs non-fibril error chains."""
import json, os, sys, random, subprocess, numpy as np, pandas as pd
from Bio.PDB import MMCIFParser
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
DL = f'{SP}/sabr/dl'
CACHE = f'{SP}/deeper/cache/verify_amyloid_integrity'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/pdb_meta_amyloid_artifact_integrity'
os.makedirs(CACHE, exist_ok=True)
AMY = {'6ic3', '6z1o', '7nsl', '9els', '9faa', '9fab', '9fac', '9liv', '9liw', '9lix', '9liy', '9lj0', '9lja', '9ljb', '9v91'}
ch = pd.read_csv(f'{SP}/deeper/chains.csv.gz', usecols=['pdb', 'chain', 'ctype', 'method', 'err_first', 'err_any', 'deposit'])
random.seed(1)
sel = []
for p in sorted(AMY):
    sel += [(p, c, 'AL fibril') for c in ch[ch.pdb == p].chain]
sel += [('6hud', c, 'AL fibril (not in set)') for c in 'ABCDE'] + [('8cpe', c, 'AL fibril (not in set)') for c in 'ABCDE']
clean = ch[(ch.method == 'EM') & ~ch.err_any]
for ct in ['L', 'K', 'H']:
    pdbs = sorted(clean[clean.ctype == ct].pdb.unique()); random.shuffle(pdbs)
    for p in pdbs[:12]:
        c = clean[(clean.pdb == p) & (clean.ctype == ct)].chain.iloc[0]
        sel.append((p, c, f'clean EM {ct}'))
err = ch[(ch.method == 'EM') & ch.err_first & ~ch.pdb.isin(AMY)]
cached = {f[:4] for f in os.listdir(DL)}
pdbs = sorted(set(err.pdb) & cached); random.shuffle(pdbs)
for p in pdbs[:30]:
    c = err[err.pdb == p].chain.iloc[0]
    sel.append((p, c, 'error EM (non-fibril)'))
want = {(p, c) for p, c, _ in sel}
REC = {}
for fn in ['new', 'snapshot_rest', 'validation']:
    for l in open(f'{SP}/sabr/out/{fn}.jsonl'):
        if l[9:13] not in {p for p, _ in want}:
            pass
        x = json.loads(l)
        k = (x['pdb'], x['chain'])
        if k in want and k not in REC:
            REC[k] = x['residues']

def get_cif(p):
    for d in (DL, CACHE):
        f = f'{d}/{p}.cif'
        if os.path.exists(f) and os.path.getsize(f) > 1000:
            return f
    f = f'{CACHE}/{p}.cif'
    subprocess.run(['curl', '-sS', '-f', '-o', f, f'https://files.rcsb.org/download/{p.upper()}.cif'], check=False)
    return f if os.path.exists(f) else None

parser = MMCIFParser(QUIET=True)
rows = []
for p, c, grp in sel:
    f = get_cif(p); R = REC.get((p, c))
    if not f or not R:
        continue
    try:
        s = parser.get_structure(p, f)[0]
    except Exception as ex:
        print('parse fail', p, ex); continue
    if c not in s:
        continue
    chn = s[c]
    nums = {(int(r[0]), (r[1] or ' ').strip() or ' ') for r in R if r[4] is not None}
    xyz = []
    for res in chn:
        key = (res.id[1], (res.id[2] or ' ').strip() or ' ')
        if key in nums and 'CA' in res:
            xyz.append(res['CA'].coord)
    X = np.array(xyz, float)
    if len(X) < 50:
        continue
    Xc = X - X.mean(0)
    ev = np.sort(np.linalg.eigvalsh(Xc.T @ Xc / len(X)))
    D = np.linalg.norm(X[:, None] - X[None], axis=-1)
    n = len(X); ii, jj = np.triu_indices(n, 10)
    rows.append(dict(pdb=p, chain=c, group=grp, n_ca=n, thickness=np.sqrt(ev[0]), mid_axis=np.sqrt(ev[1]), long_axis=np.sqrt(ev[2]),
                     rg=np.sqrt(ev.sum()), lr_contacts_per_res=(D[ii, jj] < 6.0).sum() / n))
T = pd.DataFrame(rows)
T.to_csv(f'{OUT}/fold_geometry.csv', index=False)
pd.set_option('display.width', 200)
print(T.groupby('group')[['n_ca', 'thickness', 'mid_axis', 'long_axis', 'rg', 'lr_contacts_per_res']].median().round(2).to_string())
print(T.groupby('group').size().to_string())
fib = T[T.group.str.startswith('AL')]; oth = T[~T.group.str.startswith('AL')]
print('max fibril thickness %.2f ; min non-fibril thickness %.2f' % (fib.thickness.max(), oth.thickness.min()))
print('max fibril LR contacts %.2f ; min non-fibril LR contacts %.2f' % (fib.lr_contacts_per_res.max(), oth.lr_contacts_per_res.min()))
print(fib.groupby('pdb')[['thickness', 'rg', 'lr_contacts_per_res']].median().round(2).to_string())
