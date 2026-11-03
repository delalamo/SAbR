"""Completeness check: apply the SAbR-free fold-geometry test (v05) to ALL first-occurrence error chains with a cached
mmCIF, to see whether any non-fibril error chain is also non-globular (another non-Ig-fold detector artifact the title
regex would miss)."""
import json, os, numpy as np, pandas as pd
from Bio.PDB import MMCIFParser
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
DL = f'{SP}/sabr/dl'; CACHE = f'{SP}/deeper/cache/verify_amyloid_integrity'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/pdb_meta_amyloid_artifact_integrity'
AMY = {'6ic3', '6z1o', '7nsl', '9els', '9faa', '9fab', '9fac', '9liv', '9liw', '9lix', '9liy', '9lj0', '9lja', '9ljb', '9v91'}
ch = pd.read_csv(f'{SP}/deeper/chains.csv.gz', usecols=['pdb', 'chain', 'ctype', 'method', 'err_first', 'err_modes', 'title'])
err = ch[ch.err_first]
have = {f[:4]: f'{DL}/{f}' for f in os.listdir(DL) if f.endswith('.cif')}
have.update({f[:4]: f'{CACHE}/{f}' for f in os.listdir(CACHE) if f.endswith('.cif')})
err = err[err.pdb.isin(have)]
want = set(zip(err.pdb, err.chain)); REC = {}
for fn in ['new', 'snapshot_rest', 'validation']:
    for l in open(f'{SP}/sabr/out/{fn}.jsonl'):
        x = json.loads(l); k = (x['pdb'], x['chain'])
        if k in want and k not in REC:
            REC[k] = x['residues']
parser = MMCIFParser(QUIET=True); rows = []
for p, g in err.groupby('pdb'):
    try:
        s = parser.get_structure(p, have[p])[0]
    except Exception:
        continue
    for c in g.chain:
        R = REC.get((p, c))
        if not R or c not in s:
            continue
        nums = {(int(r[0]), (r[1] or ' ').strip() or ' ') for r in R if r[4] is not None}
        X = np.array([res['CA'].coord for res in s[c] if (res.id[1], (res.id[2] or ' ').strip() or ' ') in nums and 'CA' in res], float)
        if len(X) < 40:
            rows.append(dict(pdb=p, chain=c, n_ca=len(X))); continue
        Xc = X - X.mean(0); ev = np.sort(np.linalg.eigvalsh(Xc.T @ Xc / len(X)))
        D = np.linalg.norm(X[:, None] - X[None], axis=-1); ii, jj = np.triu_indices(len(X), 10)
        rows.append(dict(pdb=p, chain=c, n_ca=len(X), thickness=np.sqrt(ev[0]), rg=np.sqrt(ev.sum()), lr=(D[ii, jj] < 6).sum() / len(X)))
T = pd.DataFrame(rows).merge(err, on=['pdb', 'chain'])
T['amy'] = T.pdb.isin(AMY)
T.to_csv(f'{OUT}/fold_geometry_all_error_chains.csv.gz', index=False)
print('error chains scanned:', len(T), 'entries:', T.pdb.nunique(), ' of', ch[ch.err_first].pdb.nunique(), 'error entries')
print(T.groupby('amy')[['thickness', 'rg', 'lr']].describe().T.round(2).to_string())
odd = T[~T.amy & ((T.thickness < 4) | (T.lr < 0.3))]
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 70)
print('non-fibril error chains with thickness<4 A or LR contacts<0.3:', len(odd))
print(odd[['pdb', 'chain', 'ctype', 'method', 'n_ca', 'thickness', 'rg', 'lr', 'err_modes', 'title']].to_string(index=False))
