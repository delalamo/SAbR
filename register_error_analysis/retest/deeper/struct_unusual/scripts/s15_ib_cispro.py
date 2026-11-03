"""Does ImmuneBuilder (trained on crystal structures) put the kappa CDR-L3 Pro95 (IMGT 115) in cis for the same
sequences? Confirms that the canonical cis peptide is implied by the sequence for case and control chains alike.
Adds pred_cis to tables/cispro95.csv."""
import json, os, numpy as np, pandas as pd
from common import SP, TAB, dihedral
from common import C as CACHE
cp = pd.read_csv(f'{TAB}/cispro95.csv', dtype={'chain': str})
fv = pd.read_csv(f'{TAB}/fv_orientation.csv', dtype={'H': str, 'L': str})
d = pd.read_csv(f'{SP}/deeper/chains.csv.gz', dtype={'chain': str}, usecols=['pdb', 'chain', 'vseq'])
seq = dict(zip(zip(d.pdb, d.chain), d.vseq))
old = {(f['Hseq'], f['Lseq']): f['fv_id'] for f in json.load(open(f'{SP}/pred/full/fvs.json'))}


def path(pdb, L):
    r = fv[(fv.pdb == pdb) & (fv.L == L)]
    if not len(r):
        return None
    H = r.H.iloc[0]
    for k in ['x', 'c']:
        p = f'{CACHE}/ib/{k}_{pdb}_{H}_{L}.pdb'
        if os.path.exists(p):
            return p
    o = old.get((seq.get((pdb, H)), seq.get((pdb, L))))
    p = f'{SP}/pred/full/immunebuilder/{o}.pdb' if o else None
    return p if p and os.path.exists(p) else None


def omega_before(fn, pos=115):
    res = {}
    for line in open(fn):
        if line.startswith('ATOM') and line[21] == 'L' and line[26] == ' ':
            res.setdefault(int(line[22:26]), {})[line[12:16].strip()] = np.array(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])])
    prev = max([k for k in res if k < pos], default=None)
    if prev is None or pos not in res:
        return np.nan
    a, b = res[prev], res[pos]
    return dihedral(a['CA'], a['C'], b['N'], b['CA'])


cp['pred_omega'] = [omega_before(p) if (p := path(r.pdb, r.chain)) else np.nan for _, r in cp.iterrows()]
cp['pred_cis'] = np.where(cp.pred_omega.notna(), (cp.pred_omega.abs() < 30).astype(float), np.nan)
cp.to_csv(f'{TAB}/cispro95.csv', index=False)
print(cp.groupby('role_cc')[['cis', 'pred_cis']].agg(['mean', 'count']).round(3))
print(pd.crosstab([cp.role_cc, cp.pred_cis], cp.cis))
