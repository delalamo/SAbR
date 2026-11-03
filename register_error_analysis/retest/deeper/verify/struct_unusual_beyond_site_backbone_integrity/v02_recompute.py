"""Independent recompute of the beyond-site geometry metrics directly from the wwPDB validation XML and the SAbR
scan records (not using the investigator's residues.pkl / unit_metrics.csv).
For every V chain in every sample entry: per residue rama class, rotamer outlier, clash flag, keyed by
(auth chain, auth resnum, icode). Site definitions:
  site_disagree : every residue where ANARCI and SAbR IMGT numbers differ (any run, incl. copied/artifact), +-3
  site_runs     : first-occurrence runs (imgt0..imgt1 on either numbering), +-3 residues (investigator's definition)
For controls the matched case's run IMGT ranges are excluded (+-3), as in the original.
Output: cache/verify_beyond/res_all.pkl (per residue, all V chains of the sample entries)"""
import gzip, json, os, numpy as np, pandas as pd, xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
VR = f'{SP}/deeper/cache/struct_unusual/vrpt'
OC = f'{SP}/deeper/cache/verify_beyond'; os.makedirs(OC, exist_ok=True)
TAB = '/home/user/SAbR/register_error_analysis/retest/deeper/struct_unusual/tables'
s = pd.read_csv(f'{TAB}/sample.csv', dtype={'chain': str})
d = pd.read_csv(f'{SP}/deeper/chains.csv.gz', dtype={'chain': str}, usecols=['pdb', 'chain'])
pdbs = sorted(set(s.pdb))
keys = set(zip(d[d.pdb.isin(pdbs)].pdb, d[d.pdb.isin(pdbs)].chain))


def load_sabr(keys):
    out = {}
    for fn in ['validation', 'snapshot_rest', 'new']:
        with open(f'{SP}/sabr/out/{fn}.jsonl') as f:
            for line in f:
                r = json.loads(line)
                k = (r['pdb'], str(r['chain']))
                if k in keys and r.get('residues'):
                    out[k] = r
    return out


def parse(p):
    out = {}
    fn = f'{VR}/{p}_validation.xml.gz'
    if not os.path.exists(fn):
        return p, None
    for ev, el in ET.iterparse(gzip.open(fn), events=('end',)):
        if el.tag == 'ModelledSubgroup':
            if el.get('model', '1') == '1':
                k = (el.get('chain'), int(el.get('resnum')), (el.get('icode') or '').strip())
                if k not in out:   # first altloc record
                    out[k] = (el.get('rama'), el.get('rota'), sum(1 for c in el if c.tag == 'clash'))
            el.clear()
    return p, out


if __name__ == '__main__':
    sab = load_sabr(keys)
    print('V chains in sample entries', len(keys), 'with SAbR records', len(sab))
    with ProcessPoolExecutor(2) as ex:
        X = dict(ex.map(parse, pdbs, chunksize=4))
    print('entries without XML', [p for p, v in X.items() if v is None])
    rows = []
    for (p, ch), r in sab.items():
        vx = X.get(p) or {}
        for i, (an, ic, rn, si, ai, ains, sj, sins) in enumerate(r['residues']):
            v = vx.get((ch, an, ic or ''))
            rows.append((p, ch, i, an, ic or '', rn, si, ai, ains or '', sj, sins or '',
                         v[0] if v else None, v[1] if v else None, v[2] if v else np.nan, v is not None))
    R = pd.DataFrame(rows, columns=['pdb', 'chain', 'idx', 'auth', 'ic', 'rn', 'si', 'apos', 'ains', 'spos', 'sins',
                                    'rama', 'rota', 'nclash', 'in_xml'])
    R.to_pickle(f'{OC}/res_all.pkl')
    print(R.shape, 'in_xml', R.in_xml.mean().round(3), R.rama.value_counts(dropna=False).to_dict())
