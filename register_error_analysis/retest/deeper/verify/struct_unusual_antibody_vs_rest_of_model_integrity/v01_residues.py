"""Independent re-derivation of per-residue validation flags for the 344 struct_unusual sample entries.
Reads the cached wwPDB validation XMLs directly, labels V-domain residues from the raw SAbR records (not from the
investigator's residues.pkl), and labels SAbR/ANARCI disagreement runs (>=3 consecutive residues) directly from the
same records. Output: cache/verify_struct_unusual_rest/res.pkl"""
import gzip, json, numpy as np, pandas as pd, xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
VR = f'{SP}/deeper/cache/struct_unusual/vrpt'
OUTC = f'{SP}/deeper/cache/verify_struct_unusual_rest'
S = pd.read_csv('/home/user/SAbR/register_error_analysis/retest/deeper/struct_unusual/tables/sample.csv', dtype={'chain': str})
PDBS = sorted(set(S.pdb))
d = pd.read_csv(f'{SP}/deeper/chains.csv.gz', usecols=['pdb', 'chain', 'ctype', 'err_first', 'err_any'], dtype={'chain': str})
d = d[d.pdb.isin(PDBS)]
KEYS = set(zip(d.pdb, d.chain))


def load_sabr():
    out = {}
    for fn in ['validation', 'snapshot_rest', 'new']:          # later files override earlier ones
        with open(f'{SP}/sabr/out/{fn}.jsonl') as f:
            for line in f:
                r = json.loads(line)
                if (r['pdb'], r['chain']) in KEYS and 'residues' in r:
                    out[(r['pdb'], r['chain'])] = r
    return out


def runs_mask(rec, flank):
    """indices (in record order) within `flank` of a >=3-residue disagreement run"""
    dis = []
    for an, ic, rn, si, ai, ains, sa, sins in rec['residues']:
        dis.append(not (ai == sa and (ains or '') == (sins or '')))
    n = len(dis); run = np.zeros(n, bool); i = 0
    while i < n:
        if dis[i]:
            j = i
            while j < n and dis[j]:
                j += 1
            if j - i >= 3:
                run[i:j] = True
            i = j
        else:
            i += 1
    m = run.copy()
    for k in np.where(run)[0]:
        m[max(0, k - flank):k + flank + 1] = True
    return run, m


def parse(p):
    rows = []
    for ev, el in ET.iterparse(gzip.open(f'{VR}/{p}_validation.xml.gz'), events=('end',)):
        if el.tag == 'ModelledSubgroup':
            if el.get('model', '1') == '1' and (el.get('rama') is not None or el.get('rota') is not None):
                q = el.get('Q_score')
                rows.append((p, el.get('chain'), int(el.get('resnum')), (el.get('icode') or '').strip(),
                             el.get('resname'), el.get('altcode'), el.get('rama'), el.get('rota'),
                             any(c.tag == 'clash' for c in el), float(q) if q else np.nan))
            el.clear()
    return rows


if __name__ == '__main__':
    sab = load_sabr()
    print('V chains in chains.csv for sample entries', len(KEYS), 'with SAbR records', len(sab))
    lab = {}
    for (p, ch), rec in sab.items():
        run, m3 = runs_mask(rec, 3)
        _, m10 = runs_mask(rec, 10)
        anyrun = bool(run.any())
        for k, r in enumerate(rec['residues']):
            lab[(p, ch, r[0], r[1] or '')] = (True, bool(run[k]), bool(m3[k]), bool(m10[k]), anyrun)
    with ProcessPoolExecutor(2) as ex:
        allrows = [r for rr in ex.map(parse, PDBS, chunksize=4) for r in rr]
    R = pd.DataFrame(allrows, columns=['pdb', 'chain', 'resnum', 'ic', 'resname', 'altcode', 'rama', 'rota', 'clash', 'q'])
    print('raw rows', len(R), 'duplicate keys (altlocs)', R.duplicated(['pdb', 'chain', 'resnum', 'ic']).sum())
    L = [lab.get((p, c, n, i), (False, False, False, False, False)) for p, c, n, i in zip(R.pdb, R.chain, R.resnum, R.ic)]
    R[['isV', 'inrun', 'm3', 'm10', 'chain_has_run']] = pd.DataFrame(L, index=R.index)
    vchains = set(KEYS)
    R['is_vchain'] = [(p, c) in vchains for p, c in zip(R.pdb, R.chain)]
    R.to_pickle(f'{OUTC}/res.pkl')
    # coverage: SAbR residues that did not match any validation residue
    nmatched = R.isV.sum(); nlab = len(lab)
    print('SAbR V residues', nlab, 'matched in XML', nmatched)
    print(R.groupby('isV').rama.value_counts(normalize=True).round(4))
