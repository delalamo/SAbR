"""Extract per-residue coordinates for every V chain in each sample entry and join with SAbR (structure-based)
and ANARCI (sequence-based) IMGT numbering. Output: cache/struct_unusual/vchains.pkl
  {(pdb, chain): {'ctype', 'res': [ {auth, ic, rn, si, anarci:(pos,ins)|None, sabr:(pos,ins)|None, atoms} ]}}"""
import pickle, pandas as pd, numpy as np
from concurrent.futures import ProcessPoolExecutor
from common import SP, C, TAB, read_atoms, cif_path, load_sabr

d = pd.read_csv(f'{SP}/deeper/chains.csv.gz', usecols=['pdb', 'chain', 'ctype', 'vhh'])
s = pd.read_csv(f'{TAB}/sample.csv')
need = d[d.pdb.isin(set(s.pdb))]
keys = set(zip(need.pdb, need.chain.astype(str)))
print('V chains to extract', len(keys))
sab = load_sabr(keys)
print('with SAbR records', len(sab))
by_pdb = need.groupby('pdb')


def work(p):
    chs = set(by_pdb.get_group(p).chain.astype(str))
    atoms = read_atoms(cif_path(p), chains=chs)
    out = {}
    for ch in chs:
        rec = sab.get((p, ch))
        if rec is None or ch not in atoms or 'residues' not in rec:
            continue
        amap = {(r[0], r[1]): r for r in atoms[ch]}
        rows = []
        for an, ic, rn, si, ai, ains, si_, sins in rec['residues']:
            a = amap.get((an, ic or ''))
            rows.append(dict(auth=an, ic=ic or '', rn=rn, si=si,
                             anarci=(ai, ains or '') if ai is not None else None,
                             sabr=(si_, sins or '') if si_ is not None else None,
                             atoms=a[3] if a else {}))
        out[(p, ch)] = dict(ctype=rec.get('chain_type'), res=rows)
    return out


if __name__ == '__main__':
    allv = {}
    with ProcessPoolExecutor(2) as ex:
        for o in ex.map(work, sorted(set(s.pdb)), chunksize=4):
            allv.update(o)
    pickle.dump(allv, open(f'{C}/vchains.pkl', 'wb'))
    miss = [(p, c) for p, c in zip(s.pdb, s.chain.astype(str)) if (p, c) not in allv]
    print('extracted', len(allv), 'sample chains missing', len(miss), miss[:10])
    nomissing_atoms = sum(1 for v in allv.values() for r in v['res'] if not r['atoms'])
    print('residues without atoms', nomissing_atoms)
