"""Per-residue table for every V chain in the sample entries: backbone dihedrals, cis peptides, CA-CA distances,
atom completeness, wwPDB validation (Ramachandran, rotamer, clashes, bond/angle outliers) and Q-score.
Output: cache/struct_unusual/residues.pkl (DataFrame) and entry_q.csv (entry-wide median Q-score)."""
import gzip, json, pickle, numpy as np, pandas as pd, xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from common import SP, C, TAB, HEAVY, dihedral

V = pickle.load(open(f'{C}/vchains.pkl', 'rb'))
pdbs = sorted({k[0] for k in V})


def vrpt(p):
    out = {}
    fn = f'{C}/vrpt/{p}_validation.xml.gz'
    try:
        for ev, el in ET.iterparse(gzip.open(fn), events=('end',)):
            if el.tag == 'ModelledSubgroup':
                if el.get('model', '1') != '1':
                    el.clear(); continue
                key = (el.get('chain'), int(el.get('resnum')), (el.get('icode') or '').strip())
                kids = [c.tag for c in el]
                q = el.get('Q_score') or el.get('Q-score') or el.get('qscore')
                out[key] = dict(rama=el.get('rama'), rota=el.get('rota'), cis_v=el.get('cis_peptide') == 'yes',
                                nclash=kids.count('clash'), nbond=kids.count('bond-outlier') + kids.count('mog-bond-outlier'),
                                nangle=kids.count('angle-outlier') + kids.count('mog-angle-outlier'),
                                vq=float(q) if q else np.nan, altcode=el.get('altcode'))
                el.clear()
    except Exception as e:
        print('vrpt fail', p, e)
    return out


def localq():
    """{(pdb, chain): {auth_str: q}} for sample entries + entry-wide medians"""
    want = set(pdbs); per, ent = {}, {}
    with open(f'{SP}/retest/local_all.jsonl') as f:
        for line in f:
            p = line[9:13]
            if p not in want:
                continue
            r = json.loads(line)
            fk = 'Q_SCORE' if 'Q_SCORE' in r['feats'] else 'RSCC'
            amap = r['auth_map']
            q = {amap[int(k) - 1]: v for k, v in r['feats'][fk].items() if v is not None}
            per[(p, r['chain'])] = q
            ent.setdefault(p, []).extend(q.values())
    return per, {p: float(np.median(v)) for p, v in ent.items()}


def work(p):
    vr = vrpt(p)
    rows = []
    for (pp, ch), rec in V.items():
        if pp != p:
            continue
        res = rec['res']
        for i, r in enumerate(res):
            a = r['atoms']
            prev = res[i - 1] if i > 0 and res[i - 1]['si'] == r['si'] - 1 else None
            nxt = res[i + 1] if i + 1 < len(res) and res[i + 1]['si'] == r['si'] + 1 else None
            phi = psi = omega = caca = np.nan
            if prev and all(k in prev['atoms'] for k in ['C']) and all(k in a for k in ['N', 'CA', 'C']):
                phi = dihedral(prev['atoms']['C'], a['N'], a['CA'], a['C'])
            if nxt and all(k in a for k in ['N', 'CA', 'C']) and 'N' in nxt['atoms']:
                psi = dihedral(a['N'], a['CA'], a['C'], nxt['atoms']['N'])
            if nxt and all(k in a for k in ['CA', 'C']) and all(k in nxt['atoms'] for k in ['N', 'CA']):
                omega = dihedral(a['CA'], a['C'], nxt['atoms']['N'], nxt['atoms']['CA'])
                caca = float(np.linalg.norm(a['CA'] - nxt['atoms']['CA']))
            v = vr.get((ch, r['auth'], r['ic']), {})
            rows.append(dict(pdb=p, chain=ch, idx=i, auth=r['auth'], ic=r['ic'], rn=r['rn'], si=r['si'],
                             apos=r['anarci'][0] if r['anarci'] else np.nan, ains=r['anarci'][1] if r['anarci'] else '',
                             spos=r['sabr'][0] if r['sabr'] else np.nan, sins=r['sabr'][1] if r['sabr'] else '',
                             phi=phi, psi=psi, omega=omega, caca_next=caca,
                             natoms=len([k for k in a if k != 'OXT']), nexp=HEAVY.get(r['rn'], np.nan),
                             bb_ok=all(k in a for k in ['N', 'CA', 'C', 'O']),
                             rama=v.get('rama'), rota=v.get('rota'), cis_v=v.get('cis_v'), nclash=v.get('nclash'),
                             nbond=v.get('nbond'), nangle=v.get('nangle'), vq=v.get('vq'), in_vrpt=bool(v),
                             prev_si_gap=(r['si'] - res[i - 1]['si'] - 1) if i > 0 else np.nan))
    return rows


if __name__ == '__main__':
    per, entq = localq()
    rows = []
    with ProcessPoolExecutor(2) as ex:
        for o in ex.map(work, pdbs, chunksize=4):
            rows.extend(o)
    R = pd.DataFrame(rows)
    R['q'] = [per.get((p, c), {}).get(f'{a}{ic}', np.nan) for p, c, a, ic in zip(R.pdb, R.chain, R.auth, R.ic)]
    R.to_pickle(f'{C}/residues.pkl')
    pd.Series(entq, name='entry_q').rename_axis('pdb').to_csv(f'{TAB}/entry_q.csv')
    print(R.shape, 'in_vrpt', R.in_vrpt.mean().round(3), 'q available', R.q.notna().mean().round(3))
    print(R.rama.value_counts(dropna=False).to_dict())
    print('vrpt Q-score available', R.vq.notna().mean().round(3))
