"""Entry-level validation metrics (wwPDB XML header) and the same metrics restricted to non-antibody-V residues
('rest of the model': antigen, constant domains, other chains). Output tables/entry_vrpt.csv"""
import gzip, numpy as np, pandas as pd, xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from common import SP, C, TAB

s = pd.read_csv(f'{TAB}/sample.csv', dtype={'chain': str})
R = pd.read_pickle(f'{C}/residues.pkl')
vkeys = set(zip(R.pdb, R.chain.astype(str), R.auth, R.ic))
HDR = ['Q_score', 'clashscore', 'percent-rama-outliers', 'percent-rota-outliers', 'bonds_rmsz', 'angles_rmsz',
       'atom_inclusion_backbone', 'EMDB-resolution']


def work(p):
    o = dict(pdb=p)
    n = rama = rota = clash = allw = 0; qs = []
    nv = vrama = vallw = vrota = vclash = 0; vqs = []
    for ev, el in ET.iterparse(gzip.open(f'{C}/vrpt/{p}_validation.xml.gz'), events=('end',)):
        if el.tag == 'Entry':
            for h in HDR:
                v = el.get(h)
                try:
                    o['e_' + h.replace('-', '_')] = float(v)
                except (TypeError, ValueError):
                    o['e_' + h.replace('-', '_')] = np.nan
        elif el.tag == 'ModelledSubgroup':
            if el.get('model', '1') == '1' and (el.get('rama') is not None or el.get('rota') is not None):
                key = (p, el.get('chain'), int(el.get('resnum')), (el.get('icode') or '').strip())
                if key in vkeys:
                    nv += 1
                    vrama += el.get('rama') == 'OUTLIER'; vallw += el.get('rama') == 'Allowed'
                    vrota += el.get('rota') == 'OUTLIER'; vclash += any(c.tag == 'clash' for c in el)
                    q = el.get('Q_score')
                    if q:
                        vqs.append(float(q))
                else:
                    allw += el.get('rama') == 'Allowed'
                    n += 1
                    rama += el.get('rama') == 'OUTLIER'; rota += el.get('rota') == 'OUTLIER'
                    clash += any(c.tag == 'clash' for c in el)
                    q = el.get('Q_score')
                    if q:
                        qs.append(float(q))
            el.clear()
    o.update(rest_n=n, rest_rama_out=100 * rama / max(n, 1), rest_rota_out=100 * rota / max(n, 1),
             rest_clash=100 * clash / max(n, 1), rest_q=np.mean(qs) if qs else np.nan, v_n=nv,
             rest_rama_allowed=100 * allw / max(n, 1), v_rama_out=100 * vrama / max(nv, 1),
             v_rama_allowed=100 * vallw / max(nv, 1), v_rota_out=100 * vrota / max(nv, 1),
             v_clash=100 * vclash / max(nv, 1), v_q=np.mean(vqs) if vqs else np.nan)
    for k in ['rama_out', 'rama_allowed', 'rota_out', 'clash', 'q']:
        o[f'gap_{k}'] = o[f'v_{k}'] - o[f'rest_{k}']  # antibody V domains minus rest of the same model
    return o


if __name__ == '__main__':
    with ProcessPoolExecutor(2) as ex:
        rows = list(ex.map(work, sorted(set(s.pdb)), chunksize=4))
    E = pd.DataFrame(rows)
    E.to_csv(f'{TAB}/entry_vrpt.csv', index=False)
    E = E.merge(s.groupby('pdb').role_cc.first(), on='pdb')
    print(E.groupby('role_cc').median(numeric_only=True).T.round(3))
