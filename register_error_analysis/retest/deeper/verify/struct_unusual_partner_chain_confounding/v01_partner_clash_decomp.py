"""Verifier (confounding lens) for struct_unusual finding 'partner_chain'.

Per partner chain (the other VH/VL of a sampled unit's Fv), recompute from the wwPDB validation XML:
  * residues with any clash (should reproduce p_clash_V);
  * decomposition by clash counterpart: intra-partner, the unit chain (error chain in cases), the unit chain's
    error site +-5 residues, other chains (antigen, constant domains, other Fvs);
  * clash magnitude;
  * VH-VL interface residues (any heavy atom within 4.5 A of the unit chain's V domain) vs non-interface;
  * sub-threshold ANARCI/SAbR disagreements (1-2 residue runs) in the partner, which the err_any filter lets through.
Output: tables/partner_decomp.csv
"""
import gzip, pickle, sys, numpy as np, pandas as pd, xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor

SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
C = f'{SP}/deeper/cache/struct_unusual'
INV = '/home/user/SAbR/register_error_analysis/retest/deeper/struct_unusual/tables'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/struct_unusual_partner_chain_confounding/tables'

U = pd.read_csv(f'{INV}/unit_metrics.csv', dtype={'chain': str, 'partner': str})
U = U[U.partner.notna()].copy()
R = pd.read_pickle(f'{C}/residues.pkl'); R['chain'] = R.chain.astype(str)
runs = pd.read_csv(f'{SP}/deeper/runs_first_occurrence.csv', dtype={'chain': str})
s = pd.read_csv(f'{INV}/sample.csv', dtype={'chain': str})
case_runs = {k: g[['imgt0', 'imgt1']].values.tolist() for k, g in runs.groupby(['pdb', 'chain'])}
set_site = {u.set_id: case_runs.get((u.pdb, u.chain), []) for _, u in s[s.role_cc == 'case'].iterrows()}
V = pickle.load(open(f'{C}/vchains.pkl', 'rb'))


def parse_clashes(p):
    """{(chain, resnum, icode): [(cid, atom, mag)]} and {cid: [(chain,resnum,icode,atom)]} for model 1."""
    byres, bycid = {}, {}
    for ev, el in ET.iterparse(gzip.open(f'{C}/vrpt/{p}_validation.xml.gz'), events=('end',)):
        if el.tag == 'ModelledSubgroup':
            if el.get('model', '1') == '1':
                key = (el.get('chain'), int(el.get('resnum')), (el.get('icode') or '').strip())
                for c in el:
                    if c.tag == 'clash':
                        cid = c.get('cid'); a = c.get('atom'); m = float(c.get('clashmag') or 'nan')
                        byres.setdefault(key, []).append((cid, a, m))
                        bycid.setdefault(cid, []).append(key + (a,))
            el.clear()
    return byres, bycid


def heavy_xyz(rec):
    return {(r['auth'], r['ic']): np.array([v for k, v in r['atoms'].items()]) for r in rec['res']}


def work(p):
    sub = U[U.pdb == p]
    byres, bycid = parse_clashes(p)
    rows = []
    for _, u in sub.iterrows():
        pc, uc = u.partner, u.chain
        g = R[(R.pdb == p) & (R.chain == pc)].sort_values('idx')
        gu = R[(R.pdb == p) & (R.chain == uc)].sort_values('idx')
        if not len(g):
            continue
        # unit-chain site residues (+-5 sequence neighbours), using either numbering
        site = set()
        rng_ = set_site.get(u.set_id, [])
        if len(gu) and rng_:
            hit = np.zeros(len(gu), bool)
            for a, b in rng_:
                hit |= ((gu.apos >= a) & (gu.apos <= b)).values | ((gu.spos >= a) & (gu.spos <= b)).values
            for i in np.where(hit)[0]:
                for j in range(max(0, i - 5), min(len(gu), i + 6)):
                    site.add((uc, int(gu.auth.iloc[j]), gu.ic.iloc[j]))
        # interface: partner residue with any heavy atom within 4.5 A of the unit chain V domain
        iface = set()
        if (p, pc) in V and (p, uc) in V:
            xp = heavy_xyz(V[(p, pc)]); xu = np.vstack([a for a in heavy_xyz(V[(p, uc)]).values() if len(a)])
            for k, a in xp.items():
                if len(a) and (np.linalg.norm(a[:, None, :] - xu[None, :, :], axis=2) < 4.5).any():
                    iface.add(k)
        n = len(g)
        cnt = dict(any=0, intra=0, unit=0, unit_site=0, other=0, intra_only=0, iface_any=0, noniface_any=0,
                   noniface_intra_only=0)
        mags, n_iface = [], 0
        for _, r in g.iterrows():
            key = (pc, int(r.auth), r.ic)
            isif = (int(r.auth), r.ic) in iface
            n_iface += isif
            cl = byres.get(key, [])
            if not cl:
                continue
            kinds = set()
            for cid, a, m in cl:
                mags.append(m)
                others = [o for o in bycid.get(cid, []) if not (o[:3] == key and o[3] == a)]
                for o in others:
                    oc = o[0]
                    if oc == pc:
                        kinds.add('intra')
                    elif oc == uc:
                        kinds.add('unit')
                        if o[:3] in site:
                            kinds.add('unit_site')
                    else:
                        kinds.add('other')
            cnt['any'] += 1
            for k in ('intra', 'unit', 'unit_site', 'other'):
                cnt[k] += k in kinds
            cnt['intra_only'] += kinds <= {'intra'}
            if isif:
                cnt['iface_any'] += 1
            else:
                cnt['noniface_any'] += 1
                cnt['noniface_intra_only'] += kinds <= {'intra'}
        # sub-threshold disagreements in the partner: residues where ANARCI and SAbR numbering differ
        both = g.apos.notna()
        dis = both & ((g.apos != g.spos) | (g.ains.fillna('') != g.sins.fillna('')))
        dd = dis.values.astype(int)
        maxrun = cur = 0
        for v in dd:
            cur = cur + 1 if v else 0; maxrun = max(maxrun, cur)
        o = dict(unit=u.unit, set_id=u.set_id, pdb=p, chain=uc, partner=pc, n_p=n, n_iface=n_iface,
                 n_noniface=n - n_iface, p_dis_res=int(dd.sum()), p_dis_maxrun=maxrun,
                 p_clash_mag_mean=np.mean(mags) if mags else np.nan, p_nclash_atoms=len(mags))
        for k, v in cnt.items():
            denom = n if not k.startswith(('iface', 'noniface')) else (n_iface if k.startswith('iface') else n - n_iface)
            o[f'pc_{k}'] = 100 * v / denom if denom else np.nan
        o['n_site_unit'] = len(site)
        rows.append(o)
    return rows


if __name__ == '__main__':
    import os
    os.makedirs(OUT, exist_ok=True)
    pdbs = sorted(U.pdb.unique())
    rows = []
    with ProcessPoolExecutor(2) as ex:
        for r in ex.map(work, pdbs, chunksize=4):
            rows.extend(r)
    D = pd.DataFrame(rows)
    D.to_csv(f'{OUT}/partner_decomp.csv', index=False)
    print(D.shape)
    print(D.describe().T.round(2))
