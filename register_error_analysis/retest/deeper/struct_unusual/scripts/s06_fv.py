"""Pair VH/VL chains, compute ABangle orientation (published Dunbar 2013 frame) for each case/control Fv, and
CDR-H3 base pseudo-torsion/angle for each VH/VHH. Uses SAbR (structure-based) IMGT positions for coordinates so
misregistered residues do not distort the frame; ANARCI (sequence-based) positions as a sensitivity check.
Outputs tables/fv_abangle.csv, tables/h3base.csv, tables/pairs.csv."""
import pickle, numpy as np, pandas as pd
from common import SP, C, TAB, abangle, core_imgt, angle, dihedral

V = pickle.load(open(f'{C}/vchains.pkl', 'rb'))
s = pd.read_csv(f'{TAB}/sample.csv', dtype={'chain': str})
d = pd.read_csv(f'{SP}/deeper/chains.csv.gz', dtype={'chain': str},
                usecols=['pdb', 'chain', 'ctype', 'vhh', 'err_first', 'err_any', 'unit', 'vseq'])
dk = d.set_index(['pdb', 'chain'])


def ca_map(rec, which):
    out = {}
    for r in rec['res']:
        n = r[which]
        if n and n[1] == '' and 'CA' in r['atoms']:
            out[n[0]] = r['atoms']['CA']
    return out


def core_centroid(rec, ct):
    m = ca_map(rec, 'sabr')
    pos = core_imgt('H' if ct == 'H' else ct).values()
    pts = [m[p] for p in pos if p in m]
    return np.mean(pts, 0) if len(pts) >= 15 else None


# ---- pairing: for each sample chain (non-VHH) the complementary V chain with the nearest core centroid
pairs = []
for _, u in s.iterrows():
    k = (u.pdb, u.chain)
    if k not in V or u.cclass == 'VHH':
        continue
    c0 = core_centroid(V[k], u.ctype)
    if c0 is None:
        continue
    best = None
    for (p, ch), rec in V.items():
        if p != u.pdb or ch == u.chain:
            continue
        ct = dk.loc[(p, ch), 'ctype'] if (p, ch) in dk.index else rec['ctype']
        if (u.ctype == 'H') == (ct == 'H'):
            continue
        c1 = core_centroid(rec, ct)
        if c1 is None:
            continue
        dd = float(np.linalg.norm(c1 - c0))
        if best is None or dd < best[1]:
            best = (ch, dd, ct)
    if best and best[1] < 32:
        H, L = (u.chain, best[0]) if u.ctype == 'H' else (best[0], u.chain)
        Lct = best[2] if u.ctype == 'H' else u.ctype
        pairs.append(dict(pdb=u.pdb, unit_chain=u.chain, unit=u.unit, role_cc=u.role_cc, set_id=u.set_id,
                          H=H, L=L, Lct=Lct, cdist=best[1]))
P = pd.DataFrame(pairs)
print(np.histogram(P.cdist, bins=[0, 18, 20, 22, 23, 24, 25, 26, 27, 28, 30, 32])[0])
P.to_csv(f'{TAB}/pairs.csv', index=False)
print('paired units', len(P), P.groupby('role_cc').size().to_dict(), 'centroid dist median', P.cdist.median())

# ---- Fv table (dedupe by pdb,H,L); an Fv is a case Fv if either chain is a first-occurrence error chain
fv = P.drop_duplicates(['pdb', 'H', 'L']).copy()
fv['H_err'] = [bool(dk.loc[(p, h), 'err_first']) for p, h in zip(fv.pdb, fv.H)]
fv['L_err'] = [bool(dk.loc[(p, l), 'err_first']) for p, l in zip(fv.pdb, fv.L)]
fv['H_any'] = [bool(dk.loc[(p, h), 'err_any']) for p, h in zip(fv.pdb, fv.H)]
fv['L_any'] = [bool(dk.loc[(p, l), 'err_any']) for p, l in zip(fv.pdb, fv.L)]
fv['fv_case'] = fv.H_err | fv.L_err
rows = []
for _, f in fv.iterrows():
    o = dict(pdb=f.pdb, H=f.H, L=f.L)
    for which in ['sabr', 'anarci']:
        a = abangle(ca_map(V[(f.pdb, f.H)], which), ca_map(V[(f.pdb, f.L)], which), f.Lct)
        if a:
            o.update({f'{k}_{which}': v for k, v in a.items()})
    rows.append(o)
fv = fv.merge(pd.DataFrame(rows), on=['pdb', 'H', 'L'])
fv.to_csv(f'{TAB}/fv_abangle.csv', index=False)
print('Fvs', len(fv), fv.fv_case.value_counts().to_dict())
print(fv.groupby('fv_case')[['HL_sabr', 'HC1_sabr', 'LC1_sabr', 'HC2_sabr', 'LC2_sabr', 'dc_sabr', 'rmsH_sabr', 'rmsL_sabr']].median().round(2))


# ---- H3 base: tau = pseudo-dihedral CA(115,116,117,118), alpha = CA angle (115,116,117) [Kabat 100x,101,102,103]
def h3base(rec, which):
    m = ca_map(rec, which)
    if all(p in m for p in (115, 116, 117, 118)):
        return dihedral(m[115], m[116], m[117], m[118]), angle(m[115], m[116], m[117])
    return np.nan, np.nan


rows = []
for (p, ch), rec in V.items():
    if (p, ch) in dk.index and dk.loc[(p, ch), 'ctype'] == 'H':
        t1, a1 = h3base(rec, 'sabr'); t2, a2 = h3base(rec, 'anarci')
        rows.append(dict(pdb=p, chain=ch, tau_sabr=t1, alpha_sabr=a1, tau_anarci=t2, alpha_anarci=a2))
pd.DataFrame(rows).to_csv(f'{TAB}/h3base.csv', index=False)
print('H3 base rows', len(rows))
