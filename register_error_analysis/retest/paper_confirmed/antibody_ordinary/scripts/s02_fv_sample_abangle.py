"""Deposited VH-VL orientation (ABangle, Dunbar 2013 frame) and deposited CDR-H3 base for the paper-confirmed case Fvs
and a random sample of paper-window control Fvs/VHHs (EM and X-ray).

Reuses struct_unusual/scripts/common.py (abangle, core_imgt, dihedral, load_sabr) verbatim; coordinates are CA-only
per chain from the RCSB model server (as model_fingerprint s05), cached in scratchpad/paperord/ca; numbering is SAbR
(structure-based IMGT) from sabr/out/*.jsonl, so a register slip does not move the core.

Cases: every (pdb, H, L) Fv instance in which a paper_case_incl_copies / paper_case_first chain sits (partner from
pred/full/units.json, which paired by structure), and every case VHH chain.
Controls: random paper_control entries (seed 11): 150 EM + 150 X-ray entries with >=1 paired H/L control Fv (one Fv
per entry: first H chain with a K/L control chain whose core centroid is <28 A away), 40 EM + 40 X-ray VHH entries.
Output: tables/fv_abangle_window.csv, tables/h3base_window.csv, tables/struct_sample.csv"""
import sys, os, json, time, urllib.request, numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, '/home/user/SAbR/register_error_analysis/retest/deeper/struct_unusual/scripts')
sys.path.insert(0, os.path.dirname(__file__))
from common import abangle, core_imgt, dihedral, angle, load_sabr  # noqa: E402
from aoutil import SP, TAB, CACHE, chain_groups  # noqa: E402

CA = f'{CACHE}/ca'; os.makedirs(CA, exist_ok=True)
OLDCA = f'{SP}/paperfp/ca'
C = chain_groups()
ck = C.set_index(['pdb', 'chain'])
rng = np.random.default_rng(11)

# ---------------- cases
u = pd.DataFrame(json.load(open(f'{SP}/pred/full/units.json')))
case_chains = C[C.paper_case_incl_copies | C.paper_case_first]
cu = case_chains[['pdb', 'chain']].merge(u[['pdb', 'chain', 'mode', 'partner_chain']].drop_duplicates(['pdb', 'chain']), how='left')
rows = []
for r in cu.itertuples():
    ct = ck.loc[(r.pdb, r.chain), 'ctype']
    if r.mode == 'nanobody' or (ck.loc[(r.pdb, r.chain), 'vhh'] == 1 and not isinstance(r.partner_chain, str)):
        rows.append(dict(pdb=r.pdb, kind='VHH', H=r.chain, L=None, role='case'))
    elif isinstance(r.partner_chain, str):
        H, L = (r.chain, r.partner_chain) if ct == 'H' else (r.partner_chain, r.chain)
        rows.append(dict(pdb=r.pdb, kind='Fv', H=H, L=L, role='case'))
cases = pd.DataFrame(rows).drop_duplicates(['pdb', 'H', 'L'])

# ---------------- controls: candidate entries
ctl = C[C.paper_control & ~C.amyloid]
ent = ctl.groupby('pdb').agg(method=('method', 'first'), nH=('ctype', lambda s: 0), )
fv_ent, vhh_ent = [], []
for p, g in ctl.groupby('pdb'):
    m = g.method.iloc[0]
    if ((g.ctype == 'H') & (g.vhh != 1)).any() and g.ctype.isin(['K', 'L']).any():
        fv_ent.append((p, m))
    if (g.vhh == 1).any():
        vhh_ent.append((p, m))
fv_ent = pd.DataFrame(fv_ent, columns=['pdb', 'method']); vhh_ent = pd.DataFrame(vhh_ent, columns=['pdb', 'method'])
pick_fv = pd.concat([g.iloc[rng.permutation(len(g))[:150]] for _, g in fv_ent.groupby('method')])
pick_vhh = pd.concat([g.iloc[rng.permutation(len(g))[:40]] for _, g in vhh_ent.groupby('method')])
print('control Fv entries', pick_fv.method.value_counts().to_dict(), 'VHH entries', pick_vhh.method.value_counts().to_dict())

# ---------------- fetch CA for all scanned V chains of the needed entries
need = set(cases.pdb) | set(pick_fv.pdb) | set(pick_vhh.pdb)
keys = [(p, c) for p, c in zip(C.pdb, C.chain) if p in need]


def fetch(pc):
    p, c = pc
    fn = f'{CA}/{p}_{c}.cif'
    if os.path.exists(fn) and os.path.getsize(fn) > 500:
        return 'cached'
    old = f'{OLDCA}/{p}_{c}.cif'
    if os.path.exists(old) and os.path.getsize(old) > 500:
        os.symlink(old, fn); return 'linked'
    url = f'https://models.rcsb.org/v1/{p}/atoms?auth_asym_id={c}&label_atom_id=CA&model_nums=1&encoding=cif'
    for a in range(4):
        try:
            data = urllib.request.urlopen(url, timeout=120).read()
            open(fn, 'wb').write(data); return 'ok'
        except Exception:
            time.sleep(3 * (a + 1))
    return 'fail'


with ThreadPoolExecutor(8) as ex:
    st = list(ex.map(fetch, keys))
print('CA fetch', pd.Series(st).value_counts().to_dict())


def read_ca(p, c):
    fn = f'{CA}/{p}_{c}.cif'
    if not os.path.exists(fn):
        return {}
    cols, out, inloop = [], {}, False
    for line in open(fn):
        if line.startswith('_atom_site.'):
            cols.append(line.strip().split('.')[1]); inloop = True; continue
        if inloop and (line.startswith('ATOM') or line.startswith('HETATM')):
            f = line.split(); i = {k: j for j, k in enumerate(cols)}
            if f[i['label_atom_id']] != 'CA' or f[i['auth_asym_id']] != c:
                continue
            ic = f[i['pdbx_PDB_ins_code']]; ic = '' if ic in '?.' else ic
            k = (int(f[i['auth_seq_id']]), ic)
            if k not in out:
                out[k] = np.array([float(f[i['Cartn_x']]), float(f[i['Cartn_y']]), float(f[i['Cartn_z']])])
    return out


SAB = load_sabr(set(keys))


def imgt_ca(p, c, which='sabr'):
    """{imgt_pos: CA} without insertion codes, numbering = SAbR (structure) or ANARCI (sequence)."""
    rec = SAB.get((p, c)); ca = read_ca(p, c)
    if rec is None or not ca:
        return {}
    out = {}
    for an, ic, rn, si, ai, ains, si_, sins in rec['residues']:
        pos, ins = (si_, sins) if which == 'sabr' else (ai, ains)
        if pos is None or (ins or '') != '':
            continue
        x = ca.get((an, ic or ''))
        if x is not None:
            out[pos] = x
    return out


def centroid(m, ct):
    pos = core_imgt('H' if ct == 'H' else ct).values()
    pts = [m[q] for q in pos if q in m]
    return np.mean(pts, 0) if len(pts) >= 15 else None


# ---------------- control pairing
crow = []
for p in pick_fv.pdb:
    g = ctl[ctl.pdb == p]
    Hs = sorted(g[(g.ctype == 'H') & (g.vhh != 1)].chain); Ls = sorted(g[g.ctype.isin(['K', 'L'])].chain)
    cents = {c: centroid(imgt_ca(p, c), ck.loc[(p, c), 'ctype']) for c in Hs + Ls}
    done = False
    for h in Hs:
        if cents[h] is None:
            continue
        best = min(((float(np.linalg.norm(cents[h] - cents[l])), l) for l in Ls if cents[l] is not None), default=None)
        if best and best[0] < 28:
            crow.append(dict(pdb=p, kind='Fv', H=h, L=best[1], role='control', cdist=best[0])); done = True; break
for p in pick_vhh.pdb:
    g = ctl[(ctl.pdb == p) & (ctl.vhh == 1)]
    crow.append(dict(pdb=p, kind='VHH', H=sorted(g.chain)[0], L=None, role='control'))
S = pd.concat([cases, pd.DataFrame(crow)], ignore_index=True)

# ---------------- ABangle + H3 base
res = []
for r in S.itertuples():
    o = dict(pdb=r.pdb, kind=r.kind, H=r.H, L=r.L, role=r.role)
    mH = imgt_ca(r.pdb, r.H)
    if r.kind == 'Fv':
        Lct = ck.loc[(r.pdb, r.L), 'ctype'] if (r.pdb, r.L) in ck.index else 'K'
        o['Lct'] = Lct
        mL = imgt_ca(r.pdb, r.L)
        cH, cL = centroid(mH, 'H'), centroid(mL, Lct)
        o['cdist'] = float(np.linalg.norm(cH - cL)) if cH is not None and cL is not None else np.nan
        for which in ['sabr', 'anarci']:
            a = abangle(mH if which == 'sabr' else imgt_ca(r.pdb, r.H, 'anarci'), mL if which == 'sabr' else imgt_ca(r.pdb, r.L, 'anarci'), Lct)
            if a:
                o.update({f'{k}_{which}': v for k, v in a.items()})
    if all(q in mH for q in (115, 116, 117, 118)):
        o['h3_tau'] = dihedral(mH[115], mH[116], mH[117], mH[118]); o['h3_kinked'] = float(-30 <= o['h3_tau'] <= 100)
    res.append(o)
A = pd.DataFrame(res)
meta = C.set_index(['pdb', 'chain'])
for col in ['method', 'res', 'year', 'paper', 'unit', 'paper_case_strict', 'paper_case_incl_copies', 'paper_case_first', 'err_first', 'err_any', 'sars2']:
    A[col + '_H'] = [meta.loc[(p, h), col] if (p, h) in meta.index else np.nan for p, h in zip(A.pdb, A.H)]
    A[col + '_L'] = [meta.loc[(p, l), col] if isinstance(l, str) and (p, l) in meta.index else np.nan for p, l in zip(A.pdb, A.L)]
A['method'] = A.method_H; A['res'] = A.res_H; A['year'] = A.year_H; A['paper'] = A.paper_H
A['strict'] = A.paper_case_strict_H.fillna(False).astype(bool) | A.paper_case_strict_L.fillna(False).astype(bool)
A['incl'] = (A.paper_case_incl_copies_H.fillna(False).astype(bool) | A.paper_case_incl_copies_L.fillna(False).astype(bool)
             | A.paper_case_first_H.fillna(False).astype(bool) | A.paper_case_first_L.fillna(False).astype(bool))
A['err_chain_is_H'] = A.paper_case_incl_copies_H.fillna(False).astype(bool) | A.paper_case_first_H.fillna(False).astype(bool)
A.to_csv(f'{TAB}/fv_abangle_window.csv', index=False)
S.to_csv(f'{TAB}/struct_sample.csv', index=False)
print(A.groupby(['role', 'kind', 'method']).size())
print('ABangle computed', A.groupby(['role', 'kind']).HL_sabr.count().to_dict())
