"""Deposited V domains vs ImmuneBuilder (ABodyBuilder2 / NanoBodyBuilder2) predictions of the same sequence.
Correspondence is by structural position: deposited SAbR IMGT (pos, ins) <-> predicted IMGT (pos, ins), so a
register error does not by itself create deviation (a sequence-based correspondence is reported separately).
Per chain: framework CA RMSD (excluding the error run +-3 in case chains), CDR CA RMSD after framework
superposition, CDR backbone-bin agreement (ABEGO) for non-H3 CDRs, predicted H3 base kink, ensemble error.
Per Fv: ABangle of the prediction and |deposited - predicted|.
Output tables/pred_compare_chain.csv, tables/pred_compare_fv.csv"""
import json, os, pickle, numpy as np, pandas as pd
from common import SP, C, TAB, dihedral, angle, rmsd_after, kabsch, abangle

V = pickle.load(open(f'{C}/vchains.pkl', 'rb'))
s = pd.read_csv(f'{TAB}/sample.csv', dtype={'chain': str})
fvo = pd.read_csv(f'{TAB}/fv_orientation.csv', dtype={'H': str, 'L': str, 'unit_chain': str})
runs = pd.read_csv(f'{SP}/deeper/runs_first_occurrence.csv', dtype={'chain': str})
case_runs = {k: g[['imgt0', 'imgt1']].values.tolist() for k, g in runs.groupby(['pdb', 'chain'])}
d = pd.read_csv(f'{SP}/deeper/chains.csv.gz', dtype={'chain': str}, usecols=['pdb', 'chain', 'vseq'])
seq = dict(zip(zip(d.pdb, d.chain), d.vseq))
old = json.load(open(f'{SP}/pred/full/fvs.json'))
old_by_seq = {(f['Hseq'], f['Lseq']): f['fv_id'] for f in old}
IB = f'{C}/ib'
FW = set(range(1, 27)) | set(range(39, 56)) | set(range(66, 105)) | set(range(118, 129))
CDR = {'1': range(27, 39), '2': range(56, 66), '3': range(105, 118)}


def read_pred(fn):
    out = {}
    order = []
    for line in open(fn):
        if line.startswith('ATOM'):
            ch = line[21]; pos = int(line[22:26]); ins = line[26].strip(); an = line[12:16].strip()
            key = (pos, ins)
            if (ch, key) not in out:
                out[(ch, key)] = {}; order.append((ch, key))
            out[(ch, key)][an] = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return out, order


def pred_path(kind, pdb, H, L=None):
    fid = f'{kind}_{pdb}_{H}' + (f'_{L}' if L else '')
    p = f'{IB}/{fid}.pdb'
    if os.path.exists(p):
        return p, f'{IB}/{fid}_err.npy'
    if kind == 'x':  # fall back to the earlier case predictions (same protocol, no error estimates)
        k = (seq.get((pdb, H)), seq.get((pdb, L)) if L else '')
        if k in old_by_seq:
            p = f'{SP}/pred/full/immunebuilder/{old_by_seq[k]}.pdb'
            if os.path.exists(p):
                return p, None
    return None, None


def dep_map(pdb, ch, which):
    out = {}
    for r in V[(pdb, ch)]['res']:
        n = r[which]
        if n and r['atoms']:
            out[(n[0], n[1])] = r['atoms']
    return out


def abego(phi, psi):
    if not (np.isfinite(phi) and np.isfinite(psi)):
        return None
    if phi < 0:
        return 'A' if -75 <= psi < 50 else 'B'
    return 'G' if -100 <= psi < 100 else 'E'


def phipsi(m, keys):
    """m: {(pos,ins): atoms}; keys: ordered list of keys (chain order). Returns {key:(phi,psi)}"""
    out = {}
    for i, k in enumerate(keys):
        a = m[k]
        phi = psi = np.nan
        if i > 0 and 'C' in m[keys[i - 1]] and all(x in a for x in 'N CA C'.split()):
            if np.linalg.norm(m[keys[i - 1]]['C'] - a['N']) < 2.0:
                phi = dihedral(m[keys[i - 1]]['C'], a['N'], a['CA'], a['C'])
        if i + 1 < len(keys) and 'N' in m[keys[i + 1]] and all(x in a for x in 'N CA C'.split()):
            if np.linalg.norm(a['C'] - m[keys[i + 1]]['N']) < 2.0:
                psi = dihedral(a['N'], a['CA'], a['C'], m[keys[i + 1]]['N'])
        out[k] = (phi, psi)
    return out


def chain_compare(pdb, ch, ct, pred, pch, err, err_keys, excl, site_runs=None):
    """excl: runs excluded from the framework fit (case errors only); site_runs: where pred_err_site is read."""
    site_runs = excl if site_runs is None else site_runs
    o = {}
    pm = {k: v for (c, k), v in pred.items() if c == pch}
    pkeys = [k for (c, k) in order_all if c == pch]
    for which in ['sabr', 'anarci']:
        dm = dep_map(pdb, ch, which)
        common_k = [k for k in pkeys if k in dm and 'CA' in dm[k] and 'CA' in pm[k]]
        fw = [k for k in common_k if k[0] in FW and k[1] == '' and not any(a - 3 <= k[0] <= b + 3 for a, b in excl)]
        if len(fw) < 30:
            continue
        P = np.array([dm[k]['CA'] for k in fw]); Q = np.array([pm[k]['CA'] for k in fw])
        rms, R, t = rmsd_after(P, Q)
        o[f'fw_rmsd_{which}'] = rms
        if which == 'sabr':
            o['n_fw'] = len(fw)
            for cn, rg in CDR.items():
                ks = [k for k in common_k if k[0] in rg]
                if len(ks) >= 3:
                    X = np.array([dm[k]['CA'] for k in ks]) @ R + t; Y = np.array([pm[k]['CA'] for k in ks])
                    o[f'cdr{cn}_rmsd'] = float(np.sqrt(((X - Y) ** 2).sum(1).mean()))
            # backbone-bin agreement for CDR1/2 (and L3 for light chains), structural correspondence
            dkeys = [k for k in sorted(dm, key=lambda x: (x[0], x[1])) if True]
            dkeys = [r['sabr'] for r in V[(pdb, ch)]['res'] if r['sabr'] and r['atoms']]
            dkeys = [(a, b) for a, b in dkeys]
            dpp = phipsi(dm, dkeys); ppp = phipsi(pm, pkeys)
            loops = ['1', '2'] + (['3'] if ct != 'H' else [])
            agree, tot, loop_ok = 0, 0, []
            for cn in loops:
                ks = [k for k in common_k if k[0] in CDR[cn]]
                bins = [(abego(*dpp.get(k, (np.nan, np.nan))), abego(*ppp.get(k, (np.nan, np.nan)))) for k in ks]
                bins = [b for b in bins if b[0] and b[1]]
                if bins:
                    m = sum(a == b for a, b in bins)
                    agree += m; tot += len(bins)
                    loop_ok.append(m == len(bins))
                    o[f'cdr{cn}_bin_agree'] = m / len(bins)
            o['cdr_bin_agree'] = agree / tot if tot else np.nan
            o['all_loops_same_bins'] = float(all(loop_ok)) if loop_ok else np.nan
            if ct == 'H':
                ca = {k[0]: pm[k]['CA'] for k in pkeys if k[1] == '' and 'CA' in pm[k]}
                if all(p in ca for p in (115, 116, 117, 118)):
                    tau = dihedral(ca[115], ca[116], ca[117], ca[118])
                    o['pred_h3_tau'] = tau; o['pred_h3_kinked'] = float(-30 <= tau <= 100)
                    o['pred_h3_alpha'] = angle(ca[115], ca[116], ca[117])
    if err is not None:
        e = dict(zip(err_keys, err))
        h3 = [v for (c, k), v in e.items() if c == pch and k[0] in CDR['3']]
        allv = [v for (c, k), v in e.items() if c == pch]
        o['pred_err_cdr3'] = float(np.mean(h3)) if h3 else np.nan
        o['pred_err_chain'] = float(np.mean(allv))
        site = [v for (c, k), v in e.items() if c == pch and any(a - 2 <= k[0] <= b + 2 for a, b in site_runs)]
        o['pred_err_site'] = float(np.mean(site)) if site else np.nan
    return o


rows, frows = [], []
# matched case runs for controls (so the predicted-error at the 'site' is measured at the same positions)
set_runs = {u.set_id: case_runs.get((u.pdb, u.chain), []) for _, u in s[s.role_cc == 'case'].iterrows()}
for _, u in s.iterrows():
    kind = 'x' if u.role_cc == 'case' else 'c'
    if (u.pdb, u.chain) not in V:
        continue
    if u.cclass == 'VHH':
        p, ep = pred_path(kind, u.pdb, u.chain)
        if p is None:
            continue
        pred, order_all = read_pred(p)
        err = np.load(ep) if ep and os.path.exists(ep) else None
        excl = case_runs.get((u.pdb, u.chain), []) if kind == 'x' else []
        o = chain_compare(u.pdb, u.chain, 'H', pred, 'H', err, order_all, excl,
                          site_runs=excl if kind == 'x' else set_runs.get(u.set_id, []))
        rows.append(dict(unit=u.unit, pdb=u.pdb, chain=u.chain, role_cc=u.role_cc, cclass=u.cclass, fv=None, **o))
for _, f in fvo.iterrows():
    kind = 'x' if f.fv_case else 'c'
    p, ep = pred_path(kind, f.pdb, f.H, f.L)
    if p is None:
        continue
    pred, order_all = read_pred(p)
    err = np.load(ep) if ep and os.path.exists(ep) else None
    fo = dict(pdb=f.pdb, H=f.H, L=f.L, fv_case=f.fv_case)
    for ch, pch, ct in [(f.H, 'H', 'H'), (f.L, 'L', f.Lct)]:
        excl = case_runs.get((f.pdb, ch), [])
        site_runs = excl if kind == 'x' else (set_runs.get(f.set_id, []) if ch == f.unit_chain else [])
        o = chain_compare(f.pdb, ch, ct, pred, pch, err, order_all, excl, site_runs=site_runs)
        err_chain = (f.H_err if pch == 'H' else f.L_err)
        rows.append(dict(pdb=f.pdb, chain=ch, role_cc='case' if f.fv_case else 'control', cclass=ct,
                         fv=f'{f.H}{f.L}', chain_has_error=bool(err_chain), **o))
    caH = {k[0]: v['CA'] for (c, k), v in pred.items() if c == 'H' and k[1] == '' and 'CA' in v}
    caL = {k[0]: v['CA'] for (c, k), v in pred.items() if c == 'L' and k[1] == '' and 'CA' in v}
    a = abangle(caH, caL, f.Lct)
    if a:
        for k in ['HL', 'HC1', 'LC1', 'HC2', 'LC2', 'dc']:
            fo[f'pred_{k}'] = a[k]; fo[f'd_{k}'] = f[f'{k}_sabr'] - a[k]
        fo['maha_dep'] = f.maha_sabr
        fo['pred_rmsH'] = a['rmsH']; fo['pred_rmsL'] = a['rmsL']
        fo['pred_core_rms'] = (a['rmsH'] + a['rmsL']) / 2
    frows.append(fo)
P = pd.DataFrame(rows); F = pd.DataFrame(frows)
P.to_csv(f'{TAB}/pred_compare_chain.csv', index=False); F.to_csv(f'{TAB}/pred_compare_fv.csv', index=False)
print(P.groupby(['role_cc']).size(), F.groupby('fv_case').size())
print(P.groupby(['role_cc', P.cclass.replace({'K': 'VL', 'L': 'VL'})])[
    ['fw_rmsd_sabr', 'fw_rmsd_anarci', 'cdr1_rmsd', 'cdr2_rmsd', 'cdr3_rmsd', 'cdr_bin_agree', 'pred_h3_kinked',
     'pred_err_cdr3', 'pred_err_chain']].median().round(3).T)
