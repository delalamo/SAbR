"""Independent recomputation of the framework C-alpha RMSD ('fw_rmsd') used in finding DE2_framework_distortion.

Sample (EM only):
  error               : chains with a first-occurrence D+/D-/E run (taken directly from runs_first_occurrence.csv)
  matched_clean       : the investigator's matched clean EM sample (struct_sample.csv)
  within_paper_clean  : the investigator's within-paper clean sample (struct_sample.csv)
  other_mode_error    : EM chains with first-occurrence runs of OTHER modes only (A, B, C, F, other, D0) - specificity check
For each chain, features computed with new code (not importing the investigator's module):
  rep_rmsd       : replica of fw_rmsd (FW = ANARCI IMGT 4-25, 39-55, 66-104; exclude runs_all positions imgt0-1..imgt1+1;
                   superpose on reference CA (5DT1 H / 1N8Z A / 5DT1 L))
  strict_rmsd    : same, also excluding every residue whose SAbR number != ANARCI number (incl. 1-2 residue disagreements)
  fr12_rmsd      : superposition + RMSD on FR1 (4-25) + FR2 (39-55) only, strict exclusions (far from CDR3/FR4)
  reg_*          : per-region RMSD after the strict global superposition (FR1, FR2, FR3a 66-95, FR3b 96-104)
  panel_mean / panel_min : strict RMSD averaged / minimised over a 4-structure reference panel per chain type
                   (VHH chains get a VHH panel)
  n_fw           : number of CA atoms used for strict superposition
"""
import os, io, gzip, json, glob, sys, time, urllib.request
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
from Bio.PDB import MMCIFParser

SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
D = f'{SP}/deeper'
CACHE = f'{D}/cache/verify_DE2_integrity'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/case_files_DE_DE2_framework_distortion_integrity'
os.makedirs(f'{CACHE}/cif', exist_ok=True)
MODES = ['Dplus_CDR3short', 'Dminus_CDR3long', 'E_FR4bulge']
CIFDIRS = [f'{CACHE}/cif'] + glob.glob(f'{D}/cache/*/cif') + [f'{SP}/sabr/dl']

def cif_path(p):
    for d in CIFDIRS:
        for ext in ('.cif.gz', '.cif'):
            f = f'{d}/{p}{ext}'
            if os.path.exists(f) and os.path.getsize(f) > 1000: return f
    f = f'{CACHE}/cif/{p}.cif.gz'
    for k in range(4):
        try:
            data = urllib.request.urlopen(f'https://files.rcsb.org/download/{p}.cif.gz', timeout=120).read()
            open(f, 'wb').write(data); return f
        except Exception:
            time.sleep(2 + 3 * k)
    return None

def load_chain(p, c):
    f = cif_path(p)
    if f is None: return None
    h = io.StringIO(gzip.open(f, 'rt').read()) if f.endswith('.gz') else open(f)
    s = MMCIFParser(QUIET=True).get_structure(p, h)
    m = next(iter(s))
    if c not in m: return None
    out = {}
    for r in m[c]:
        if r.id[0] != ' ' or 'CA' not in r: continue
        out[(r.id[1], r.id[2].strip())] = r['CA'].coord.astype(float)
    return out

FR1 = list(range(4, 26)); FR2 = list(range(39, 56)); FR3a = list(range(66, 96)); FR3b = list(range(96, 105))
FW = FR1 + FR2 + FR3a + FR3b

def kabsch(P, Q):
    Pc, Qc = P.mean(0), Q.mean(0)
    H = (P - Pc).T @ (Q - Qc); U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T)); R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, Qc - Pc @ R.T

def fit(tgt, ref, pos_sup, pos_eval=None):
    P = np.array([tgt[n] for n in pos_sup if n in tgt and n in ref]); Q = np.array([ref[n] for n in pos_sup if n in tgt and n in ref])
    if len(P) < 20: return np.nan, len(P), {}
    R, t = kabsch(P, Q)
    rms = float(np.sqrt((((P @ R.T + t) - Q) ** 2).sum(1).mean()))
    reg = {}
    for nm, pos in (pos_eval or {}).items():
        P2 = np.array([tgt[n] for n in pos if n in tgt and n in ref]); Q2 = np.array([ref[n] for n in pos if n in tgt and n in ref])
        reg[nm] = float(np.sqrt((((P2 @ R.T + t) - Q2) ** 2).sum(1).mean())) if len(P2) >= 5 else np.nan
    return rms, len(P), reg

def work(job):
    p, items, refs = job
    out = []
    try:
        f = cif_path(p)
        if f is None: return [dict(pdb=p, chain=c, status='no cif') for c, *_ in items]
        h = io.StringIO(gzip.open(f, 'rt').read()) if f.endswith('.gz') else open(f)
        m = next(iter(MMCIFParser(QUIET=True).get_structure(p, h)))
        for c, recs, runpos, ct, vhh in items:
            row = dict(pdb=p, chain=c, status='ok')
            if c not in m: row['status'] = 'chain missing'; out.append(row); continue
            ca = {(r.id[1], r.id[2].strip()): r['CA'].coord.astype(float) for r in m[c] if r.id[0] == ' ' and 'CA' in r}
            v = [x for x in recs if x[4] is not None]
            # ANARCI number (no insertion) -> CA
            tgt = {}; disagree = set()
            for x in v:
                if x[5]: continue
                k = (x[0], x[1])
                if k in ca: tgt[x[4]] = ca[k]
                if (x[4], x[5]) != (x[6], x[7]): disagree.add(x[4])
            row['n_disagree_fw'] = len(disagree & set(FW))
            rep_pos = [n for n in FW if n not in runpos]
            strict_pos = [n for n in rep_pos if n not in disagree]
            rp = refs['orig']
            row['rep_rmsd'], row['n_rep'], _ = fit(tgt, rp, rep_pos)
            ev = {'FR1': [n for n in FR1 if n in strict_pos], 'FR2': [n for n in FR2 if n in strict_pos],
                  'FR3a': [n for n in FR3a if n in strict_pos], 'FR3b': [n for n in FR3b if n in strict_pos]}
            row['strict_rmsd'], row['n_fw'], reg = fit(tgt, rp, strict_pos, ev)
            for k_, v_ in reg.items(): row[f'reg_{k_}'] = v_
            row['fr12_rmsd'], row['n_fr12'], _ = fit(tgt, rp, [n for n in FR1 + FR2 if n in strict_pos])
            row['fr13a_rmsd'], _, _ = fit(tgt, rp, [n for n in FR1 + FR2 + FR3a if n in strict_pos])
            pan = [fit(tgt, r_, strict_pos)[0] for r_ in refs['panel']]
            row['panel_mean'] = float(np.nanmean(pan)) if np.isfinite(pan).any() else np.nan
            row['panel_min'] = float(np.nanmin(pan)) if np.isfinite(pan).any() else np.nan
            for i, x in enumerate(pan): row[f'panel_{i}'] = x
            out.append(row)
    except Exception as e:
        out.append(dict(pdb=p, chain='*', status=f'error {e}'))
    return out

def ref_coords(p, c, tpl):
    ca = load_chain(p, c)
    num = {(x[0], x[1]): (x[6], x[7]) for x in tpl if x[6] is not None}
    return {num[k][0]: v for k, v in ca.items() if k in num and not num[k][1]}

if __name__ == '__main__':
    C = pd.read_csv(f'{D}/chains.csv.gz')
    R = pd.read_csv(f'{D}/runs_first_occurrence.csv'); RA = pd.read_csv(f'{D}/runs_all.csv')
    S = pd.read_csv(f'{D}/cache/case_files_DE/struct_sample.csv')
    em = C[C.method == 'EM']
    de_keys = set(zip(R[R['mode'].isin(MODES)].pdb, R[R['mode'].isin(MODES)].chain))
    oth_keys = set(zip(R[~R['mode'].isin(MODES)].pdb, R[~R['mode'].isin(MODES)].chain)) - de_keys
    err = em[[k in de_keys for k in zip(em.pdb, em.chain)]].assign(group='error')
    oth = em[[k in oth_keys for k in zip(em.pdb, em.chain)]].assign(group='other_mode_error')
    ctl = S[S.group != 'error'][['pdb', 'chain', 'group']].merge(C, on=['pdb', 'chain'])
    ctl = ctl[ctl.method == 'EM']
    sel = pd.concat([err, oth, ctl])
    print(sel.groupby('group').agg(chains=('pdb', 'size'), units=('unit', 'nunique'), entries=('pdb', 'nunique')))
    keys = set(zip(sel.pdb, sel.chain))
    # references (SAbR numbering from templates.jsonl)
    want = {('5dt1', 'H'), ('6cbv', 'H'), ('4jqi', 'H'), ('4m48', 'H'), ('1n8z', 'A'), ('6cbv', 'L'), ('4jqi', 'L'), ('4m48', 'L'),
            ('5dt1', 'L'), ('5cck', 'L'), ('6axk', 'B'), ('5iq9', 'B'), ('4b50', 'A'), ('7z1d', 'FFF'), ('6yz5', 'F'), ('7jkm', 'K')}
    tp = {}
    for l in open(f'{SP}/retest/templates.jsonl'):
        r = json.loads(l)
        if r.get('status') == 'ok' and (r['pdb'], r['chain']) in want: tp[(r['pdb'], r['chain'])] = r['residues']
    REF = {k: ref_coords(*k, tp[k]) for k in want}
    for k, v in REF.items(): print('ref', k, len(v))
    ORIG = {'H': REF[('5dt1', 'H')], 'K': REF[('1n8z', 'A')], 'L': REF[('5dt1', 'L')]}
    PANEL = {'H': [REF[k] for k in [('5dt1', 'H'), ('6cbv', 'H'), ('4jqi', 'H'), ('4m48', 'H')]],
             'K': [REF[k] for k in [('1n8z', 'A'), ('6cbv', 'L'), ('4jqi', 'L'), ('4m48', 'L')]],
             'L': [REF[k] for k in [('5dt1', 'L'), ('5cck', 'L'), ('6axk', 'B'), ('5iq9', 'B')]],
             'VHH': [REF[k] for k in [('4b50', 'A'), ('7z1d', 'FFF'), ('6yz5', 'F'), ('7jkm', 'K')]]}
    # sabr records
    rec = {}
    for fn in ['validation', 'snapshot_rest', 'new']:
        for l in open(f'{SP}/sabr/out/{fn}.jsonl'):
            r = json.loads(l)
            if r.get('status') != 'ok': continue
            k = (r['pdb'], r['chain'])
            if k in keys and k not in rec: rec[k] = r['residues']
    runpos = {}
    for r in RA.itertuples():
        runpos.setdefault((r.pdb, r.chain), set()).update(range(int(r.imgt0) - 1, int(r.imgt1) + 2))
    info = dict(zip(zip(C.pdb, C.chain), zip(C.ctype, C.vhh)))
    jobs = []
    for p, g in sel.drop_duplicates(['pdb', 'chain']).groupby('pdb'):
        items = []
        for c in sorted(set(g.chain)):
            if (p, c) not in rec: continue
            ct, vhh = info[(p, c)]
            items.append((c, rec[(p, c)], runpos.get((p, c), set()), ct, vhh))
        if not items: continue
        ct0 = items[0][3]
        # refs per chain: handle per chain type by splitting jobs per chain type
        for ctk in sorted({(it[3], it[4]) for it in items}):
            its = [it for it in items if (it[3], it[4]) == ctk]
            jobs.append((p, its, {'orig': ORIG[ctk[0]], 'panel': PANEL['VHH' if ctk[1] == 1 else ctk[0]]}))
    print('jobs', len(jobs), flush=True)
    res = []; t0 = time.time()
    with ProcessPoolExecutor(2) as ex:
        for i, o in enumerate(ex.map(work, jobs, chunksize=2)):
            res.extend(o)
            if i % 100 == 0: print(i, round(time.time() - t0), flush=True)
    F = pd.DataFrame(res)
    F.to_csv(f'{OUT}/fw_recomputed.csv', index=False)
    sel[['pdb', 'chain', 'group', 'unit', 'paper', 'year', 'ctype', 'vhh', 'res', 'species', 'v_id', 'v_gene', 'L1', 'L2', 'L3', 'sars2', 'deposit']].to_csv(
        f'{OUT}/sample_groups.csv', index=False)
    print(F.status.value_counts())
