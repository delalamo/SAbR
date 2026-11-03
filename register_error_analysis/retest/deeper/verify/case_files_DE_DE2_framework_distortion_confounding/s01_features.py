"""Recompute framework deviation for the DE2 sample (EM chains of s00.load_sample) with several variants that
probe confounding / definitional overlap, plus covariates (local map Q-score, unbuilt residues, other runs).

Per target chain (CA atoms, ANARCI numbering of the deposited sequence, as in the original s02):
  fw_orig      : original definition: FW = IMGT 4-25, 39-55, 66-104 minus every runs_all run +-1, Kabsch to the
                 single reference (5DT1 H / 1N8Z A / 5DT1 L) -> RMSD (reproduces fw_rmsd)
  fw_core      : FW minus 4-10 (N-terminal strand), minus 96-104 (CDR3 base / strand F next to strand G), minus
                 +-3 around ANY residue where SAbR != ANARCI (including 1-2 residue disagreements below run size)
  fw_far       : after the fw_orig superposition, RMSD over FW residues whose reference CA is > 15 A from the
                 reference CA of IMGT 118 (i.e. spatially away from the slip site);  fw_near: <= 15 A
  fw_med       : median per-residue CA deviation (fw_orig superposition) - robust to a few outliers
  fw_panel_min : minimum fw_orig-style RMSD over a panel of high-resolution (<=1.9 A X-ray) clean reference
                 chains of the same chain type (removes 'unlucky single reference' effects)
  fw_seqref    : RMSD to the panel reference with the highest framework sequence identity (germline-matched ref)
Covariates:
  q_fw / q_v   : mean EM Q-score over FW residues / whole V domain (local_all.jsonl)
  n_unbuilt_v  : SEQRES residues inside the V region without coordinates (seq_index gaps)
  n_disagree   : residues with SAbR != ANARCI ; other_run : chain has a non-D/E run (runs_all)
"""
import sys, os, io, gzip, json, time, urllib.request, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s00_reproduce import load_sample, SP, D, OUT, MODES
from Bio.PDB import MMCIFParser

VC = f'{D}/cache/verify_DE2_confounding'
CIFDIRS = [f'{VC}/cif', f'{D}/cache/case_files_DE/cif', f'{D}/cache/struct_unusual/cif',
           f'{D}/cache/case_files_ABCF/cif', f'{SP}/sabr/dl']
os.makedirs(f'{VC}/cif', exist_ok=True)
FW = list(range(4, 26)) + list(range(39, 56)) + list(range(66, 105))
AA3 = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H',
       'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
       'TYR': 'Y', 'VAL': 'V'}


def cif_path(p):
    for d in CIFDIRS:
        for ext in ('.cif.gz', '.cif'):
            f = f'{d}/{p}{ext}'
            if os.path.exists(f) and os.path.getsize(f) > 1000: return f
    f = f'{VC}/cif/{p}.cif.gz'
    for k in range(4):
        try:
            data = urllib.request.urlopen(f'https://files.rcsb.org/download/{p}.cif.gz', timeout=120).read()
            open(f, 'wb').write(data); return f
        except Exception:
            time.sleep(2 + 3 * k)
    return None


def chain_ca(p, c):
    f = cif_path(p)
    if f is None: return None
    h = io.StringIO(gzip.open(f, 'rt').read()) if f.endswith('.gz') else open(f)
    s = MMCIFParser(QUIET=True).get_structure(p, h)
    m = next(iter(s))
    if c not in m: return None
    return {(r.id[1], r.id[2].strip()): r['CA'].coord.astype(float) for r in m[c] if r.id[0] == ' ' and 'CA' in r}


def kabsch(P, Q):
    Pc, Qc = P.mean(0), Q.mean(0)
    H = (P - Pc).T @ (Q - Qc); U, S, Vt = np.linalg.svd(H); d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, Qc - Pc @ R.T


def sup(tgt, ref, pos):
    """tgt, ref: {imgt:int -> coord}; pos: positions to use. Returns (rmsd, per-position deviation dict)."""
    use = [n for n in pos if n in tgt and n in ref]
    if len(use) < 40: return np.nan, {}
    P = np.array([tgt[n] for n in use]); Q = np.array([ref[n] for n in use])
    R, t = kabsch(P, Q)
    dev = np.linalg.norm(P @ R.T + t - Q, axis=1)
    return float(np.sqrt((dev ** 2).mean())), dict(zip(use, dev))


def load_records(keys):
    rec = {}
    for f in ['validation', 'snapshot_rest', 'new']:
        for l in open(f'{SP}/sabr/out/{f}.jsonl'):
            r = json.loads(l)
            k = (r['pdb'], r['chain'])
            if r.get('status') == 'ok' and k in keys: rec[k] = r['residues']
    return rec


def target_feats(job):
    p, c, res, runpos, ct, ref, panel = job
    out = dict(pdb=p, chain=c)
    ca = chain_ca(p, c)
    if ca is None: out['status'] = 'no coords'; return out
    v = [x for x in res if x[4] is not None]
    tgt, seq = {}, {}
    disagree = set()
    for x in v:
        if x[5]: continue  # insertion codes are not framework positions used here
        k = (x[0], x[1])
        if k in ca: tgt[x[4]] = ca[k]
        seq[x[4]] = AA3.get(x[2], 'X')
    for x in v:
        if x[6] is None or (x[4], x[5]) != (x[6], x[7]): disagree.add(x[4])
    # unbuilt residues inside the V region (gaps in SEQRES index between consecutive built V residues)
    idx = sorted(x[3] for x in v)
    out['n_unbuilt_v'] = int(sum(b - a - 1 for a, b in zip(idx, idx[1:]) if b - a > 1))
    out['n_disagree'] = len(disagree)
    base = [n for n in FW if n not in runpos]
    out['fw_orig'], dev = sup(tgt, ref, base)
    out['n_fw'] = len(dev)
    if dev:
        out['fw_med'] = float(np.median(list(dev.values())))
        out['fw_frac_gt2'] = float(np.mean([d > 2 for d in dev.values()]))
        if 118 in ref:
            dist = {n: float(np.linalg.norm(ref[n] - ref[118])) for n in dev}
            far = [dev[n] for n in dev if dist[n] > 15]; near = [dev[n] for n in dev if dist[n] <= 15]
            out['fw_far'] = float(np.sqrt(np.mean(np.square(far)))) if len(far) >= 15 else np.nan
            out['fw_near'] = float(np.sqrt(np.mean(np.square(near)))) if len(near) >= 8 else np.nan
            out['n_far'] = len(far); out['n_near'] = len(near)
        for seg, rng in {'fr1': range(4, 26), 'fr2': range(39, 56), 'fr3a': range(66, 96), 'fr3b': range(96, 105)}.items():
            dd = [dev[n] for n in rng if n in dev]
            out[f'dev_{seg}'] = float(np.sqrt(np.mean(np.square(dd)))) if len(dd) >= 4 else np.nan
        out['dev_json'] = json.dumps({int(k): round(float(x), 3) for k, x in dev.items()})
    excl = set()
    for n in disagree | set(runpos): excl.update(range(n - 3, n + 4))
    core = [n for n in FW if not (4 <= n <= 10) and not (96 <= n <= 104) and n not in excl]
    out['fw_core'], dcore = sup(tgt, ref, core)
    out['n_core'] = len(dcore)
    # panel
    best_id, best_idn, rms_all = None, -1, []
    for pid, (pct, pref, pseq) in panel.items():
        if pct != ct: continue
        r, _ = sup(tgt, pref, base)
        if not np.isfinite(r): continue
        rms_all.append(r)
        sh = [n for n in base if n in seq and n in pseq]
        idn = np.mean([seq[n] == pseq[n] for n in sh]) if sh else 0
        if idn > best_idn: best_idn, best_id, best_r = idn, pid, r
    if rms_all:
        out['fw_panel_min'] = float(min(rms_all)); out['fw_panel_med'] = float(np.median(rms_all))
        out['fw_seqref'] = float(best_r); out['seqref_id'] = best_id; out['seqref_fw_identity'] = float(best_idn)
    out['status'] = 'ok'
    return out


def build_refs():
    # panel of high-resolution clean references
    tinfo = json.load(open(f'{VC}/template_entry_info.json'))
    tmpl = [json.loads(l) for l in open(f'{SP}/retest/templates.jsonl')]
    tmpl = [r for r in tmpl if r.get('status') == 'ok' and r['chain_type'] in ('H', 'K', 'L')
            and tinfo.get(r['pdb'], {}).get('method') == 'X-ray' and (tinfo[r['pdb']]['res'] or 9) <= 1.9]
    panel = {}
    for r in tmpl:
        v = [x for x in r['residues'] if x[4] is not None]
        if any(x[6] is None or (x[4], x[5]) != (x[6], x[7]) for x in v): continue  # need SAbR == ANARCI
        ca = chain_ca(r['pdb'], r['chain'])
        if ca is None: continue
        ref = {x[4]: ca[(x[0], x[1])] for x in v if not x[5] and (x[0], x[1]) in ca}
        if sum(n in ref for n in FW) < 70: continue
        pseq = {x[4]: AA3.get(x[2], 'X') for x in v if not x[5]}
        panel[f"{r['pdb']}_{r['chain']}"] = (r['chain_type'], ref, pseq)
    print('panel', pd.Series([v[0] for v in panel.values()]).value_counts().to_dict(), flush=True)
    # original single references (SAbR numbering, as s02)
    trec = {}
    for r in tmpl + [json.loads(l) for l in open(f'{SP}/retest/templates.jsonl')]:
        if (r['pdb'], r['chain']) in {('1n8z', 'A'), ('5dt1', 'H'), ('5dt1', 'L')} and r.get('status') == 'ok':
            trec[(r['pdb'], r['chain'])] = r['residues']
    REF = {}
    for (p, c), ct in {('5dt1', 'H'): 'H', ('1n8z', 'A'): 'K', ('5dt1', 'L'): 'L'}.items():
        ca = chain_ca(p, c)
        REF[ct] = {x[6]: ca[(x[0], x[1])] for x in trec[(p, c)] if x[6] is not None and not x[7] and (x[0], x[1]) in ca}
    return panel, REF


if __name__ == '__main__':
    from concurrent.futures import ProcessPoolExecutor
    X = load_sample()
    keys = set(zip(X.pdb, X.chain))
    panel, REF = build_refs()
    RA = pd.read_csv(f'{D}/runs_all.csv')
    runpos = {}
    for r in RA.itertuples():
        runpos.setdefault((r.pdb, r.chain), set()).update(range(int(r.imgt0) - 1, int(r.imgt1) + 2))
    rec = load_records(keys)
    ctype = dict(zip(zip(X.pdb, X.chain), X.ctype))
    jobs = [(p, c, rec[(p, c)], runpos.get((p, c), set()), ctype[(p, c)], REF[ctype[(p, c)]], panel)
            for p, c in sorted(keys) if (p, c) in rec]
    print('jobs', len(jobs), flush=True)
    t0 = time.time(); res = []
    with ProcessPoolExecutor(2) as ex:
        for i, o in enumerate(ex.map(target_feats, jobs, chunksize=4)):
            res.append(o)
            if i % 100 == 0: print(i, round(time.time() - t0), flush=True)
    F = pd.DataFrame(res)
    # local map quality (EM Q-score) per chain
    want = set(keys); q = {}
    for l in open(f'{SP}/retest/local_all.jsonl'):
        r = json.loads(l)
        if (r['pdb'], r['chain']) not in want or 'Q_SCORE' not in r['feats']: continue
        am = r['auth_map']; qs = r['feats']['Q_SCORE']
        q[(r['pdb'], r['chain'])] = {am[int(k) - 1]: v for k, v in qs.items() if v is not None}
    qrows = []
    for (p, c), qq in q.items():
        if (p, c) not in rec: continue
        v = [x for x in rec[(p, c)] if x[4] is not None]
        rp = runpos.get((p, c), set())
        fw = [qq.get(f'{x[0]}{x[1]}') for x in v if x[4] in FW and not x[5] and x[4] not in rp]
        vv = [qq.get(f'{x[0]}{x[1]}') for x in v]
        fw = [z for z in fw if z is not None]; vv = [z for z in vv if z is not None]
        qrows.append(dict(pdb=p, chain=c, q_fw=np.mean(fw) if fw else np.nan, q_v=np.mean(vv) if vv else np.nan))
    F = F.merge(pd.DataFrame(qrows), on=['pdb', 'chain'], how='left')
    oth = RA[~RA['mode'].isin(MODES)].groupby(['pdb', 'chain']).size().rename('n_other_runs').reset_index()
    F = F.merge(oth, on=['pdb', 'chain'], how='left'); F['n_other_runs'] = F.n_other_runs.fillna(0)
    F.to_csv(f'{OUT}/features_recomputed.csv', index=False)
    print(F.status.value_counts())
    print(F.drop(columns=['dev_json']).describe().T.round(3).to_string())
