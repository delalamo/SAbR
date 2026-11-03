"""Framework C-alpha deviation from a reference (T5), recomputed with the code of
deeper/verify/case_files_DE_DE2_framework_distortion_integrity/v01_recompute_fw.py (imported: FW positions, fit(),
reference chains 5DT1 H / 1N8Z A / 5DT1 L, run exclusion imgt0-1..imgt1+1 from runs_all).
Coordinates: CA atoms of the single chain from RCSB ModelServer (cached in scratchpad/paperfp/ca/), so no full mmCIF.
Sample (chains):
  all paper-confirmed chains (strict, loose, copies); every V chain of their entries (within-entry contrast);
  random window controls (one chain per entry): 300 EM entries + 300 X-ray entries (seed 11);
  all X-ray rest_case chains + 120 random post-window X-ray clean chains (heterogeneity, X-ray);
  a 40-chain overlap with the verifier's fw_recomputed.csv (reproduction check).
EM rest_case chains and the verifier's EM controls are taken from fw_recomputed.csv (same code) in s06.
Output: tables/fw_computed.csv"""
import os, sys, time, importlib.util, json, urllib.request, numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, '/home/user/SAbR/register_error_analysis/retest/paper_confirmed/model_fingerprint/scripts')
from common import SP, DEEP, TAB, load_frame
VDIR = f'{DEEP}/verify/case_files_DE_DE2_framework_distortion_integrity'
spec = importlib.util.spec_from_file_location('v01fw', f'{VDIR}/v01_recompute_fw.py'); M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)
M.CACHE = f'{SP}/paperfp/fwcache'; os.makedirs(f'{M.CACHE}/cif', exist_ok=True)   # any new full-cif download goes here
CA = f'{SP}/paperfp/ca'; os.makedirs(CA, exist_ok=True)


def fetch_ca(pc):
    p, c = pc
    fn = f'{CA}/{p}_{c}.cif'
    if os.path.exists(fn) and os.path.getsize(fn) > 500: return 'cached'
    url = f'https://models.rcsb.org/v1/{p}/atoms?auth_asym_id={c}&label_atom_id=CA&model_nums=1&encoding=cif'
    for a in range(4):
        try:
            data = urllib.request.urlopen(url, timeout=120).read()
            open(fn, 'wb').write(data); return 'ok'
        except Exception:
            time.sleep(3 * (a + 1))
    return 'fail'


def read_ca(p, c):
    fn = f'{CA}/{p}_{c}.cif'
    if not os.path.exists(fn): return None
    cols, out, inloop = [], {}, False
    for line in open(fn):
        if line.startswith('_atom_site.'):
            cols.append(line.strip().split('.')[1]); inloop = True; continue
        if inloop and (line.startswith('ATOM') or line.startswith('HETATM')):
            f = line.split()
            r = dict(zip(cols, f))
            if r.get('label_atom_id') != 'CA' or r.get('auth_asym_id') != c: continue
            if r.get('group_PDB') != 'ATOM': continue
            if r.get('label_alt_id', '.') not in ('.', 'A', '?'): continue
            ins = r.get('pdbx_PDB_ins_code', '?'); ins = '' if ins in ('?', '.') else ins
            k = (int(r['auth_seq_id']), ins)
            if k not in out:
                out[k] = np.array([float(r['Cartn_x']), float(r['Cartn_y']), float(r['Cartn_z'])])
    return out


if __name__ == '__main__':
    D = load_frame(); D = D[~D.amyloid]
    rng = np.random.default_rng(11)
    paper = D[D.grp.isin(['paper_strict', 'paper_loose', 'paper_copy'])]
    neigh = D[D.pdb.isin(set(paper.pdb))]
    def one_per_entry(x, n):
        ents = x.pdb.unique(); pick = rng.choice(ents, min(n, len(ents)), replace=False)
        x = x[x.pdb.isin(pick)].copy(); x['_r'] = rng.random(len(x))
        return x.sort_values('_r').drop_duplicates('pdb').drop(columns='_r')
    ctl_em = one_per_entry(D[(D.grp == 'control') & (D.method == 'EM')], 300)
    ctl_x = one_per_entry(D[(D.grp == 'control') & (D.method == 'X-ray')], 300)
    rest_x = D[(D.grp == 'rest_case') & (D.method == 'X-ray')]
    clean_x_post = one_per_entry(D[(D.grp == 'clean_other') & (D.method == 'X-ray') & ~D.in_paper_window], 120)
    F0 = pd.read_csv(f'{VDIR}/fw_recomputed.csv', dtype={'chain': str})
    overlap = F0.sample(40, random_state=3)[['pdb', 'chain']].merge(D, on=['pdb', 'chain'])
    sel = pd.concat([paper.assign(sample='paper'), neigh.assign(sample='paper_entry_neighbour'), ctl_em.assign(sample='window_control_EM'),
                     ctl_x.assign(sample='window_control_Xray'), rest_x.assign(sample='rest_case_Xray'),
                     clean_x_post.assign(sample='clean_Xray_post_window'), overlap.assign(sample='overlap_check')])
    sel = sel.drop_duplicates(['pdb', 'chain', 'sample'])
    keys = sorted(set(zip(sel.pdb, sel.chain)))
    print('chains to compute', len(keys), sel['sample'].value_counts().to_dict(), flush=True)
    with ThreadPoolExecutor(4) as ex:
        st = list(ex.map(fetch_ca, keys))
    print(pd.Series(st).value_counts().to_dict(), flush=True)
    # references and SAbR records
    want = {('5dt1', 'H'), ('1n8z', 'A'), ('5dt1', 'L')}
    tp = {}
    for l in open(f'{SP}/retest/templates.jsonl'):
        r = json.loads(l)
        if r.get('status') == 'ok' and (r['pdb'], r['chain']) in want: tp[(r['pdb'], r['chain'])] = r['residues']
    REF = {k: M.ref_coords(*k, tp[k]) for k in want}
    ORIG = {'H': REF[('5dt1', 'H')], 'K': REF[('1n8z', 'A')], 'L': REF[('5dt1', 'L')]}
    rec = {}
    ks = set(keys)
    for fn in ['validation', 'snapshot_rest', 'new']:
        for l in open(f'{SP}/sabr/out/{fn}.jsonl'):
            if l[9:13] not in {p for p, _ in ks}: continue
            r = json.loads(l)
            k = (r['pdb'], r['chain'])
            if r.get('status') == 'ok' and k in ks and k not in rec: rec[k] = r['residues']
    RA = pd.read_csv(f'{SP}/deeper/runs_all.csv', dtype={'chain': str})
    runpos = {}
    for r in RA.itertuples():
        runpos.setdefault((r.pdb, r.chain), set()).update(range(int(r.imgt0) - 1, int(r.imgt1) + 2))
    ctype = dict(zip(zip(D.pdb, D.chain), D.ctype))
    rows = []
    for k in keys:
        row = dict(pdb=k[0], chain=k[1])
        ca = read_ca(*k); recs = rec.get(k)
        if ca is None or recs is None or not ca:
            row['status'] = 'no coords' if not ca else 'no sabr'; rows.append(row); continue
        tgt, dis = {}, set()
        for x in recs:
            if x[4] is None or x[5]: continue
            kk = (x[0], (x[1] or '').strip())
            if kk in ca: tgt[x[4]] = ca[kk]
            if (x[4], x[5]) != (x[6], x[7]): dis.add(x[4])
        rp = runpos.get(k, set())
        rep = [n for n in M.FW if n not in rp]; strict = [n for n in rep if n not in dis]
        ref = ORIG[ctype[k]]
        row['rep_rmsd'], row['n_rep'], _ = M.fit(tgt, ref, rep)
        row['strict_rmsd'], row['n_fw'], _ = M.fit(tgt, ref, strict)
        row['fr12_rmsd'], row['n_fr12'], _ = M.fit(tgt, ref, [n for n in M.FR1 + M.FR2 if n in strict])
        row['status'] = 'ok'; rows.append(row)
    F = pd.DataFrame(rows)
    smp = sel.groupby(['pdb', 'chain'])['sample'].apply(lambda s: ';'.join(sorted(set(s)))).reset_index()
    F = F.merge(smp, on=['pdb', 'chain'])
    F.to_csv(f'{TAB}/fw_computed.csv', index=False)
    print(F.status.value_counts().to_dict())
    chk = F.merge(F0[['pdb', 'chain', 'rep_rmsd', 'strict_rmsd']], on=['pdb', 'chain'], suffixes=('', '_verifier')).dropna(subset=['rep_rmsd', 'rep_rmsd_verifier'])
    print('reproduction vs verifier fw_recomputed.csv: n', len(chk), 'max |diff| rep', float((chk.rep_rmsd - chk.rep_rmsd_verifier).abs().max()),
          'strict', float((chk.strict_rmsd - chk.strict_rmsd_verifier).abs().max()))
