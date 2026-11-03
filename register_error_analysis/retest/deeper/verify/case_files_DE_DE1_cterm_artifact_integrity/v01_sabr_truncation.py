"""V01: direct test of the claimed SAbR chain-end artifact (DE1_cterm_artifact).

Pipeline mirrors the scan (sabr_anarci_scan/scripts/sabr_scan_light.py, fast mode): chain_type fixed, chain trimmed to
the V domain +-3 author numbers, unobserved-residue metadata passed as a minimal mmcif_dict. ANARCI numbering is taken
from the scan records (sequence-based, so independent of how much of the chain is modelled).

Experiments
  R  reproduce : the 12 deposited D+ models that end at ANARCI IMGT <=121 (the investigator's 'short' group), untouched.
  P  predicted : ImmuneBuilder models of the same 12 antibodies (correct register by construction), keeping only the
                 ANARCI IMGT positions that are built in the deposited model (same gaps, same C-terminal end).
                 Also run the full predicted model.
  T  truncation: clean chains (no run of any kind) from the investigator's control sample, H/K/L, model built to >=125,
                 truncated after ANARCI IMGT 117,118,119,120,121,122,123,125 (removed residues listed as unobserved,
                 as in a deposited model with unbuilt FR4), plus the untruncated chain. Variant without metadata at 118.
  L  long D+   : the investigator's 'long' D+ models (ending >121; real shifts per superposition) truncated after 118:
                 does SAbR still report the +1 shift? (positive control for detection at a chain end)
Output: tables/v01_sabr_runs.csv (one row per chain x condition).
"""
import os, sys, io, gzip, json, time, re, warnings, logging
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[_v] = "1"
warnings.filterwarnings('ignore'); logging.disable(logging.WARNING)
import numpy as np, pandas as pd, urllib.request
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
D = f'{SP}/deeper'
CA = f'{D}/cache/verify_DE1'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/case_files_DE_DE1_cterm_artifact_integrity'
os.makedirs(f'{CA}/cif', exist_ok=True); os.makedirs(f'{OUT}/tables', exist_ok=True)
CIFDIRS = [f'{CA}/cif', f'{D}/cache/case_files_DE/cif', f'{D}/cache/struct_unusual/cif', f'{D}/cache/case_files_ABCF/cif',
           f'{SP}/sabr/dl', f'{D}/cache/verify_artifact_indel/cif']
A3 = dict(ALA='A', ARG='R', ASN='N', ASP='D', CYS='C', GLN='Q', GLU='E', GLY='G', HIS='H', ILE='I', LEU='L', LYS='K', MET='M',
          PHE='F', PRO='P', SER='S', THR='T', TRP='W', TYR='Y', VAL='V', MSE='M')
A1 = {v: k for k, v in A3.items() if k != 'MSE'}


def cif_path(p):
    for d in CIFDIRS:
        for ext in ('.cif.gz', '.cif'):
            f = f'{d}/{p}{ext}'
            if os.path.exists(f) and os.path.getsize(f) > 1000: return f
    f = f'{CA}/cif/{p}.cif.gz'
    for k in range(4):
        try:
            data = urllib.request.urlopen(f'https://files.rcsb.org/download/{p}.cif.gz', timeout=120).read()
            open(f, 'wb').write(data); return f
        except Exception:
            time.sleep(2 + 3 * k)
    return None


def sabr_records(keys):
    rec = {}
    for f in ['validation', 'snapshot_rest', 'new']:
        for l in open(f'{SP}/sabr/out/{f}.jsonl'):
            if not any(f'"{p}"' in l[:40] for p, _ in keys): continue
            r = json.loads(l)
            k = (r['pdb'], r['chain'])
            if k in keys and r.get('status') == 'ok': rec[k] = r
    return rec


def runs_of(rows):
    """rows: list of (anarci(num,ins), sabr(num,ins)); returns list of runs (start_anarci, end_anarci, delta_last)"""
    flags = [a != s for a, s in rows]
    out, cur = [], []
    for k, f in enumerate(flags + [False]):
        if f: cur.append(k)
        else:
            if len(cur) >= 3:
                i, j = cur[0], cur[-1]
                out.append((rows[i][0], rows[j][0], rows[j][1][0] - rows[j][0][0]))
            cur = []
    return out


def run_sabr(st, model, chobj, c, ct, keep_keys, unobs, use_meta=True):
    """keep only residues whose (num,ins) in keep_keys; unobs = list of ((num,ins), aa1) to declare unobserved."""
    from Bio.PDB.Structure import Structure
    from Bio.PDB.Model import Model
    from sabr import renumber_structure
    tst = Structure('x'); tst.header = dict(getattr(st, 'header', {}) or {}); tm = Model(model.id, model.serial_num); tst.add(tm)
    tc = chobj.copy()
    for r in list(tc):
        if r.id[0] != ' ' or (r.id[1], r.id[2].strip()) not in keep_keys: tc.detach_child(r.id)
    tm.add(tc)
    kw = dict(chain=c, chain_type=ct)
    if use_meta:
        kw['mmcif_dict'] = {'_pdbx_unobs_or_zero_occ_residues.auth_asym_id': [c] * len(unobs),
                            '_pdbx_unobs_or_zero_occ_residues.auth_seq_id': [str(k[0]) for k, _ in unobs],
                            '_pdbx_unobs_or_zero_occ_residues.PDB_ins_code': [k[1] or '?' for k, _ in unobs],
                            '_pdbx_unobs_or_zero_occ_residues.auth_comp_id': [A3.get(a, 'UNK') for _, a in unobs],
                            '_pdbx_unobs_or_zero_occ_residues.PDB_model_num': [str(model.serial_num)] * len(unobs)}
    new = renumber_structure(tst, **kw)
    olds = [r for r in tc]
    news = [r for r in next(iter(new))[c]]
    return {(o.id[1], o.id[2].strip()): (n.id[1], n.id[2].strip()) for o, n in zip(olds, news)}


def summarise(tag, label, pdb, c, anarci, sab, order, extra=None):
    rows = [(anarci[k], sab[k]) for k in order if k in anarci and k in sab]
    rr = runs_of(rows)
    # runs touching the CDR3/FR4 end (ANARCI >= 105)
    end = [r for r in rr if r[1][0] >= 105]
    last = rows[-1] if rows else None
    d = dict(exp=tag, label=label, pdb=pdb, chain=c, n_res=len(rows),
             last_anarci=f'{last[0][0]}{last[0][1]}' if last else None, last_sabr=f'{last[1][0]}{last[1][1]}' if last else None,
             runs=';'.join(f'{a[0]}{a[1]}-{b[0]}{b[1]}:{dl:+d}' for a, b, dl in rr),
             cterm_run=bool(end), cterm_run_pos=any(dl > 0 for _, _, dl in end), cterm_run_neg=any(dl < 0 for _, _, dl in end),
             w118_sabr=None)
    for k in order:
        if anarci.get(k) == (118, '') and k in sab: d['w118_sabr'] = f'{sab[k][0]}{sab[k][1]}'
    if extra: d.update(extra)
    return d


def deposited_job(job):
    """job: (pdb, [(chain, ctype, label, trunc_list, residues)])"""
    from Bio.PDB import MMCIFParser
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict
    p, reqs = job
    out = []
    try:
        f = cif_path(p)
        fh = io.StringIO(gzip.open(f, 'rt').read()) if f.endswith('.gz') else open(f)
        parser = MMCIFParser(QUIET=True); st = parser.get_structure(p, fh); md = parser._mmcif_dict
        model = next(iter(st))
        ps = pd.DataFrame({k.split('.')[1]: md[k] for k in md if k.startswith('_pdbx_poly_seq_scheme.')})
        for c, ct, label, truncs, res in reqs:
            try:
                if c not in model: out.append(dict(exp=label, pdb=p, chain=c, note='chain missing')); continue
                chobj = model[c]
                q = ps[ps.pdb_strand_id == c]
                seqlab = [((int(n), '' if i in ('.', '?') else i), A1.get(m, 'X')) for n, i, m in zip(q.pdb_seq_num, q.pdb_ins_code, q.mon_id)]
                observed = {(r.id[1], r.id[2].strip()) for r in chobj if r.id[0] == ' '}
                unobs0 = [(k, a) for k, a in seqlab if k not in observed]
                anarci = {(x[0], x[1]): (x[4], x[5]) for x in res if x[4] is not None}
                scan_sabr = {(x[0], x[1]): (x[6], x[7]) for x in res if x[6] is not None}
                vnums = [k[0] for k in anarci]
                lo, hi = min(vnums) - 3, max(vnums) + 3
                order = [(r.id[1], r.id[2].strip()) for r in chobj if r.id[0] == ' ']
                base = [k for k in order if lo <= k[0] <= hi]
                if label.startswith('R'):
                    out.append(summarise('scan_record', label, p, c, anarci, scan_sabr, base))
                for T in truncs:
                    if T == 'full':
                        keep = base; extra_unobs = []
                    else:
                        T_ = int(str(T).replace('nometa', ''))
                        idx = [i for i, k in enumerate(base) if anarci.get(k) == (T_, '')]
                        if not idx: continue
                        keep = base[:idx[0] + 1]
                        extra_unobs = [(k, A1.get(chobj[(' ', k[0], k[1] or ' ')].get_resname(), 'X')) for k in base[idx[0] + 1:]]
                    use_meta = not str(T).endswith('nometa')
                    sab = run_sabr(st, model, chobj, c, ct, set(keep), unobs0 + extra_unobs, use_meta)
                    out.append(summarise(f'trunc_{T}', label, p, c, anarci, sab, keep))
            except Exception as e:
                out.append(dict(exp=label, pdb=p, chain=c, note=f'error {type(e).__name__}: {str(e)[:120]}'))
    except Exception as e:
        out.append(dict(exp='entry', pdb=p, note=f'error {type(e).__name__}: {str(e)[:120]}'))
    return out


def anarci_number(seq, ct):
    from anarci import run_anarci
    an = run_anarci([('q', seq)], scheme='imgt')
    for dom, det in zip(an[1][0] or [], an[2][0] or []):
        if det['chain_type'] == ct or (ct in 'KL' and det['chain_type'] in 'KL'):
            num = [((pp, ii.strip()), a) for (pp, ii), a in dom[0] if a != '-']
            return dom[1], [n for n, a in num]
    return None


def predicted_job(job):
    """job: (pdb, chain, ctype, fv_file, pred_chain, deposited residues)"""
    from Bio.PDB import PDBParser
    p, c, ct, fpath, pc, res = job
    out = []
    try:
        st = PDBParser(QUIET=True).get_structure('pred', fpath); model = next(iter(st)); chobj = model[pc]
        rr = [r for r in chobj if r.id[0] == ' ']
        seq = ''.join(A3.get(r.get_resname(), 'X') for r in rr)
        h = anarci_number(seq, 'H' if ct == 'H' else ct)
        if h is None: return [dict(exp='pred', pdb=p, chain=c, note='no anarci')]
        s0, num = h
        anarci = {}
        for k, r in enumerate(rr):
            if s0 <= k < s0 + len(num): anarci[(r.id[1], r.id[2].strip())] = num[k - s0]
        order = [(r.id[1], r.id[2].strip()) for r in rr if (r.id[1], r.id[2].strip()) in anarci]
        # deposited built ANARCI positions
        dep_built = {(x[4], x[5]) for x in res if x[4] is not None}
        keep_dep = [k for k in order if anarci[k] in dep_built]
        dep_last = max(x[4] for x in res if x[4] is not None)
        keep_end = [k for k in order if anarci[k][0] <= dep_last]
        for tag, keep in [('pred_full', order), ('pred_same_end', keep_end), ('pred_same_built', keep_dep)]:
            unobs = [(k, A3.get(chobj[(' ', k[0], k[1] or ' ')].get_resname(), 'X')) for k in order if k not in set(keep)]
            sab = run_sabr(st, model, chobj, pc, ct, set(keep), unobs, True)
            out.append(summarise(tag, 'P', p, c, anarci, sab, keep, dict(dep_last=dep_last)))
    except Exception as e:
        out.append(dict(exp='pred', pdb=p, chain=c, note=f'error {type(e).__name__}: {str(e)[:150]}'))
    return out


if __name__ == '__main__':
    from multiprocessing import Pool
    T = pd.read_csv('/home/user/SAbR/register_error_analysis/retest/deeper/case_files_DE/tables/independent_check_DE_units.csv')
    dp = T[T['mode'] == 'Dplus_CDR3short']
    short = dp[dp.model_ends_le121]; long_ = dp[~dp.model_ends_le121]
    S = pd.read_csv(f'{D}/cache/case_files_DE/struct_sample.csv')
    X = pd.read_pickle(f'{D}/cache/case_files_DE/X_struct.pkl')
    cl = X[X.group.isin(['matched_clean', 'within_paper_clean']) & (X.last_built_imgt >= 125)].drop_duplicates(['pdb', 'chain'])
    # choose up to 25 per chain type, prefer entries with a cached CIF (keeps downloads small)
    cached = lambda p: any(os.path.exists(f'{d}/{p}{e}') for d in CIFDIRS for e in ('.cif.gz', '.cif'))
    cl = cl.assign(cached=[cached(p) for p in cl.pdb]).sort_values(['cached', 'pdb'], ascending=[False, True])
    rng = np.random.default_rng(11)
    pick = []
    for ct, g in cl[cl.cached].groupby('ctype'):
        g = g.drop_duplicates('pdb')
        pick.append(g.sample(min(len(g), 25), random_state=int(rng.integers(1e9))))
    clean = pd.concat(pick)
    keys = set(zip(short.pdb, short.chain)) | set(zip(long_.pdb, long_.chain)) | set(zip(clean.pdb, clean.chain))
    rec = sabr_records(keys)
    print('records', len(rec), 'of', len(keys), flush=True)
    ctype = {(p, c): ct for p, c, ct in zip(T.pdb, T.chain, T.ctype)}
    ctype.update({(p, c): ct for p, c, ct in zip(clean.pdb, clean.chain, clean.ctype)})
    jobs = {}
    TR = [117, 118, 119, 120, 121, 122, 123, 125, 'full', '118nometa']
    for p, c in zip(short.pdb, short.chain):
        if (p, c) in rec: jobs.setdefault(p, []).append((c, rec[(p, c)]['chain_type'], 'R short D+', ['full'], rec[(p, c)]['residues']))
    for p, c in zip(long_.pdb, long_.chain):
        if (p, c) in rec: jobs.setdefault(p, []).append((c, rec[(p, c)]['chain_type'], 'L long D+', [118, 119, 'full'], rec[(p, c)]['residues']))
    for p, c in zip(clean.pdb, clean.chain):
        if (p, c) in rec: jobs.setdefault(p, []).append((c, rec[(p, c)]['chain_type'], 'T clean', TR, rec[(p, c)]['residues']))
    # predicted models of the short D+ antibodies
    units = json.load(open(f'{SP}/pred/full/units.json'))
    fvmap = {}
    for u in units:
        if (u['pdb'], u['chain']) in set(zip(short.pdb, short.chain)): fvmap[(u['pdb'], u['chain'])] = (u['fv_id'], u['mode'])
    pjobs = []
    for (p, c), (fv, mode) in fvmap.items():
        f = f'{SP}/pred/full/immunebuilder/{fv}.pdb'
        if not os.path.exists(f) or (p, c) not in rec: continue
        ct = rec[(p, c)]['chain_type']
        pc = 'H' if (mode == 'nanobody' or ct == 'H') else 'L'
        pjobs.append((p, c, ct, f, pc, rec[(p, c)]['residues']))
    print('deposited entries', len(jobs), 'pred jobs', len(pjobs), flush=True)
    t0 = time.time(); res = []
    with Pool(2) as pool:
        for i, o in enumerate(pool.imap_unordered(deposited_job, list(jobs.items()))):
            res += o
            if i % 10 == 0: print(i, round(time.time() - t0), flush=True)
        for o in pool.imap_unordered(predicted_job, pjobs): res += o
    O = pd.DataFrame(res)
    O.to_csv(f'{OUT}/tables/v01_sabr_runs.csv', index=False)
    print('done', round(time.time() - t0), flush=True)
