"""V02: direct causal test of the claimed SAbR chain-end artifact.
Take correctly modelled chains (clean cryo-EM controls of the investigator's s02 sample: no run of any kind in the
entry, model built to >= IMGT 125), and computationally delete every residue after ANARCI IMGT position T
(T = 117 ... 124), i.e. create a model that stops at / just after W/F118 while the coordinates of everything that
remains are unchanged and known to be in register. Rerun SAbR exactly as the scan did (FAST mode: chain type fixed,
chain trimmed to the V domain +-3 author numbers, unobserved-residue metadata from the mmCIF) and compare to the stored
sequence-based ANARCI numbers.
If SAbR has a chain-end artifact, truncation alone will create D+-like runs (SAbR > ANARCI) at the C-terminus.
Second arm: the flagged chains themselves (all D+ units with model ending <= IMGT 121, plus long D+ positives truncated
at 118) to check that the scan result is reproduced and to see what the detector does to genuine slips at a chain end.
Output: tables/v02_truncation_runs.csv, tables/v02_truncation_summary.csv
"""
import os, sys, io, gzip, json, time, warnings, logging, numpy as np, pandas as pd
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[_v] = "1"
warnings.filterwarnings('ignore'); logging.disable(logging.WARNING)
from multiprocessing import Pool
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
D = f'{SP}/deeper'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/case_files_DE_DE1_cterm_artifact_confounding'
CIFDIRS = [f'{D}/cache/case_files_DE/cif', f'{D}/cache/struct_unusual/cif', f'{D}/cache/case_files_ABCF/cif', f'{SP}/sabr/dl']
CUTS = [117, 118, 119, 120, 121, 122, 124]


def cif_path(p):
    for d in CIFDIRS:
        for ext in ('.cif.gz', '.cif'):
            f = f'{d}/{p}{ext}'
            if os.path.exists(f) and os.path.getsize(f) > 1000: return f
    return None


def runs_of(flags):
    out, cur = [], []
    for k, f in enumerate(flags):
        if f: cur.append(k)
        else:
            if len(cur) >= 3: out.append((cur[0], cur[-1]))
            cur = []
    if len(cur) >= 3: out.append((cur[0], cur[-1]))
    return out


def job(a):
    p, chains = a  # chains: list of (chain, ctype, residues, label)
    from Bio.PDB import MMCIFParser
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict
    from Bio.PDB.Structure import Structure
    from Bio.PDB.Model import Model
    from sabr import renumber_structure
    out = []
    f = cif_path(p)
    if f is None:
        return [dict(pdb=p, chain=c, label=l, status='no cif') for c, _, _, l in chains]
    txt = gzip.open(f, 'rt').read() if f.endswith('.gz') else open(f).read()
    md = MMCIF2Dict(io.StringIO(txt))
    st = MMCIFParser(QUIET=True).get_structure(p, io.StringIO(txt))
    model = next(iter(st))
    for c, ct, res, label in chains:
        try:
            if c not in model:
                out.append(dict(pdb=p, chain=c, label=label, status='chain missing')); continue
            v = [x for x in res if x[4] is not None]
            an = {(x[0], x[1]): (x[4], x[5] or '') for x in v}
            stored = {(x[0], x[1]): (x[6], x[7] or '') for x in res if x[6] is not None}
            nums = [x[0] for x in v]
            lo, hi = min(nums) - 3, max(nums) + 3
            chobj = model[c]
            # author ids in chain order of the V-domain residues (ANARCI-numbered)
            order = [(r.id[1], r.id[2].strip()) for r in chobj if r.id[0] == ' ' and lo <= r.id[1] <= hi]
            pos_of = {k: i for i, k in enumerate(order)}
            for cut in ['full'] + CUTS:
                if cut == 'full':
                    keep_last = len(order) - 1
                else:
                    # last residue (chain order) whose ANARCI position <= cut
                    idx = [pos_of[k] for k, a_ in an.items() if k in pos_of and a_[0] <= cut]
                    if not idx: continue
                    keep_last = max(idx)
                    if an.get(order[keep_last], (0,))[0] != cut and cut != 117:
                        # the model does not contain this exact position (e.g. built gap); still record
                        pass
                keep = set(order[:keep_last + 1])
                tst = Structure(p); tst.header = dict(getattr(st, 'header', {}) or {})
                tm = Model(model.id, model.serial_num); tst.add(tm); tc = chobj.copy()
                for r in list(tc):
                    if r.id[0] != ' ' or not (lo <= r.id[1] <= hi) or (r.id[1], r.id[2].strip()) not in keep:
                        tc.detach_child(r.id)
                tm.add(tc)
                t0 = time.time()
                new = renumber_structure(tst, chain=c, chain_type=ct, mmcif_dict=md)
                dt = time.time() - t0
                rows = []
                for o, n in zip(list(tc), list(next(iter(new))[c])):
                    k = (o.id[1], o.id[2].strip())
                    if k not in an: continue
                    rows.append((k, o.get_resname(), an[k], (n.id[1], n.id[2].strip())))
                flags = [a_ != s_ for _, _, a_, s_ in rows]
                rr = runs_of(flags)
                last = rows[-1]
                cterm = [(i, j) for i, j in rr if j >= len(rows) - 2 and rows[j][3][0] > rows[j][2][0] and rows[j][2][0] >= 104]
                cterm_any = [(i, j) for i, j in rr if j >= len(rows) - 2 and rows[j][2][0] >= 104]
                w = [r_ for r_ in rows if r_[2] == (118, '')]
                rec = dict(pdb=p, chain=c, ctype=ct, label=label, cut=str(cut), status='ok', n_res=len(rows),
                           last_anarci=last[2][0], last_aa=last[1], last_sabr=last[3][0], last_sabr_ins=last[3][1],
                           w118_aa=w[0][1] if w else None, w118_sabr=w[0][3][0] if w else None,
                           n_runs=len(rr), cterm_dplus_run=bool(cterm), cterm_any_run=bool(cterm_any),
                           cterm_run=';'.join(f'{rows[i][2][0]}{rows[i][2][1]}-{rows[j][2][0]}{rows[j][2][1]}(d{rows[j][3][0]-rows[j][2][0]:+d})' for i, j in cterm_any),
                           all_runs=';'.join(f'{rows[i][2][0]}{rows[i][2][1]}-{rows[j][2][0]}{rows[j][2][1]}' for i, j in rr),
                           sabr_s=round(dt, 2))
                if cut == 'full':
                    # agreement with the stored scan output (sanity check of the pipeline)
                    rec['match_stored'] = float(np.mean([stored.get(k) == s_ for k, _, _, s_ in rows]))
                out.append(rec)
        except Exception as ex:
            out.append(dict(pdb=p, chain=c, label=label, status=f'error {type(ex).__name__}: {str(ex)[:120]}'))
    return out


if __name__ == '__main__':
    CH = pd.read_csv(f'{D}/chains.csv.gz')
    S = pd.read_csv(f'{D}/cache/case_files_DE/struct_sample.csv')
    I = pd.read_csv('/home/user/SAbR/register_error_analysis/retest/deeper/case_files_DE/tables/independent_check_DE_units.csv')
    SF = pd.read_csv('/home/user/SAbR/register_error_analysis/retest/deeper/case_files_DE/tables/struct_features.csv')
    ctmap = dict(zip(zip(CH.pdb, CH.chain), CH.ctype))
    vhh = dict(zip(zip(CH.pdb, CH.chain), CH.vhh))
    clean = S[S.group == 'matched_clean'].merge(SF[['pdb', 'chain', 'last_built_imgt']], on=['pdb', 'chain'])
    clean = clean[clean.last_built_imgt >= 125]
    clean['ct'] = np.where(clean.vhh == 1, 'VHH', clean.ctype)
    rng = np.random.default_rng(11)
    quota = {'H': 30, 'K': 20, 'L': 15, 'VHH': 15}
    pick = []
    for ct, g in clean.groupby('ct'):
        g = g.drop_duplicates('paper')  # one chain per paper -> independent
        pick.append(g.sample(min(len(g), quota[ct]), random_state=int(rng.integers(1e9))))
    clean = pd.concat(pick)
    sel = [(r.pdb, r.chain, 'clean_' + r.ct) for r in clean.itertuples()]
    dp = I[I['mode'] == 'Dplus_CDR3short']
    for r in dp.itertuples():
        sel.append((r.pdb, r.chain, 'Dplus_short_' + r.confirm.replace(' ', '_') if r.model_ends_le121 else 'Dplus_long_' + r.confirm.replace(' ', '_')))
    keys = {(p, c) for p, c, _ in sel}
    rec = {}
    for f in ['validation', 'snapshot_rest', 'new']:
        for l in open(f'{SP}/sabr/out/{f}.jsonl'):
            r = json.loads(l)
            if r.get('status') == 'ok' and (r['pdb'], r['chain']) in keys: rec[(r['pdb'], r['chain'])] = (r['chain_type'], r['residues'])
    jobs = {}
    for p, c, lab in sel:
        if (p, c) in rec:
            jobs.setdefault(p, []).append((c, rec[(p, c)][0], rec[(p, c)][1], lab))
    print('entries', len(jobs), 'chains', sum(len(v) for v in jobs.values()), flush=True)
    res = []; t0 = time.time()
    with Pool(2) as pool:
        for i, o in enumerate(pool.imap_unordered(job, list(jobs.items()), chunksize=1)):
            res += o
            if i % 20 == 0: print(i, round(time.time() - t0), flush=True)
    T = pd.DataFrame(res)
    T.to_csv(f'{OUT}/tables/v02_truncation_runs.csv', index=False)
    ok = T[T.status == 'ok'].copy()
    ok['grp'] = np.where(ok.label.str.startswith('clean'), 'clean (all types)', ok.label)
    summ = ok.groupby(['grp', 'cut']).agg(n=('pdb', 'size'), cterm_dplus=('cterm_dplus_run', 'sum'), cterm_any=('cterm_any_run', 'sum'),
                                          w118_numbered_119plus=('w118_sabr', lambda s: int((s >= 119).sum()))).reset_index()
    summ['frac_cterm_dplus'] = summ.cterm_dplus / summ.n
    summ.to_csv(f'{OUT}/tables/v02_truncation_summary.csv', index=False)
    pd.set_option('display.width', 250); pd.set_option('display.max_rows', 300)
    print(summ.to_string())
    cl = ok[ok.label.str.startswith('clean')]
    print(cl.groupby(['label', 'cut']).cterm_dplus_run.agg(['size', 'sum']).unstack(0))
    print('baseline match with stored scan output:', cl[cl.cut == 'full'].match_stored.describe())
