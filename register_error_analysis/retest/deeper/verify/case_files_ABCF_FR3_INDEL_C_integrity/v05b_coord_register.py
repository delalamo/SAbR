"""Detector-independent check: is each FR3-indel 'error' run a real register shift in the coordinates?
For each EM mode-C FR3-indel error chain that has another structure of the identical V region, superimpose the V domain
(CA, excluding the run window +-2) and, for every residue in the run window, compare the distance of query CA(i) to the
reference CA(i) with the distance to reference CA(i-1)/CA(i+1). A register shift of one residue on a strand puts
CA_q(i) ~0.5-1.5 A from CA_r(i+-1) and ~3.5 A from CA_r(i); a correctly registered (or merely differently numbered)
model puts CA_q(i) closest to CA_r(i).
Negative control: 2x7l A vs 5dhv A (a confirmed numbering artifact: both disagree identically with ANARCI).
Output tables/v05b_coord_register.csv (per residue), tables/v05b_coord_summary.csv"""
import json, os, numpy as np, pandas as pd, warnings
from Bio.PDB import MMCIFParser
from Bio.Align import PairwiseAligner
from Bio.SVDSuperimposer import SVDSuperimposer
warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/case_files_ABCF_FR3_INDEL_C_integrity'
DIRS = [f'{SP}/sabr/dl', f'{SP}/deeper/cache/case_files_ABCF/hires_cif', f'{SP}/deeper/cache/verify_fr3_integrity/cif']
A3 = dict(ALA='A', ARG='R', ASN='N', ASP='D', CYS='C', GLN='Q', GLU='E', GLY='G', HIS='H', ILE='I', LEU='L', LYS='K', MET='M', PHE='F',
          PRO='P', SER='S', THR='T', TRP='W', TYR='Y', VAL='V', MSE='M')
def path(p):
    for d in DIRS:
        f = f'{d}/{p}.cif'
        if os.path.exists(f): return f
    raise FileNotFoundError(p)
def chain_res(p, c):
    st = MMCIFParser(QUIET=True).get_structure(p, path(p)); ch = next(iter(st))[c]
    res = [r for r in ch if r.id[0] in (' ', 'H_MSE') and r.get_resname() in A3 and 'CA' in r]
    return res, ''.join(A3[r.get_resname()] for r in res)
REC = {}
for fn in ['validation', 'new', 'snapshot_rest']:
    for line in open(f'{SP}/sabr/out/{fn}.jsonl'):
        x = json.loads(line)
        if x.get('status') == 'ok': REC.setdefault((x['pdb'], x['chain']), x['residues'])
RF = pd.read_csv(f'{SP}/deeper/runs_all.csv')
PAIRS = [('7te9', 'H', '7te4', 'B'), ('7te9', 'H', '7te4', 'H'), ('7teb', 'H', '7te4', 'B'), ('8xlk', 'K', '7te4', 'B'), ('8xlk', 'K', '7te9', 'H'),
         ('7y0o', 'H', '7y0n', 'C'), ('7y0o', 'H', '7y0n', 'E'), ('9yfu', 'N', '9jrt', 'N'), ('9yfu', 'N', '8tb7', 'N'), ('7m3l', 'H2', '7m3n', 'H'),
         ('2x7l', 'A', '5dhv', 'A')]
al = PairwiseAligner(); al.mode = 'global'; al.match_score = 2; al.mismatch_score = -1; al.open_gap_score = -5; al.extend_gap_score = -0.5
rows, summ = [], []
for qp, qc, rp, rc in PAIRS:
    try:
        qres, qs = chain_res(qp, qc); rres, rs = chain_res(rp, rc)
    except Exception as e:
        print('skip', qp, qc, rp, rc, repr(e)); continue
    aln = al.align(qs, rs)[0]
    m = {}
    for (a0, a1), (b0, b1) in zip(*aln.aligned):
        for k in range(a1 - a0): m[a0 + k] = b0 + k
    rec = REC.get((qp, qc)); runs = RF[(RF.pdb == qp) & (RF.chain == qc) & (RF['mode'] == 'C_FR3')]
    if rec is None or not len(runs): print('no record/run', qp, qc); continue
    r0 = runs.iloc[0]
    auth = {(r[0], (r[1] or '').strip()): (r[4], r[6]) for r in rec}
    qidx = {(r.id[1], r.id[2].strip()): i for i, r in enumerate(qres)}
    imgt_q = {qidx[k]: v for k, v in auth.items() if k in qidx}
    win = [i for i, (a, s) in imgt_q.items() if a is not None and r0.imgt0 <= a <= r0.imgt1]
    vdom = [i for i, (a, s) in imgt_q.items() if a is not None and a <= 128]
    fit = [i for i in vdom if i in m and (not win or (i < min(win) - 2 or i > max(win) + 2))]
    X = np.array([qres[i]['CA'].coord for i in fit]); Y = np.array([rres[m[i]]['CA'].coord for i in fit])
    sup = SVDSuperimposer(); sup.set(Y, X); sup.run(); rot, tran = sup.get_rotran()
    tq = lambda i: qres[i]['CA'].coord @ rot + tran
    rms = sup.get_rms()
    fr3 = [i for i in vdom if imgt_q[i][0] is not None and 66 <= imgt_q[i][0] <= 104 and i in m]
    for i in fr3:
        j = m[i]; d = {}
        for s in (-2, -1, 0, 1, 2):
            d[s] = float(np.linalg.norm(tq(i) - rres[j + s]['CA'].coord)) if 0 <= j + s < len(rres) else np.nan
        best = min(d, key=lambda s: d[s] if d[s] == d[s] else 1e9)
        rows.append(dict(query=f'{qp}_{qc}', ref=f'{rp}_{rc}', aa=qs[i], anarci_imgt=imgt_q[i][0], sabr_imgt=imgt_q[i][1], in_run=i in win,
                         d0=d[0], dm1=d[-1], dp1=d[1], dm2=d[-2], dp2=d[2], best_offset=best))
    R = pd.DataFrame([r for r in rows if r['query'] == f'{qp}_{qc}' and r['ref'] == f'{rp}_{rc}'])
    w = R[R.in_run]; o = R[~R.in_run]
    summ.append(dict(query=f'{qp}_{qc}', ref=f'{rp}_{rc}', run=f'{r0.imgt0}-{r0.imgt1} ({r0.delta:+.0f})' if r0.delta == r0.delta else f'{r0.imgt0}-{r0.imgt1}',
                     seq_identity=f'{sum(qs[i] == rs[j] for i, j in m.items())}/{len(m)}', fit_rmsd=round(rms, 2), n_run_res=len(w),
                     run_median_d0=round(w.d0.median(), 2), run_median_best_shift_d=round(w[['dm1', 'dp1']].min(1).median(), 2),
                     run_frac_closer_to_neighbour=round((w[['dm1', 'dp1']].min(1) < w.d0).mean(), 2),
                     run_best_offsets=''.join(f'{int(b):+d}' for b in w.best_offset), FR3_outside_median_d0=round(o.d0.median(), 2)))
    print(summ[-1], flush=True)
pd.DataFrame(rows).to_csv(f'{OUT}/tables/v05b_coord_register.csv', index=False)
S = pd.DataFrame(summ); S.to_csv(f'{OUT}/tables/v05b_coord_summary.csv', index=False)
pd.set_option('display.width', 250); print(S.to_string())
