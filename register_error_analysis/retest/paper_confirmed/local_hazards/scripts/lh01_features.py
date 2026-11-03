"""Local-hazard features at the slipped strand, for the paper-confirmed re-test.

Reuses chain_feats() from retest/deepdive/scripts/deepdive.py unchanged (same windows, same definitions of unbuilt
residues near the window, first built IMGT residue, window map-fit percentile within the domain, Fv-only model,
per-position residues). Computes it for
  * every case run in runs_all.csv whose mode is A/F/B/C/D+/D-/E (paper-confirmed cases, copies, rest of the deeper
    first-occurrence set) at the window of its mode, and
  * every chain with no SAbR/ANARCI run of any origin (err_any False) at all 7 windows (controls; the paper controls
    are the in-window subset of these).
Output: ../tables/lh_frame.pkl.gz (one row per chain x window, role = case/control) -- the reusable frame.
"""
import json, numpy as np, pandas as pd, warnings, multiprocessing as mp
from Bio.Align import substitution_matrices
warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
O = '/home/user/SAbR/register_error_analysis/retest/paper_confirmed/local_hazards/tables'
B62 = substitution_matrices.load('BLOSUM62')
C = pd.read_csv(f'{SP}/deeper/chains.csv.gz', low_memory=False)
REC = {}
for fn in ['new.jsonl', 'snapshot_rest.jsonl', 'validation.jsonl']:
    for line in open(f'{SP}/sabr/out/{fn}'):
        r = json.loads(line)
        if r.get('status') == 'ok' and r.get('residues'): REC[(r['pdb'], r['chain'])] = r['residues']
LQ = {}
for line in open(f'{SP}/retest/local_all.jsonl'):
    r = json.loads(line); feats = r.get('feats') or {}
    key = 'Q_SCORE' if 'Q_SCORE' in feats else ('RSCC' if 'RSCC' in feats else None)
    if key is None: continue
    am = r['auth_map']; v = feats[key]
    LQ[(r['pdb'], r['chain'])] = ({am[int(k) - 1]: x for k, x in v.items() if x is not None and 0 < int(k) <= len(am)}, len(am), key)
AC = json.load(open(f'{SP}/sabr/anarci_cache.json'))
print('loaded', len(REC), len(LQ), flush=True)

def numbering(seq, ct):
    for d in AC.get(seq) or []:
        if (d['chain_type'] in 'KL') == (ct in 'KL'):
            return [((p, i), seq[s]) for s, p, i in d['pos']]
    return None

def chain_feats(pdb, chain, seq, ct, lo, hi):
    """verbatim logic of deepdive.py chain_feats (plus which local-fit metric was used)"""
    key = (pdb, chain); num = numbering(seq, ct); res = REC.get(key)
    f = {}
    if num:
        w = [(p, a) for p, a in num if lo <= p[0] <= hi]
        s = ''.join(a for _, a in w)
        f['win_seq'] = s
        f['pos_aa'] = {f'{p[0]}{p[1]}': a for p, a in w}
    if res:
        mod = {(x[4], x[5] or '') for x in res if x[4] is not None}
        aps = [x[4] for x in res if x[4] is not None]
        f['first_imgt'] = min(aps) if aps else np.nan
        f['last_imgt'] = max(aps) if aps else np.nan
        f['nterm_missing'] = (min(aps) > 3) if aps else np.nan
        if num:
            allp = [p for p, _ in num]; f['unmod_domain'] = sum(p not in mod for p in allp)
            f['unmod_near'] = sum(p not in mod for p in allp if lo - 5 <= p[0] <= hi + 5)
        lq = LQ.get(key)
        if lq:
            vals = {}
            for x in res:
                a = f'{x[0]}{x[1] or ""}'
                if a in lq[0] and x[4] is not None: vals[x[4]] = lq[0][a]
            if len(vals) >= 30:
                v = pd.Series(vals); rk = v.rank(pct=True)
                inw = [p for p in v.index if lo <= p <= hi]
                f['win_fit_pct'] = rk[inw].mean() if inw else np.nan
                f['win_fit'] = v[inw].mean() if inw else np.nan
            f['fv_only'] = lq[1] <= len(res) + 15
            f['fit_metric'] = lq[2]
    return f

WIN = {'A': (1, 14), 'F': (20, 45), 'B': (50, 75), 'C': (71, 100), 'D+': (104, 124), 'D-': (104, 124), 'E': (115, 128)}
MODEMAP = {'A_FR1Nterm': 'A', 'F_CDR1': 'F', 'B_CDR2FR3': 'B', 'C_FR3': 'C', 'Dplus_CDR3short': 'D+', 'Dminus_CDR3long': 'D-', 'E_FR4bulge': 'E'}
SEQ = dict(zip(zip(C.pdb, C.chain), zip(C.seq, C.ctype)))

def work(job):
    role, pdb, chain, m, extra = job
    seq, ct = SEQ[(pdb, chain)]
    lo, hi = WIN[m]
    d = dict(role=role, pdb=pdb, chain=chain, mode=m, **extra)
    d.update(chain_feats(pdb, chain, seq, ct, lo, hi))
    return d

if __name__ == '__main__':
    R = pd.read_csv(f'{SP}/deeper/runs_all.csv'); R['m'] = R['mode'].map(MODEMAP)
    R = R[R.m.notna()]
    jobs = [('case', r.pdb, r.chain, r.m, dict(imgt0=r.imgt0, imgt1=r.imgt1, delta=r.delta, cls=str(r.cls), copied_or_artifact=bool(r.copied_or_artifact)))
            for r in R.itertuples()]
    clean = C[~C.err_any.astype(bool)]
    jobs += [('control', p, c, m, {}) for p, c in zip(clean.pdb, clean.chain) for m in ['A', 'F', 'B', 'C', 'D+', 'D-', 'E'] if m != 'D-']
    print('jobs', len(jobs), flush=True)
    with mp.Pool(4) as pool:
        rows = pool.map(work, jobs, chunksize=500)
    F = pd.DataFrame(rows)
    # D- shares the D+ window: copy D+ control rows
    dc = F[(F.role == 'control') & (F['mode'] == 'D+')].copy(); dc['mode'] = 'D-'
    F = pd.concat([F, dc], ignore_index=True)
    F.to_pickle(f'{O}/lh_frame.pkl.gz')
    print(F.groupby(['role', 'mode']).size())
