"""Is there an algorithmic signature behind the CDR3-FR4 (and other loop-boundary) register slips?

Tests on the AlphaFold-independent 3-5 A scan (15,865 single-domain V chains, 666 classified runs; ../unexplained):
 1. Flank asymmetry. If chains are traced/extended N->C, a residue miscounted in a poorly resolved loop is pushed into
    the strand AFTER the loop (C-flank); built C->N, into the strand before it (N-flank). For each CDR, count runs on the
    N-flank vs the C-flank. CDR1 and CDR3 have a disulfide Cys (23, 104) on the N-flank; CDR2 has no Cys anchor, so it
    is the cleaner test of direction.
 2. Shift sign on each flank: on the C-flank, delta (SAbR - ANARCI) = +1 means the loop was built with one residue too
    few (the last loop residue sits in the first strand slot); on the N-flank, the meaning is reversed.
 3. Starting model and software: CDR3-FR4 junction error rate by recorded starting-model type (none / PDB template /
    AlphaFold / other in-silico) and by recorded software, EM chains, adjusted for period, resolution and local map
    quality.
 4. Template CDR3 length: for chains whose starting model is a PDB entry, the closest antibody domain of the same type
    in that entry (SEQRES + ANARCI). Is the junction error more common when the template's CDR3 length differs from
    the target's, and does the shift sign follow the length difference?
"""
import json, gzip, numpy as np, pandas as pd, statsmodels.formula.api as smf, warnings
from anarci import run_anarci
from Bio import Align
warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
O = '/home/user/SAbR/register_error_analysis/retest/junction/tables'
log = open(f'{O}/mechanism.log', 'w')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n')
pd.set_option('display.width', 250)
M = pd.read_pickle(f'{SP}/unexpl/master4.pkl'); VR = json.load(open(f'{SP}/unexpl/vregion.json'))
M['v'] = M.seq.map(VR).fillna(M.seq)
R = pd.read_csv(f'{SP}/unexpl/runs_origin_v2.csv'); R['cls'] = R.origin.str.split(' ').str[0]
# exclude known artifacts (HIV FR3-insertion bnAbs, FR3 convention region) and chain-end runs from the flank test
Rf = R[~R.cls.isin(['1a', '4', '3'])].copy()
flanks = {'CDR1': ((18, 26), (39, 47)), 'CDR2': ((47, 55), (66, 74)), 'CDR3': ((96, 104), (118, 128))}
P('== 1. Which side of each CDR do the register runs sit on? (runs; excl. HIV/FR3-convention artifacts and chain-end/gap runs)')
rows = []
for cdr, ((n0, n1), (c0, c1)) in flanks.items():
    for side, (a, b) in (('N-flank (before loop)', (n0, n1)), ('C-flank (after loop)', (c0, c1))):
        s = Rf[(Rf.imgt0 <= b) & (Rf.imgt1 >= a)]
        ev = s.drop_duplicates(['paper', 'seqkey', 'imgt0', 'delta'])
        sg = s.delta.dropna()
        rows.append(dict(CDR=cdr, side=side, runs=len(s), events=len(ev), plus=int((sg > 0).sum()), minus=int((sg < 0).sum())))
T1 = pd.DataFrame(rows); P(T1.to_string(index=False)); T1.to_csv(f'{O}/flank_asymmetry.csv', index=False)
ev_all = R.drop_duplicates(['paper', 'seqkey', 'imgt0', 'delta'])
P('\nsame, all runs incl. chain-end/gap ones:')
for cdr, ((n0, n1), (c0, c1)) in flanks.items():
    n = R[(R.imgt0 <= n1) & (R.imgt1 >= n0) & ~R.cls.isin(['1a', '4'])]; c = R[(R.imgt0 <= c1) & (R.imgt1 >= c0) & ~R.cls.isin(['1a', '4'])]
    P(f'  {cdr}: N-flank {len(n)} runs / C-flank {len(c)} runs')

# 3. starting model / software (V-chain level, EM)
P('\n== 3. CDR3-FR4 junction error rate by recorded starting model and software (EM V chains)')
J = set(zip(R[R.cls == '5'].pdb, R[R.cls == '5'].chain)); B = set(zip(R[R.cls == '6'].pdb, R[R.cls == '6'].chain))
M['junction'] = [k in J for k in zip(M.pdb, M.chain)]; M['bulge'] = [k in B for k in zip(M.pdb, M.chain)]
M['cdr3fr4'] = M.junction | M.bulge
E = M[(M.method == 'EM') & M.res.notna()].copy(); E['y'] = E.cdr3fr4.astype(int)
E['period'] = pd.cut(E.year, [0, 2019, 2021, 2030], labels=['<=2019', '2020-21', '>=2022']).astype(str)
E['lqz'] = E.lq_z.fillna(0)
g = E.groupby('irm').agg(chains=('pdb', 'size'), junction=('junction', 'sum'), bulge=('bulge', 'sum'))
g['pct_cdr3fr4'] = (100 * (g.junction + g.bulge) / g.chains).round(2); P(g.to_string())
try:
    m = smf.logit("y ~ C(irm, Treatment('none recorded')) + C(period) + res + lqz", data=E).fit(disp=0, cov_type='cluster', cov_kwds={'groups': pd.factorize(E.pdb)[0]})
    ci = np.exp(m.conf_int()); P('adjusted OR (cluster by entry):')
    for k in m.params.index:
        if 'irm' in k: P(f'  {k.split("T.")[-1][:-1]}: OR {np.exp(m.params[k]):.2f} ({ci.loc[k,0]:.2f}-{ci.loc[k,1]:.2f}) p={m.pvalues[k]:.3g}')
except Exception as e: P('model failed', e)
sw = [c for c in E.columns if c.startswith('sw_')]
rows = []
for c in sw:
    x = E[c].fillna(False).astype(bool)
    rows.append(dict(software=c[3:], chains_recording=int(x.sum()), pct_cdr3fr4_with=round(100 * E.cdr3fr4[x].mean(), 2) if x.sum() else np.nan,
                     pct_cdr3fr4_without=round(100 * E.cdr3fr4[~x].mean(), 2)))
P(pd.DataFrame(rows).to_string(index=False))
if 'fit' in E:
    P('\nby EM fitting protocol:'); P(E.groupby('fit').agg(chains=('pdb', 'size'), pct=('cdr3fr4', lambda x: round(100 * x.mean(), 2))).to_string())

# 4. template CDR3 length
P('\n== 4. Starting-model (PDB template) CDR3 length vs target CDR3 length')
SM = {}
for line in open(f'{SP}/retest/modelling.jsonl'):
    e = json.loads(line); irm = e.get('pdbx_initial_refinement_model') or []
    ids = sorted({(x.get('accession_code') or '').lower() for x in irm if (x.get('source_name') or '').upper() == 'PDB' and x.get('accession_code')})
    if ids: SM[e['rcsb_id'].lower()] = ids
X = M[M.pdb.isin(SM)].copy(); X['sm'] = X.pdb.map(SM)
need = {t for l in X.sm for t in l}
seqs = {}
with gzip.open('/tmp/supp/pdbwide/pdb_seqres.txt.gz', 'rt') as f:
    h = None
    for line in f:
        if line[0] == '>': h = line[1:7] if line[1:5].lower() in need and 'mol:protein' in line else None
        elif h: seqs.setdefault(h[:4].lower(), set()).add(line.strip())
useq = sorted({s for v in seqs.values() for s in v if len(s) >= 80})
P('template entries', len(need), 'with sequences', len(seqs), 'unique chains', len(useq))
TD = {}
for k in range(0, len(useq), 1000):
    chunk = [(str(i), s) for i, s in enumerate(useq[k:k + 1000])]
    _, num, det, _ = run_anarci(chunk, scheme='imgt', ncpu=4)
    for (i, s), n, d in zip(chunk, num, det):
        if not d: continue
        for dom, dd in zip(n, d):
            if dd['chain_type'] not in 'HKL': continue
            numbered = dom[0]
            l3 = sum(1 for (p, ic), aa in numbered if 105 <= p <= 117 and aa != '-')
            TD.setdefault(s, []).append((dd['chain_type'], s[dd['query_start']:dd['query_end'] + 1], l3))
al = Align.PairwiseAligner(mode='global', match_score=1, mismatch_score=0, open_gap_score=-1, extend_gap_score=-0.5)
rows = []
tl3 = {}
for r in X.itertuples():
    best = None
    for t in r.sm:
        for s in seqs.get(t, []):
            for ct, v, l3 in TD.get(s, []):
                if (ct == 'H') != (r.ctype == 'H'): continue
                idn = al.score(r.v, v) / max(len(r.v), len(v))
                if best is None or idn > best[0]: best = (idn, t, l3, v == r.v)
    if best: rows.append(dict(pdb=r.pdb, chain=r.chain, ctype=r.ctype, EM=r.method == 'EM', year=r.year, cdr3_len=r.cdr3_len, t_identity=best[0],
                              template=best[1], t_cdr3_len=best[2], identical=best[3], junction=r.junction, bulge=r.bulge))
D = pd.DataFrame(rows); D['dl'] = D.cdr3_len - D.t_cdr3_len
D['rel'] = np.select([D.identical, D.dl == 0, D.dl.abs() == 1, D.dl.abs() >= 2], ['identical antibody', 'other Ab, same CDR3 length', 'CDR3 length differs by 1', 'differs by >=2'], 'other')
D.to_csv(f'{O}/template_cdr3.csv', index=False)
G = D[D.EM].groupby('rel').agg(chains=('pdb', 'size'), junction=('junction', 'sum'), bulge=('bulge', 'sum'))
G['pct_junction'] = (100 * G.junction / G.chains).round(2); P('EM chains whose starting model contains an antibody domain of the same type:'); P(G.to_string())
jj = R[R.cls == '5'][['pdb', 'chain', 'delta']].merge(D, on=['pdb', 'chain'])
jj = jj[~jj.identical]
P('\njunction runs with a non-identical template: sign of shift vs (target - template) CDR3 length')
P(pd.crosstab(np.sign(jj.delta), np.sign(jj.dl), rownames=['shift sign'], colnames=['sign(target L3 - template L3)']).to_string())
log.close()
