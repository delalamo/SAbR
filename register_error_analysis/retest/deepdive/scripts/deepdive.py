"""Deep dive: what do first-occurrence register errors at the same location have in common, versus every V domain
modelled correctly at that location?

Population: AlphaFold-independent SAbR-vs-ANARCI scan, single-domain antibody V chains at 3-5 A (15,865 chains).
Errors: disagreement runs not copied from an earlier structure of the same antibody and not known detector artifacts
(origins 1a, 1b, 2, 4 of ../unexplained/runs_origin_v2.csv excluded); one unit per antibody per paper.

Modes (by where the run starts, and the direction of the shift):
  A  FR1 N-terminal strand        start <= IMGT 10
  F  CDR1 boundaries              start 20-45
  B  CDR2 -> FR3 boundary         start 50-70
  C  FR3 (C'' / D strand)         start 71-100
  D+ CDR3 -> FR4, shift +1        start 101-118, delta > 0   (CDR3 built one residue short)
  D- CDR3 -> FR4, shift -1        start 101-118, delta < 0   (CDR3 built one residue long)
  E  FR4 bulge after W/F-G        start 119-122 (or origin 6)
Controls for a mode: units of the same chain type (H vs light) with no disagreement run of any origin overlapping the
mode's window. Main comparison restricted to cryo-EM (about 90% of the errors).

Features: local sequence (residue at each IMGT position of the window; neighbour self-similarity = how similar the
window is to itself shifted by one residue, BLOSUM62; bulky "landmark" residues; Gly/Pro/Ser), germline, CDR lengths,
modelling completeness (first modelled IMGT position, unmodelled residues in the window +-5 and in the domain, Fv-only
model), map (resolution, domain map-quality z, window map fit as percentile within the domain), process (starting
model, fitting protocol, software, year), context (role, SARS-CoV-2, nanobody).
Outputs: ../tables/deepdive_<mode>_features.csv, deepdive_<mode>_residues.csv, deepdive_members.csv, deepdive.log
"""
import json, os, re, numpy as np, pandas as pd, warnings
from scipy.stats import fisher_exact, mannwhitneyu
from statsmodels.stats.multitest import multipletests
from Bio.Align import substitution_matrices
warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
O = '/home/user/SAbR/register_error_analysis/retest/deepdive/tables'; os.makedirs(O, exist_ok=True)
log = open(f'{O}/deepdive.log', 'w')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n')
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 60)
B62 = substitution_matrices.load('BLOSUM62')
AA3 = dict(ALA='A', ARG='R', ASN='N', ASP='D', CYS='C', GLN='Q', GLU='E', GLY='G', HIS='H', ILE='I', LEU='L', LYS='K', MET='M',
           PHE='F', PRO='P', SER='S', THR='T', TRP='W', TYR='Y', VAL='V')

# ---------- chains and per-residue records
M = pd.read_pickle(f'{SP}/unexpl/master4.pkl')
VR = json.load(open(f'{SP}/unexpl/vregion.json')); G = json.load(open(f'{SP}/unexpl/germline.json'))
M['vseq'] = M.seq.map(VR).fillna(M.seq); M['j_gene'] = M.seq.map(lambda s: (G.get(s) or {}).get('j_gene'))
M['paper'] = M.cite_key.fillna(M.pdb); M['light'] = M.ctype.isin(['K', 'L'])
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
    LQ[(r['pdb'], r['chain'])] = ({am[int(k) - 1]: x for k, x in v.items() if x is not None and 0 < int(k) <= len(am)}, len(am))
AC = json.load(open(f'{SP}/sabr/anarci_cache.json'))
P('chains', len(M), 'with residue records', sum(k in REC for k in zip(M.pdb, M.chain)), 'with local map fit', sum(k in LQ for k in zip(M.pdb, M.chain)))

def numbering(seq, ct):
    """full (incl. unmodelled) IMGT numbering of the chain's V domain: list of ((pos, ins), aa)"""
    for d in AC.get(seq) or []:
        if (d['chain_type'] in 'KL') == (ct in 'KL'):
            return [((p, i), seq[s]) for s, p, i in d['pos']]
    return None
# ---------- runs
R = pd.read_csv(f'{SP}/unexpl/runs_origin_v2.csv'); R['cls'] = R.origin.str.split(' ').str[0]
first = R[~R.cls.isin(['1a', '1b', '2', '4'])].copy()
def mode(r):
    s = r.imgt0
    if r.cls == '6' or 119 <= s <= 122: return 'E'
    if s <= 10: return 'A'
    if 20 <= s <= 45: return 'F'
    if 50 <= s <= 70: return 'B'
    if 71 <= s <= 100: return 'C'
    if 101 <= s <= 118: return 'D+' if (r.delta or 0) > 0 else ('D-' if (r.delta or 0) < 0 else 'D0')
    return 'other'
first['mode'] = first.apply(mode, axis=1)
WIN = {'A': (1, 14), 'F': (20, 45), 'B': (50, 75), 'C': (71, 100), 'D+': (104, 124), 'D-': (104, 124), 'E': (115, 128)}

# ---------- per-chain features
def chain_feats(r, lo, hi):
    key = (r.pdb, r.chain); num = numbering(r.seq, r.ctype); res = REC.get(key)
    f = {}
    if num:
        w = [(p, a) for p, a in num if lo <= p[0] <= hi]
        s = ''.join(a for _, a in w)
        f['win_seq'] = s
        f['nbr_sim'] = np.mean([B62[a][b] for a, b in zip(s, s[1:])]) if len(s) > 1 else np.nan
        f['nbr2_sim'] = np.mean([B62[a][b] for a, b in zip(s, s[2:])]) if len(s) > 2 else np.nan
        f['landmarks'] = sum(c in 'WFYRKHM' for c in s); f['gly'] = s.count('G'); f['pro'] = s.count('P'); f['ser_thr'] = sum(c in 'ST' for c in s)
        f['pos_aa'] = {f'{p[0]}{p[1]}': a for p, a in w}
    if res:
        mod = {(x[4], x[5] or '') for x in res if x[4] is not None}
        aps = [x[4] for x in res if x[4] is not None]
        f['first_imgt'] = min(aps) if aps else np.nan
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
    return f
rows = []
for m, (lo, hi) in WIN.items():
    case_runs = first[first['mode'] == m]
    case_keys = set(zip(case_runs.pdb, case_runs.chain))
    over = R[(R.imgt0 <= hi) & (R.imgt1 >= lo)]; bad = set(zip(over.pdb, over.chain))
    light = m in ()  # placeholder
    sub = M.copy(); sub['case'] = [k in case_keys for k in zip(sub.pdb, sub.chain)]
    sub = sub[sub.case | ~pd.Series([k in bad for k in zip(sub.pdb, sub.chain)], index=sub.index)]
    sub = sub[sub.light.isin(set(M[[k in case_keys for k in zip(M.pdb, M.chain)]].light))]
    # one unit per antibody per paper: the case chain if any, else the first chain
    sub = sub.sort_values(['case'], ascending=False).drop_duplicates(['paper', 'vseq', 'case'])
    sub = sub[~(sub.duplicated(['paper', 'vseq'], keep=False) & ~sub.case)]
    for r in sub.itertuples():
        f = chain_feats(r, lo, hi)
        rows.append(dict(mode=m, case=r.case, pdb=r.pdb, chain=r.chain, ctype=r.ctype, method=r.method, res=r.res, year=r.year,
                         lq_z=r.lq_z, irm=r.irm, fit=r.fit, species=r.species, vfam=str(r.v_gene).split('-')[0].split('*')[0],
                         j_gene=str(r.j_gene).split('*')[0], v_id=r.v_id, L1=r.L1, L2=r.L2, L3=r.L3, L1_atyp=r.L1_atyp, L2_atyp=r.L2_atyp,
                         L3_atyp=r.L3_atyp, vhh=r.vhh, role=r.role, sars2=r.sars2, aux=r.aux_reagent, last_author=r.last_author, paper=r.paper,
                         sw_Coot=r.sw_Coot, sw_ISOLDE=r.sw_ISOLDE, sw_Rosetta=r.sw_Rosetta, sw_Phenix=r.sw_Phenix, **{k: v for k, v in f.items()}))
    P(f'mode {m}: case units {int(sub.case.sum())}, control units {int((~sub.case).sum())}')
D = pd.DataFrame(rows); D.drop(columns=['pos_aa']).to_pickle(f'{SP}/unexpl/deepdive_units.pkl'); D.to_pickle(f'{SP}/unexpl/deepdive_units_full.pkl')
P('built unit table', D.shape)
