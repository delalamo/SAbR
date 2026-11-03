"""Case selection for the detective case files (modes A_FR1Nterm, B_CDR2FR3, C_FR3, F_CDR1).

Cases: first-occurrence error entries (runs_first_occurrence.csv), excluding the 15 AL amyloid fibril entries (detector
artifact; ../pdb_meta). For every candidate entry/run we compute
  * run length and the map fit of the run's residues (EM Q-score / X-ray RSCC) as a percentile within the V domain;
  * unbuilt residues next to the run; first built IMGT position;
then choose ~6 per mode: one entry per paper and per lab, mostly EM, spread over years, preferring long runs with
poor map fit ("clearest").
Controls: for each case 1-2 entries with NO disagreement run in any chain (entries.csv err_any == False), same method,
same light/heavy chain class present, chosen in this order:
  (1) same antigen (UniProt accession) deposited within +-18 months by a different paper ("same antigen" control);
  (2) otherwise same target class, same deposition year, resolution within 0.4 A ("period" control);
  plus, where available, a "near-miss" control: a clean entry of the same period whose V domain shares the case's
  local hazard (mode A: N-terminus not built; B/C/F: unbuilt residues within the mode window).
Outputs: tables/cases.csv, tables/controls.csv, tables/candidates_<mode>.csv
"""
import json, numpy as np, pandas as pd, os
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
D = f'{SP}/deeper'; OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/case_files_ABCF'
T = f'{OUT}/tables'; os.makedirs(T, exist_ok=True)
AG = '/home/user/SAbR/register_error_analysis/retest/deeper/antigen_competition/tables'
amy = set(pd.read_csv('/home/user/SAbR/register_error_analysis/retest/deeper/pdb_meta/tables/amyloid_fibril_chains.csv').pdb.str.lower())
MODES = ['A_FR1Nterm', 'B_CDR2FR3', 'C_FR3', 'F_CDR1']
WIN = {'A_FR1Nterm': (1, 14), 'F_CDR1': (20, 45), 'B_CDR2FR3': (50, 75), 'C_FR3': (71, 100)}

R = pd.read_csv(f'{D}/runs_first_occurrence.csv'); R = R[~R.pdb.isin(amy) & R['mode'].isin(MODES)]
C = pd.read_csv(f'{D}/chains.csv.gz', low_memory=False)
E = pd.read_csv(f'{D}/entries.csv')
A = pd.read_csv(f'{AG}/pdbwide_ab_entries_antigen.csv.gz')
need = set(R.pdb) | set(E.pdb)
REC = {}
for fn in ['new.jsonl', 'snapshot_rest.jsonl', 'validation.jsonl']:
    for line in open(f'{SP}/sabr/out/{fn}'):
        x = json.loads(line)
        if x.get('residues'): REC[(x['pdb'], x['chain'])] = x['residues']
LQ = {}
for line in open(f'{SP}/retest/local_all.jsonl'):
    x = json.loads(line)
    if x['pdb'] not in need: continue
    f = x.get('feats') or {}; key = 'Q_SCORE' if 'Q_SCORE' in f else ('RSCC' if 'RSCC' in f else None)
    if key is None: continue
    am = x['auth_map']
    LQ[(x['pdb'], x['chain'])] = {am[int(k) - 1]: v for k, v in f[key].items() if v is not None and 0 < int(k) <= len(am)}

def site_fit(pdb, ch, lo_imgt, hi_imgt):
    """mean map fit of residues with ANARCI IMGT in [lo,hi], and their mean percentile within the V domain"""
    rec = REC.get((pdb, ch)); lq = LQ.get((pdb, ch))
    if not rec or not lq: return np.nan, np.nan, 0
    vals = {}
    for x in rec:
        a = f'{x[0]}{x[1] or ""}'
        if a in lq and x[4] is not None: vals[(x[4], x[5] or '', x[3])] = lq[a]
    if len(vals) < 30: return np.nan, np.nan, 0
    v = pd.Series(list(vals.values()), index=pd.MultiIndex.from_tuples(list(vals.keys())))
    rk = v.rank(pct=True); sel = [k for k in v.index if lo_imgt <= k[0] <= hi_imgt]
    if not sel: return np.nan, np.nan, 0
    return float(v[sel].mean()), float(rk[sel].mean()), len(sel)

def unbuilt_near(pdb, ch, lo, hi):
    rec = REC.get((pdb, ch))
    if not rec: return np.nan, np.nan
    aps = sorted({x[4] for x in rec if x[4] is not None})
    first = aps[0] if aps else np.nan
    # gaps in seq_index among residues in window+-5
    w = [x for x in rec if x[4] is not None and lo - 5 <= x[4] <= hi + 5]
    si = sorted(x[3] for x in w)
    gaps = sum(max(0, b - a - 1) for a, b in zip(si, si[1:]))
    return first, gaps

rows = []
for (pdb, ch, m), g in R.groupby(['pdb', 'chain', 'mode']):
    lo, hi = int(g.imgt0.min()), int(g.imgt1.max())
    fit, pct, n = site_fit(pdb, ch, lo, hi)
    first, gaps = unbuilt_near(pdb, ch, lo, hi)
    r0 = g.iloc[0]
    rows.append(dict(mode=m, pdb=pdb, chain=ch, ctype=r0.ctype, method=r0.method, res=r0.res, year=r0.year, deposit=r0.deposit,
                     last_author=r0.last_author, paper=r0.paper, imgt0=lo, imgt1=hi, delta=r0.delta, run_len=int(g.length.max()),
                     origin=r0.origin, irm=r0.irm, fit=r0.fit, lq_z=r0.lq_z, sars2=r0.sars2, role=r0.role, chain_desc=r0.chain_desc,
                     site_fit=fit, site_fit_pct=pct, n_site=n, first_imgt=first, gaps_near=gaps, spo=bool(g.same_paper_other_entry.any())))
K = pd.DataFrame(rows)
# one row per (mode, pdb): representative chain = poorest site fit percentile, then longest
K = K.sort_values(['mode', 'pdb', 'site_fit_pct', 'run_len'], ascending=[True, True, True, False])
KE = K.drop_duplicates(['mode', 'pdb']).copy()
KE['n_chains_err'] = K.groupby(['mode', 'pdb']).chain.transform('size').reindex(KE.index)
KE = KE.merge(A[['pdb', 'lab', 'ab_status', 'tgt_keys', 'tgt_fam', 'tgt_cls', 'tgt_name']], on='pdb', how='left')
# clarity score: long run (capped), poor fit, EM
KE['clarity'] = np.minimum(KE.run_len, 20) / 20 + (1 - KE.site_fit_pct.fillna(0.5)) + (KE.method == 'EM') * 0.5 \
    - (KE.run_len > 60) * 0.7  # whole-domain runs are a different phenomenon: de-prioritise
for m in MODES:
    KE[KE['mode'] == m].sort_values('clarity', ascending=False).to_csv(f'{T}/candidates_{m}.csv', index=False)

# ---------- curated pick (from the ranked candidate tables; see notes.md for the rationale)
# rules: one entry per paper; a lab used once across the four modes; an entry used in one mode only; mostly EM
# (6 EM + 1 X-ray per mode); spread over years; prefer long runs and poor site fit, but keep the modal short-run
# sub-pattern of each mode represented.
CURATED = {
    'A_FR1Nterm': ['6ad0', '7czu', '7jhg', '7tz5', '8e8l', '8rvn', '9mnb', '6bpe'],
    'B_CDR2FR3': ['7e39', '12iu', '8cr9', '7oh0', '7mkl', '8dg8', '7wvl'],
    'C_FR3': ['9dm7', '7e23', '8w5e', '7kra', '7mhz', '9yfu', '2x7l'],
    'F_CDR1': ['3jbf', '6lht', '7t6x', '8gse', '9ml5', '9yc6', '7e5y'],
}
pick = []
for m, ids in CURATED.items():
    for p in ids:
        x = KE[(KE['mode'] == m) & (KE.pdb == p)]
        assert len(x) == 1, (m, p)
        pick.append(x.iloc[0].to_dict())
P = pd.DataFrame(pick)
P.to_csv(f'{T}/cases.csv', index=False)
print(P[['mode', 'pdb', 'chain', 'ctype', 'method', 'res', 'year', 'last_author', 'imgt0', 'imgt1', 'delta', 'run_len',
         'site_fit_pct', 'first_imgt', 'gaps_near', 'tgt_cls', 'tgt_name', 'clarity']].to_string())

# ---------- controls
clean = E[(E.err_any == False)].copy()
clean = clean.merge(A[['pdb', 'lab', 'ab_status', 'tgt_keys', 'tgt_fam', 'tgt_cls', 'tgt_name']], on='pdb', how='left')
clean['dep'] = pd.to_datetime(clean.deposit)
paper_of = C.drop_duplicates('pdb').set_index('pdb').paper
clean['paper'] = clean.pdb.map(paper_of)
cc = C[C.pdb.isin(clean.pdb)]
light_present = cc[cc.ctype.isin(['K', 'L'])].groupby('pdb').size(); heavy_present = cc[cc.ctype == 'H'].groupby('pdb').size()
ctype_present = {'K': set(cc[cc.ctype == 'K'].pdb), 'L': set(cc[cc.ctype == 'L'].pdb), 'H': set(cc[cc.ctype == 'H'].pdb)}
vhh_pdb = set(cc[cc.vhh == 1].pdb)
case_vhh = set(C[C.vhh == 1].pdb + '|' + C[C.vhh == 1].chain)

def hazard(pdb, ct, m):
    """does a clean entry have the same local hazard in a chain of the case's chain type?"""
    lo, hi = WIN[m]
    for ch in cc[(cc.pdb == pdb) & (cc.ctype == ct)].chain:
        first, gaps = unbuilt_near(pdb, ch, lo, hi)
        if m == 'A_FR1Nterm' and first == first and first >= 3: return ch
        if m != 'A_FR1Nterm' and gaps == gaps and gaps > 0: return ch
    return None

ctrl = []
used = set()
for r in P.itertuples():
    pool = clean[(clean.method == r.method) & clean.pdb.isin(ctype_present[r.ctype]) & (clean.paper != r.paper) & ~clean.pdb.isin(used)]
    # not the same (possibly unpublished) study: same lab within 90 days of the case deposition is excluded
    pool = pool[~((pool.lab == r.lab) & ((pool.dep - pd.Timestamp(r.deposit)).abs().dt.days <= 90))]
    is_vhh = f'{r.pdb}|{r.chain}' in case_vhh
    if is_vhh: pool = pool[pool.pdb.isin(vhh_pdb)]
    dep = pd.Timestamp(r.deposit)
    pool = pool.assign(dd=(pool.dep - dep).abs().dt.days, dres=(pool.res - r.res).abs())
    keys = set(str(r.tgt_keys).split(';')) - {'nan', ''}
    same_ag = pool[pool.tgt_keys.fillna('').map(lambda s: bool(keys & set(s.split(';')))) & (pool.dd <= 548) & (pool.lab != r.lab)]
    chosen = []
    if len(same_ag):
        x = same_ag.sort_values(['dd', 'dres']).iloc[0]; chosen.append((x, 'same antigen'))
    else:
        per = pool[(pool.tgt_cls == r.tgt_cls) & (pool.year == r.year) & (pool.dres <= 0.4)]
        if not len(per): per = pool[(pool.year == r.year) & (pool.dres <= 0.4)]
        if len(per):
            x = per.sort_values(['dres', 'dd']).iloc[0]; chosen.append((x, 'period/class'))
    # near-miss: same hazard, same period
    nm = pool[(pool.dd <= 365) & (pool.dres <= 0.5) & ~pool.pdb.isin([c[0].pdb for c in chosen])].sort_values('dd')
    for x in nm.head(60).itertuples():
        ch = hazard(x.pdb, r.ctype, r.mode)
        if ch:
            chosen.append((nm.loc[x.Index], f'near-miss ({ch})')); break
    for x, why in chosen:
        used.add(x.pdb)
        ctrl.append(dict(case_pdb=r.pdb, mode=r.mode, case_chain=r.chain, ctype=r.ctype, ctrl_pdb=x.pdb, why=why,
                         method=x.method, res=x.res, year=x.year, deposit=x.deposit, last_author=x.last_author, doi=x.doi,
                         tgt_name=x.tgt_name, tgt_cls=x.tgt_cls, lab=x.lab))
CT = pd.DataFrame(ctrl); CT.to_csv(f'{T}/controls.csv', index=False)
print(CT[['mode', 'case_pdb', 'ctrl_pdb', 'why', 'res', 'year', 'last_author', 'tgt_name']].to_string())
