"""How many of the 2024 study's antibody errors were inherited from the first deposited structure of the same antibody?

For every antibody-chain error in Table S1 (Sánchez Rodríguez et al. 2024):
  1. identify the antibody by the exact V-region sequence of the flagged chain (ANARCI on SEQRES);
  2. find every PDB chain containing that V region (pdb_seqres), with deposition dates (entries.idx);
  3. take the earlier entries (deposited before the flagged entry): the first one is the antibody's first instance;
  4. call the register of the flagged segment in the first instance and in up to 6 more earlier entries, by superposing
     on the flagged chain (template_propagation tmpl4_register.call: local core outside the segment, segment CA
     coincide at offset 0 -> 'same as flagged', at +-shift -> 'other register');
  5. also call up to 6 later entries, to see whether an original error was recycled afterwards.
Output: tables/first_instance_errors.csv (one row per error) and a printed summary.
"""
import sys, os, gzip, json, re, numpy as np, pandas as pd, urllib.request, time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
sys.path.insert(0, '/tmp/scripts'); from ca_fetch import get_ca
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from tmpl4_register import call
from anarci import anarci
OUT = '/home/user/SAbR/register_error_analysis/template_propagation/tables'
PW = '/tmp/supp/pdbwide'

def vregion(seq):
    n, a, _ = anarci([('q', seq)], scheme='imgt', assign_germline=False)
    if not n[0]: return []
    return [(x['chain_type'], x['query_start'], x['query_end']) for x in a[0]]

def scan_chunk(args):
    lines, keys = args
    hits = []
    for h, s in lines:
        for k in keys:
            if k in s: hits.append((k, h[:4].lower(), h[5:].split()[0]))
    return hits

def load_ca(p, c):
    ca = get_ca(p, c)
    return {sid - 1: np.array(xyz) for sid, *xyz in ca} if ca else None

if __name__ == '__main__':
    t = pd.read_excel('/tmp/supp/epmc/m-11-00938-sup2.xlsx').drop(columns=['Unnamed: 0'], errors='ignore')
    t[['pdb', 'chain']] = t.PDB_CHAIN_ID.str.split(':', expand=True); t['pdb'] = t.pdb.str.lower()
    ch = pd.read_pickle('/tmp/analysis/chains.pkl')[['pdb', 'chain', 'ab']]
    t = t.merge(ch, on=['pdb', 'chain'], how='left'); t = t[t.ab == 'antibody'].copy()
    em = pd.read_pickle('/tmp/tmpl/errors_seqmapped.pkl')[['PDB_CHAIN_ID', 'RESIDUE_RANGE', 's0', 's1', 'seq']]
    t = t.merge(em, on=['PDB_CHAIN_ID', 'RESIDUE_RANGE'], how='left')
    print('antibody errors', len(t), 'chains', t[['pdb', 'chain']].drop_duplicates().shape[0], 'entries', t.pdb.nunique(), 'unmapped', t.s0.isna().sum())
    # deposition dates and authors
    dep, auth = {}, {}
    for line in open(f'{PW}/entries.idx', encoding='latin-1'):
        p = line.rstrip('\n').split('\t')
        if len(p) < 6 or len(p[0]) != 4: continue
        try: dep[p[0].lower()] = pd.to_datetime(p[2], format='%m/%d/%y')
        except Exception: continue
        a = [x.strip() for x in p[5].split(',') if x.strip()]
        auth[p[0].lower()] = a[-1] if a else ''
    # flagged chain sequences: errors_seqmapped 'seq' is the chain SEQRES; V region by ANARCI
    seqs = t.dropna(subset=['seq']).drop_duplicates(['pdb', 'chain'])[['pdb', 'chain', 'seq']]
    V = {}
    for r in seqs.itertuples():
        d = vregion(r.seq)
        if d:  # the V domain of this chain (first one; scFv chains use the first domain as identity)
            V[(r.pdb, r.chain)] = r.seq[d[0][1]:d[0][2] + 1]
    keys = sorted(set(V.values()))
    print('distinct antibody V regions', len(keys))
    # scan the whole PDB for chains carrying each V region
    lines = []; h = None
    with gzip.open(f'{PW}/pdb_seqres.txt.gz', 'rt') as f:
        for line in f:
            if line[0] == '>': h = line[1:] if 'mol:protein' in line else None
            elif h: lines.append((h, line.strip()))
    chunks = [(lines[i:i + 20000], keys) for i in range(0, len(lines), 20000)]
    with ProcessPoolExecutor(4) as ex: hits = [x for c in ex.map(scan_chunk, chunks) for x in c]
    occ = pd.DataFrame(hits, columns=['vseq', 'pdb', 'chain']); occ['dep'] = occ.pdb.map(dep); occ['last_author'] = occ.pdb.map(auth)
    occ = occ.dropna(subset=['dep'])
    full = {(h[:4].lower(), h[5:].split()[0]): s for h, s in lines}
    print('PDB chains carrying one of the V regions', len(occ), 'entries', occ.pdb.nunique())
    rows = []; jobs = []
    for r in t.itertuples():
        key = (r.pdb, r.chain); v = V.get(key)
        base = dict(pdb=r.pdb, chain=r.chain, residue_range=r.RESIDUE_RANGE, shift=r.SHIFT, res=r.RESOLUTION, year=r.YEAR, method=r.METHOD,
                    flagged_dep=dep.get(r.pdb), flagged_last_author=auth.get(r.pdb))
        if pd.isna(r.s0) or v is None: rows.append(dict(base, status='unmapped')); continue
        o = occ[(occ.vseq == v) & (occ.pdb != r.pdb)]
        e1 = o.drop_duplicates('pdb').sort_values('dep')
        prior = e1[e1.dep < dep[r.pdb]]; later = e1[e1.dep > dep[r.pdb]]
        base.update(s0=int(r.s0), s1=int(r.s1), v_len=len(v), n_entries_same_v=e1.shape[0] + 1, n_prior=len(prior), n_later=len(later),
                    first_instance=None if prior.empty else prior.pdb.iloc[0], first_dep=None if prior.empty else prior.dep.iloc[0],
                    first_last_author=None if prior.empty else prior.last_author.iloc[0])
        pick_prior = list(prior.head(6).itertuples()) + ([prior.iloc[-1]] if len(prior) > 6 else [])
        pick_later = list(later.head(6).itertuples())
        rows.append(dict(base, status='ok', seq=r.seq))
        for when, sel in (('prior', pick_prior), ('later', pick_later)):
            for x in sel:
                x = x if isinstance(x, pd.Series) else pd.Series(x._asdict())
                jobs.append(dict(err=len(rows) - 1, when=when, cpdb=x['pdb'], cdep=x['dep'], cauthor=x['last_author']))
    # all chains of each candidate entry carrying the V region; register on each, entry = 'same' if any copy is 'same'
    J = pd.DataFrame(jobs)
    cand_chains = occ.groupby(['vseq', 'pdb']).chain.apply(lambda c: sorted(set(c))[:4]).to_dict()
    need = set((p, c) for p, c in zip(t.pdb, t.chain))
    for j in jobs:
        e = rows[j['err']]
        for c in cand_chains.get((V[(e['pdb'], e['chain'])], j['cpdb']), []): need.add((j['cpdb'], c))
    print('chains to fetch', len(need))
    with ThreadPoolExecutor(8) as ex: list(ex.map(lambda pc: get_ca(*pc), sorted(need)))
    calls = []
    for j in jobs:
        e = rows[j['err']]; ref = load_ca(e['pdb'], e['chain'])
        if not ref: calls.append(dict(j, call='no ref')); continue
        rs = e['seq']; s0, s1 = e['s0'], e['s1']; k = max(1, abs(int(e['shift'])))
        win = rs[max(0, s0 - 25):s1 + 26]
        best = None
        for c in cand_chains.get((V[(e['pdb'], e['chain'])], j['cpdb']), []):
            cs = full.get((j['cpdb'], c), ''); w = cs.find(win)
            if w < 0:  # fall back to the V region itself as anchor
                w2 = cs.find(V[(e['pdb'], e['chain'])]); w1 = rs.find(V[(e['pdb'], e['chain'])])
                if w2 < 0: continue
                off = w2 - w1
            else:
                off = w - max(0, s0 - 25)
            cand = load_ca(j['cpdb'], c)
            if not cand: continue
            lab = call(ref, cand, off, s0, s1, k)
            rank = {'same as flagged': 0, 'other register': 1, 'ambiguous': 2, 'segment missing': 3, 'core mismatch': 4, 'no core': 5}.get(lab[0], 6)
            if best is None or rank < best[0]: best = (rank, lab[0], c, lab[1], lab[2], lab[3])
        calls.append(dict(j, call=best[1] if best else 'no coords', cchain=best[2] if best else None,
                          core_rmsd=best[3] if best else None, d_same=best[4] if best else None, d_shift=best[5] if best else None))
    C = pd.DataFrame(calls)
    C.to_csv(f'{OUT}/first_instance_calls.csv', index=False)
    E = pd.DataFrame(rows).drop(columns=['seq'], errors='ignore')
    for i, e in E.iterrows():
        cc = C[C.err == i]
        pr = cc[cc.when == 'prior'].sort_values('cdep'); la = cc[cc.when == 'later']
        E.loc[i, 'first_call'] = pr.call.iloc[0] if len(pr) else None
        E.loc[i, 'prior_same'] = int((pr.call == 'same as flagged').sum()); E.loc[i, 'prior_other'] = int((pr.call == 'other register').sum())
        E.loc[i, 'prior_called'] = len(pr)
        E.loc[i, 'earliest_same_prior'] = pr[pr.call == 'same as flagged'].cpdb.iloc[0] if (pr.call == 'same as flagged').any() else None
        E.loc[i, 'earliest_same_author'] = pr[pr.call == 'same as flagged'].cauthor.iloc[0] if (pr.call == 'same as flagged').any() else None
        E.loc[i, 'later_same'] = int((la.call == 'same as flagged').sum()); E.loc[i, 'later_other'] = int((la.call == 'other register').sum())
        E.loc[i, 'later_called'] = len(la)
    def cls(e):
        if e.status != 'ok': return '9 could not be mapped'
        if e.n_prior == 0:
            return '1 first instance: original error, later copied' if e.later_same > 0 else '2 first instance: original error, not copied (or no later structures)'
        if e.first_call == 'same as flagged': return '3 inherited: first instance already had it'
        if e.prior_same > 0: return '4 inherited: an earlier (not the first) structure had it'
        if e.prior_other > 0: return '5 new error: earlier structures of this antibody are correct'
        return '6 earlier structures exist, register undetermined'
    E['origin'] = E.apply(cls, axis=1)
    E['same_lab_as_source'] = np.where(E.origin.str.startswith('3'), E.flagged_last_author == E.first_last_author,
                                       np.where(E.origin.str.startswith('4'), E.flagged_last_author == E.earliest_same_author, np.nan))
    E.to_csv(f'{OUT}/first_instance_errors.csv', index=False)
    pd.set_option('display.width', 250)
    for unit, g in (('errors', E), ('chains', E.drop_duplicates(['pdb', 'chain'])), ('entries', E.sort_values('origin').drop_duplicates('pdb'))):
        vc = g.origin.value_counts().sort_index()
        print(f'\n== by {unit} (n={len(g)})'); print(pd.DataFrame(dict(n=vc, pct=(100 * vc / len(g)).round(1))).to_string())
    print('\nsame lab as the source, inherited errors:'); print(E[E.origin.str[0].isin(['3', '4'])].same_lab_as_source.value_counts())
