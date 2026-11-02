"""Build the full job list for ABodyBuilder2/NanoBodyBuilder2 + IgFold.

Two sources of (chain, error window) units:
 A. every SAbR-vs-ANARCI disagreement run (../sabr_anarci_scan/tables/disagreement_runs.csv);
 B. every chain in the antibody error families whose flagged register recurs in >=2 papers
    ("problematic templates", ../template_propagation), whatever its call (same/other register/ambiguous).

For each unit the Fv to predict is the flagged V domain plus the nearest complementary V domain
(any chain, including the same chain for scFvs; domain CA-centroid distance <= 35 A). A heavy domain
with no light partner is predicted as a nanobody (NanoBodyBuilder2; IgFold heavy chain only). A light
domain with no heavy partner is skipped. Predictions are made once per unique (H, L) sequence pair.

usage: python h19_jobs.py  -> SP/pred/full/units.json, SP/pred/full/fvs.json
"""
import os, sys, json, gzip, urllib.request, time, warnings, logging
import numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
warnings.filterwarnings('ignore'); logging.disable(logging.WARNING)
from Bio.PDB import MMCIFParser
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from anarci import anarci
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
R = '/home/user/SAbR/register_error_analysis'
CIF = f'{SP}/sabr/dl'; OUT = f'{SP}/pred/full'
os.makedirs(CIF, exist_ok=True); os.makedirs(OUT, exist_ok=True)
CDR = lambda p: 27 <= p <= 38 or 56 <= p <= 65 or 105 <= p <= 117

def dl(pdb):
    fn = f'{CIF}/{pdb}.cif'
    if os.path.exists(fn) and os.path.getsize(fn) > 0: return fn
    for url in (f'https://files.rcsb.org/download/{pdb.upper()}.cif.gz', f'https://www.ebi.ac.uk/pdbe/entry-files/download/{pdb}.cif'):
        for k in range(3):
            try:
                d = urllib.request.urlopen(url, timeout=180).read()
                if url.endswith('.gz'): d = gzip.decompress(d)
                open(fn, 'wb').write(d); return fn
            except Exception as e:
                if '404' in str(e): break
                time.sleep(2 ** k)
    return None

def entry_domains(pdb):
    """-> dict with entity sequences per chain, V domains per chain (type, start, end, centroid), meta."""
    fn = dl(pdb)
    if not fn: return dict(pdb=pdb, error='download failed')
    try:
        md = MMCIF2Dict(fn); st = MMCIFParser(QUIET=True).get_structure(pdb, fn)[0]
    except Exception as e:
        return dict(pdb=pdb, error=f'parse: {e}')
    strand = md['_pdbx_poly_seq_scheme.pdb_strand_id']; ent = md['_pdbx_poly_seq_scheme.entity_id']
    seqid = md['_pdbx_poly_seq_scheme.seq_id']; num = md['_pdbx_poly_seq_scheme.pdb_seq_num']; ins = md['_pdbx_poly_seq_scheme.pdb_ins_code']
    ids = md['_entity_poly.entity_id']; seqs = md['_entity_poly.pdbx_seq_one_letter_code_can']; typ = md['_entity_poly.type']
    if isinstance(ids, str): ids, seqs, typ = [ids], [seqs], [typ]
    e2s = {i: s.replace('\n', '').replace(';', '').strip() for i, s, t in zip(ids, seqs, typ) if 'peptide' in t}
    c2e = {}; idx = {}
    for s, e, i, n, c in zip(strand, ent, seqid, num, ins):
        c2e[s] = e; idx[(s, int(n), '' if c in '.?' else c)] = int(i) - 1
    chains = {c: e2s[e] for c, e in c2e.items() if e in e2s and len(e2s[e]) >= 70}
    useq = sorted(set(chains.values()))
    numb, ali, _ = anarci([(f's{i}', s) for i, s in enumerate(useq)], scheme='imgt', assign_germline=False)
    dom_by_seq = {}
    for s, n, a in zip(useq, numb, ali):
        dom_by_seq[s] = [] if not n else [(x['chain_type'], x['query_start'], x['query_end']) for x in a]
    doms = []
    for c, s in chains.items():
        if c not in st: continue
        cas = {}
        for r in st[c]:
            k = idx.get((c, r.id[1], r.id[2].strip()))
            if k is not None and 'CA' in r: cas[k] = r['CA'].coord
        for di, (t, a, b) in enumerate(dom_by_seq.get(s, [])):
            if t not in 'HKL': continue
            xyz = [v for k, v in cas.items() if a <= k <= b]
            if len(xyz) < 20: continue
            doms.append(dict(chain=c, dom=di, ndom=len(dom_by_seq[s]), type=t, start=a, end=b, seq=s[a:b + 1],
                             cen=np.mean(xyz, 0).tolist(), nres=len(xyz)))
    rel = sorted(md.get('_pdbx_audit_revision_history.revision_date', ['?']))[0]
    return dict(pdb=pdb, doms=doms, chains=chains, release=rel)

def pick_partner(doms, chain, ctype, seqhint=None):
    """flagged domain = domain of `chain` with type ctype (H vs light); partner = nearest complementary domain."""
    light = ctype in 'KL'
    mine = [d for d in doms if d['chain'] == chain and ((d['type'] in 'KL') == light)]
    if not mine: return None, None, None
    me = mine[0]
    oth = [d for d in doms if (d['type'] in 'KL') != light and not (d['chain'] == me['chain'] and d['dom'] == me['dom'])]
    best = None
    for d in oth:
        dist = float(np.linalg.norm(np.array(d['cen']) - np.array(me['cen'])))
        if best is None or dist < best[0]: best = (dist, d)
    if best is None or best[0] > 35: return me, None, None if best is None else best[0]
    return me, best[1], best[0]

def fr_insertion(vseq):
    n, _, _ = anarci([('q', vseq)], scheme='imgt', assign_germline=False)
    if not n[0]: return None
    return any(i.strip() and not CDR(p) for (p, i), a in n[0][0][0] if a != '-')

if __name__ == '__main__':
    # ---- A: disagreement runs
    r = pd.read_csv(f'{R}/sabr_anarci_scan/tables/disagreement_runs.csv')
    m = pd.read_csv(f'{R}/sabr_anarci_scan/tables/mapfit_detections.csv')
    m = m[m.set == 'detected'].rename(columns={'imgt0': 'imgt_start', 'imgt1': 'imgt_end'})
    r = r.merge(m[['pdb', 'chain', 'imgt_start', 'imgt_end', 'kind', 'pct']], on=['pdb', 'chain', 'imgt_start', 'imgt_end'], how='left')
    A = [dict(source='disagreement', pdb=x.pdb, chain=x.chain, chain_type=x.chain_type, imgt_start=int(x.imgt_start), imgt_end=int(x.imgt_end),
              delta=None if pd.isna(x.delta) else int(x.delta), loc=x.loc, run_len=int(x.length), kind=None if pd.isna(x.kind) else x.kind,
              mapfit_pct=None if pd.isna(x.pct) else float(x.pct), method=x.method, res=x.res, era=x.era) for x in r.itertuples()]
    # ---- B: problematic-template antibody families
    T = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad/tmpl2/out'
    fams = [203, 231, 232, 237, 372, 504, 1359, 2712, 2788]
    F = pd.read_pickle('/tmp/tmpl/families.pkl').set_index('family')
    rc = pd.read_pickle(f'{T}/register_calls_chain.pkl'); rc = rc[rc.family.isin(fams)]
    B = [dict(source='template_family', family=int(x.family), pdb=x.pdb, chain=x.chain, call=x.call, family_flagged=bool(x.flagged),
              win=F.loc[x.family, 'win'], s0=int(F.loc[x.family, 's0']), s1=int(F.loc[x.family, 's1']), conkit_shift=int(F.loc[x.family, 'shift']),
              method=x.method, res=x.res) for x in rc.itertuples()]
    pdbs = sorted(set(u['pdb'] for u in A + B))
    print('units', len(A), '+', len(B), 'entries', len(pdbs), flush=True)
    with ThreadPoolExecutor(8) as ex: list(ex.map(dl, pdbs))
    print('downloaded', sum(os.path.exists(f'{CIF}/{p}.cif') for p in pdbs), flush=True)
    with ProcessPoolExecutor(4) as ex: E = {e['pdb']: e for e in ex.map(entry_domains, pdbs, chunksize=4)}
    print('parsed', sum('doms' in e for e in E.values()), flush=True)
    units = []; fvs = {}
    for u in A + B:
        e = E[u['pdb']]
        if 'doms' not in e: u['status'] = e.get('error'); units.append(u); continue
        u['release'] = e['release']
        if u['source'] == 'template_family':
            s = e['chains'].get(u['chain'], ''); w = s.find(u['win'])
            cands = [d for d in e['doms'] if d['chain'] == u['chain']]
            if w < 0 or not cands: u['status'] = 'window or V domain not found'; units.append(u); continue
            a, b = w + u['s0'], w + u['s1']
            d = [d for d in cands if d['start'] <= a and b <= d['end']]
            if not d: u['status'] = 'error window outside the V region'; units.append(u); continue
            u['chain_type'] = d[0]['type']; u['seq_window'] = [a - d[0]['start'], b - d[0]['start']]
            u['delta'] = -u['conkit_shift']  # conkit's sign convention is opposite to SAbR - ANARCI
        me, partner, dist = pick_partner(e['doms'], u['chain'], u['chain_type'])
        if me is None: u['status'] = 'flagged V domain not modelled'; units.append(u); continue
        u['pair_dist'] = None if dist is None else round(dist, 1)
        if partner is None and me['type'] in 'KL': u['status'] = 'light chain without heavy partner (skipped)'; units.append(u); continue
        if partner is None:
            H, L, mode = me['seq'], '', 'nanobody'
        else:
            H, L = (me['seq'], partner['seq']) if me['type'] == 'H' else (partner['seq'], me['seq'])
            mode = 'scfv' if partner['chain'] == me['chain'] else 'fv'
            u['partner_chain'] = partner['chain']
        if len(H) > 140 or len(L) > 140: u['status'] = 'V region >140 residues (skipped)'; units.append(u); continue
        key = f'{H}|{L}'
        if key not in fvs: fvs[key] = dict(fv_id=f'fv{len(fvs):04d}', Hseq=H, Lseq=L, mode=mode)
        u.update(fv_id=fvs[key]['fv_id'], mode=mode, vseq=me['seq'], status='ok', fr_insertion=fr_insertion(me['seq']))
        units.append(u)
    json.dump(units, open(f'{OUT}/units.json', 'w'), indent=0, default=str)
    json.dump(list(fvs.values()), open(f'{OUT}/fvs.json', 'w'), indent=0)
    s = pd.DataFrame(units)
    print(s.groupby(['source', 'status']).size().to_string())
    print(s[s.status == 'ok'].groupby(['source', 'mode']).size().to_string())
    print('unique Fvs', len(fvs), pd.Series([f['mode'] for f in fvs.values()]).value_counts().to_dict())
