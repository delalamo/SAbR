"""Coordinate-level features at the CDR3/FR4 site for D+/D-/E error chains and matched clean controls.

Sample
  errors  : every chain with a first-occurrence D+/D-/E run (all methods).
  controls: (a) clean units from the same papers (within-paper controls);
            (b) clean cryo-EM units (entry without any run) matched on deposition year x chain type x nanobody,
                up to 4 per error unit (one chain per unit).
Features (per chain; IMGT positions are ANARCI = sequence-based, so they follow the deposited sequence)
  fr4_seqres_after118 : SEQRES residues after the ANARCI-118 residue (construct/record length past W/F118)
  fr4_built_after118  : modelled residues after 118 within the V domain (to the end of the V region)
  unbuilt_105_128     : SEQRES residues between ANARCI 105 and 128 that have no coordinates
  last_built_imgt     : last modelled ANARCI IMGT position of the V domain
  w118_ca10           : number of V-domain CA atoms within 10 A of the side-chain centroid of the ANARCI-118 residue
                        (< 14 = pulled out of the core; structure-only, independent of SAbR)
  slot118_aa          : residue the model places in the structural (SAbR) 118 slot
  cdr3_ag_contacts    : CDR3 (ANARCI 105-117) residues with any heavy atom within 4.5 A of a non-antibody chain
  cdr3_partner_contacts: same, to another antibody chain (VH-VL partner)
  sc_trunc_site       : fraction of non-Gly/Ala residues in ANARCI 105-128 lacking side-chain atoms beyond CB
  sc_trunc_rest       : same for the rest of the V domain
  b_site_rel          : mean B of ANARCI 105-128 / mean B of the V domain
"""
import sys, os, io, gzip, json, time, urllib.request, numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, '/home/user/SAbR/register_error_analysis/retest/deeper/case_files_DE/scripts')
from common import *
from Bio.PDB import MMCIFParser, NeighborSearch

CIFDIR = f'{CACHE}/cif'; os.makedirs(CIFDIR, exist_ok=True)
OTHER = [f'{D}/cache/struct_unusual/cif', f'{D}/cache/case_files_ABCF/cif', f'{SP}/sabr/dl']
AB_KW = ('heavy', 'light', 'fab', 'antibod', 'nanobod', 'igg', 'kappa', 'lambda', 'scfv', 'vhh', 'megabod', 'sybod',
         'immunoglobulin', 'fv ', ' fv', 'ig ', 'ig gamma', 'variable', 'mab', 'diabody', 'tribody', 'igh', 'igk', 'igl')

def cif_path(p):
    for d in [CIFDIR] + OTHER:
        for ext in ('.cif.gz', '.cif'):
            f = f'{d}/{p}{ext}'
            if os.path.exists(f) and os.path.getsize(f) > 1000: return f
    f = f'{CIFDIR}/{p}.cif.gz'
    for k in range(4):
        try:
            data = urllib.request.urlopen(f'https://files.rcsb.org/download/{p}.cif.gz', timeout=120).read()
            open(f, 'wb').write(data); return f
        except Exception:
            time.sleep(2 + 3 * k)
    return None

def open_cif(f):
    return io.StringIO(gzip.open(f, 'rt').read()) if f.endswith('.gz') else open(f)

FW = [i for i in range(4, 26)] + [i for i in range(39, 56)] + [i for i in range(66, 105)]
def kabsch(P, Q):
    Pc, Qc = P.mean(0), Q.mean(0)
    H = (P - Pc).T @ (Q - Qc); U, S, Vt = np.linalg.svd(H); d = np.sign(np.linalg.det(Vt.T @ U.T))
    Dm = np.diag([1, 1, d]); Rm = Vt.T @ Dm @ U.T
    return Rm, Qc - Pc @ Rm.T
def slot_shift(rmap, anarci, ref, runpos):
    # superpose target framework CA on reference framework CA (ANARCI numbering, outside any run), then
    # assign each target residue with ANARCI 110-128 to the nearest reference CA slot (ref numbering 105-128)
    P, Q = [], []
    for n in FW:
        a = anarci.get((n, ''))
        if a and a in rmap and 'CA' in rmap[a] and (n, '') in ref and n not in runpos:
            P.append(rmap[a]['CA'].coord); Q.append(ref[(n, '')])
    if len(P) < 40: return {}
    P, Q = np.array(P), np.array(Q); Rm, t = kabsch(P, Q)
    rms = float(np.sqrt(((P @ Rm.T + t - Q) ** 2).sum(1).mean()))
    slots = [k for k in ref if 105 <= k[0] <= 128]; S = np.array([ref[k] for k in slots])
    order = sorted(slots, key=lambda k: (k[0], k[1]))
    rank = {k: i for i, k in enumerate(order)}
    out = {'fw_rmsd': rms}
    for n in (116, 117, 118, 119, 120, 121, 122, 123, 124):
        a = anarci.get((n, ''))
        if a and a in rmap and 'CA' in rmap[a]:
            x = rmap[a]['CA'].coord @ Rm.T + t; dd = np.linalg.norm(S - x, axis=1); j = int(dd.argmin())
            out[f'slot_{n}'] = slots[j][0] + (0.5 if slots[j][1] else 0); out[f'slotd_{n}'] = float(dd[j])
            out[f'slotshift_{n}'] = rank[slots[j]] - rank.get((n, ''), np.nan) if (n, '') in rank else np.nan
    return out

def analyse_entry(job):
    p, chains_req, vchains_all, refs, runpos_by_chain = job
    out = []
    try:
        f = cif_path(p)
        if f is None: return [dict(pdb=p, chain=c, status='no cif') for c, _ in chains_req]
        parser = MMCIFParser(QUIET=True)
        s = parser.get_structure(p, open_cif(f)); md = parser._mmcif_dict
        model = next(iter(s))
        # entity descriptions per auth chain
        ent_desc = dict(zip(md.get('_entity.id', []), md.get('_entity.pdbx_description', [])))
        asym2ent = dict(zip(md.get('_struct_asym.id', []), md.get('_struct_asym.entity_id', [])))
        # poly seq scheme
        ps = pd.DataFrame({k.split('.')[1]: md[k] for k in md if k.startswith('_pdbx_poly_seq_scheme.')})
        auth2desc = {}
        if len(ps):
            for a, sa in zip(ps.pdb_strand_id, ps.asym_id): auth2desc[a] = ent_desc.get(asym2ent.get(sa, ''), '')
        ab_auth = set(vchains_all)
        for a, d in auth2desc.items():
            dl = ' ' + d.lower() + ' '
            if any(k in dl for k in AB_KW): ab_auth.add(a)
        ag_atoms = [at for ch in model if ch.id not in ab_auth for r in ch if r.id[0] == ' ' for at in r if at.element != 'H']
        ab_other_atoms = {c: [at for r in model[c] if r.id[0] == ' ' for at in r if at.element != 'H'] for c in ab_auth if c in model}
        ns_ag = NeighborSearch(ag_atoms) if ag_atoms else None
        for c, res in chains_req:
            row = dict(pdb=p, chain=c, status='ok')
            if c not in model: row['status'] = 'chain missing'; out.append(row); continue
            ch = model[c]
            v = [x for x in res if x[4] is not None]
            key = {(x[0], x[1]): x for x in v}
            # map residues
            rmap = {}
            for r in ch:
                if r.id[0] != ' ': continue
                k = (r.id[1], r.id[2].strip())
                if k in key: rmap[k] = r
            anarci = {(x[4], x[5]): (x[0], x[1]) for x in v}
            sabr = {(x[6], x[7]): (x[0], x[1]) for x in v if x[6] is not None}
            # SEQRES scheme for this chain
            q = ps[ps.pdb_strand_id == c] if len(ps) else pd.DataFrame()
            a118 = anarci.get((118, ''))
            if len(q) and a118:
                q = q.reset_index(drop=True)
                q['built'] = q.pdb_mon_id != '?'
                q['auth'] = list(zip(pd.to_numeric(q.auth_seq_num, errors='coerce'), q.pdb_ins_code.replace({'.': ''})))
                # position of W118 in seqres: find built row whose auth num matches
                hit = q.index[(q.built) & (q.auth.map(lambda t: (t[0], t[1] if isinstance(t[1], str) else '') == (a118[0], a118[1])))]
                if len(hit):
                    i118 = hit[0]
                    row['fr4_seqres_after118'] = int(len(q) - i118 - 1)
                    after = q.iloc[i118 + 1:i118 + 11]
                    row['fr4_built_after118_10'] = int(after.built.sum())
                    row['fr4_unbuilt_after118_10'] = int((~after.built).sum())
            # unbuilt within 105-128 (between first built 105.. and 128 in seqres order)
            built_v = [x for x in v if 100 <= x[4] <= 128]
            if len(q) and built_v:
                try:
                    lo = rmap.get((built_v[0][0], built_v[0][1])); hi = rmap.get((built_v[-1][0], built_v[-1][1]))
                    qa = q.copy(); qa['an'] = pd.to_numeric(qa.auth_seq_num, errors='coerce')
                    # seqres rows between those two built residues
                    i_lo = qa.index[(qa.built) & (qa.an == built_v[0][0])]
                    i_hi = qa.index[(qa.built) & (qa.an == built_v[-1][0])]
                    if len(i_lo) and len(i_hi): row['unbuilt_100_128'] = int((~qa.loc[i_lo[0]:i_hi[-1], 'built']).sum())
                except Exception: pass
            row['last_built_imgt'] = max(x[4] for x in v)
            row['first_built_imgt'] = min(x[4] for x in v)
            # V-domain CA
            ca = np.array([r['CA'].coord for r in rmap.values() if 'CA' in r])
            if a118 and a118 in rmap:
                r118 = rmap[a118]
                sc = [at.coord for at in r118 if at.get_id() not in ('N', 'CA', 'C', 'O', 'CB')]
                row['aa118'] = r118.get_resname()
                if sc:
                    cen = np.mean(sc, axis=0); row['w118_ca10'] = int((np.linalg.norm(ca - cen, axis=1) < 10).sum())
            s118 = sabr.get((118, ''))
            if s118 and s118 in rmap: row['slot118_aa'] = rmap[s118].get_resname()
            # CDR3 contacts
            cdr3 = [rmap[a] for (n, i), a in anarci.items() if 105 <= n <= 117 and a in rmap]
            row['cdr3_built'] = len(cdr3)
            if ns_ag is not None:
                row['cdr3_ag_contacts'] = sum(any(ns_ag.search(at.coord, 4.5) for at in r if at.element != 'H') for r in cdr3)
                vres = list(rmap.values())
                row['v_ag_contacts'] = sum(any(ns_ag.search(at.coord, 4.5) for at in r if at.element != 'H') for r in vres)
            else:
                row['cdr3_ag_contacts'] = 0; row['v_ag_contacts'] = 0
            part = [at for cc, ats in ab_other_atoms.items() if cc != c for at in ats]
            if part:
                nsp = NeighborSearch(part)
                row['cdr3_partner_contacts'] = sum(any(nsp.search(at.coord, 4.5) for at in r if at.element != 'H') for r in cdr3)
            row['n_ag_atoms'] = len(ag_atoms)
            # side-chain truncation
            def trunc(r):
                if r.get_resname() in ('GLY', 'ALA'): return None
                return not any(at.get_id() not in ('N', 'CA', 'C', 'O', 'CB', 'OXT') for at in r)
            site = [rmap[a] for (n, i), a in anarci.items() if 105 <= n <= 128 and a in rmap]
            rest = [rmap[a] for (n, i), a in anarci.items() if not (105 <= n <= 128) and a in rmap]
            ts = [t for t in map(trunc, site) if t is not None]; tr = [t for t in map(trunc, rest) if t is not None]
            row['sc_trunc_site'] = float(np.mean(ts)) if ts else np.nan
            row['sc_trunc_rest'] = float(np.mean(tr)) if tr else np.nan
            bs = [at.get_bfactor() for r in site for at in r]; bv = [at.get_bfactor() for r in rmap.values() for at in r]
            row['b_site_rel'] = float(np.mean(bs) / np.mean(bv)) if bs and np.mean(bv) > 0 else np.nan
            row['b_v_mean'] = float(np.mean(bv)) if bv else np.nan
            row['desc'] = auth2desc.get(c, '')
            ref = refs[refs['_ct'][c]] if c in refs['_ct'] else None
            if ref is not None:
                row.update(slot_shift(rmap, anarci, ref, runpos_by_chain.get(c, set())))
            out.append(row)
    except Exception as e:
        out.append(dict(pdb=p, chain='*', status=f'error {e}'))
    return out

if __name__ == '__main__':
    C = chains(); R = runs()
    ent = pd.read_csv(f'{D}/entries.csv')
    clean_entries = set(ent[~ent.err_any].pdb)
    E = R[R['mode'].isin(MODES)][['pdb', 'chain']].drop_duplicates()
    err_keys = set(zip(E.pdb, E.chain))
    err_units = C[[k in err_keys for k in zip(C.pdb, C.chain)]]
    # within-paper clean controls
    wp = C[C.paper.isin(err_units.paper) & C.pdb.isin(clean_entries)].drop_duplicates('unit')
    # matched controls
    rng = np.random.default_rng(7)
    pool = C[(C.method == 'EM') & C.pdb.isin(clean_entries)].drop_duplicates('unit')
    eu = err_units[err_units.method == 'EM'].drop_duplicates('unit')
    picks = []
    for (y, ct, vh), g in eu.groupby(['year', 'ctype', 'vhh']):
        cand = pool[(pool.year == y) & (pool.ctype == ct) & (pool.vhh == vh)]
        n = min(len(cand), 4 * len(g))
        if n: picks.append(cand.sample(n, random_state=int(rng.integers(1e9))))
    mc = pd.concat(picks)
    sel = pd.concat([err_units.assign(group='error'), wp.assign(group='within_paper_clean'), mc.assign(group='matched_clean')])
    sel = sel.drop_duplicates(['pdb', 'chain', 'group'])
    sel[['pdb', 'chain', 'group', 'unit', 'paper', 'year', 'ctype', 'vhh', 'method', 'res']].to_csv(f'{CACHE}/struct_sample.csv', index=False)
    print(sel.groupby('group').agg(chains=('pdb', 'size'), entries=('pdb', 'nunique'), units=('unit', 'nunique')))
    keys = set(zip(sel.pdb, sel.chain))
    rec = sabr_records()
    vall = C.groupby('pdb').chain.apply(list).to_dict()
    # reference frameworks (high-resolution, clean): heavy 5dt1 H, kappa 1n8z A, lambda 5dt1 L
    tp = {}
    for l in open(f'{SP}/retest/templates.jsonl'):
        r = json.loads(l)
        if r.get('status') == 'ok' and (r['pdb'], r['chain']) in {('1n8z', 'A'), ('5dt1', 'H'), ('5dt1', 'L')}: tp[(r['pdb'], r['chain'])] = r['residues']
    REF = {}
    for (p, c), ct in {('5dt1', 'H'): 'H', ('1n8z', 'A'): 'K', ('5dt1', 'L'): 'L'}.items():
        parser = MMCIFParser(QUIET=True); st = parser.get_structure(p, open_cif(cif_path(p))); ch = next(iter(st))[c]
        num = {(x[0], x[1]): (x[6], x[7]) for x in tp[(p, c)] if x[6] is not None}
        REF[ct] = {num[(r.id[1], r.id[2].strip())]: r['CA'].coord for r in ch if (r.id[1], r.id[2].strip()) in num and 'CA' in r}
        print('ref', ct, p, c, len(REF[ct]))
    ctype = dict(zip(zip(C.pdb, C.chain), C.ctype))
    RA = runs_all()
    runpos = {}
    for r in RA.itertuples():
        runpos.setdefault((r.pdb, r.chain), set()).update(range(int(r.imgt0) - 1, int(r.imgt1) + 2))
    jobs = []
    for p, g in sel.groupby('pdb'):
        req = [(c, rec[(p, c)]) for c in sorted(set(g.chain)) if (p, c) in rec]
        refs = {'_ct': {c: ctype[(p, c)] for c, _ in req}}
        refs.update(REF)
        if req: jobs.append((p, req, vall.get(p, []), refs, {c: runpos.get((p, c), set()) for c, _ in req}))
    print('entries to analyse', len(jobs), flush=True)
    res = []
    t0 = time.time()
    with ProcessPoolExecutor(3) as ex:
        for i, o in enumerate(ex.map(analyse_entry, jobs, chunksize=2)):
            res.extend(o)
            if i % 50 == 0: print(i, round(time.time() - t0), flush=True)
    F = pd.DataFrame(res)
    F.to_csv(f'{OUT}/tables/struct_features.csv', index=False)
    print(F.status.value_counts())
