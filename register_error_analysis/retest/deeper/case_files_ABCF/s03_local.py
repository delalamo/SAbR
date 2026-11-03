"""Local structure at the error site, for every case and control (same ANARCI IMGT window).

For the case chain (and for the matching-chain-type V chain of each control):
  * window sequence (SEQRES-based ANARCI IMGT numbering) with 3-residue flanks
  * unbuilt residues in the window +-5, first built IMGT position
  * map fit: Q-score (EM) / RSCC (X-ray) mean in the window and its percentile among the V-domain residues
  * wwPDB validation XML: Ramachandran / rotamer outliers and clashes per residue (window vs rest of V domain),
    Q-score from the report, X-ray RSRZ
  * EMDB validation analysis: per-residue local resolution (window, V domain, whole map); SMOC where present
  * B-factors (window z within the V domain)
  * contacts (heavy atoms <= 4.5 A) of the window residues: partner antibody chain of the same Fab/Fv, other
    antibody chains (other Fabs, symmetry copies), and non-antibody chains (antigen / complex); and whether the
    V domain contacts any non-antibody chain at all
Output: tables/local_sites.csv
"""
import json, gzip, os, re, numpy as np, pandas as pd, xml.etree.ElementTree as ET
from scipy.spatial import cKDTree
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
CA = f'{SP}/deeper/cache/case_files_ABCF'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/case_files_ABCF'
cases = pd.read_csv(f'{OUT}/tables/cases.csv'); ctrls = pd.read_csv(f'{OUT}/tables/controls.csv')
C = pd.read_csv(f'{SP}/deeper/chains.csv.gz', low_memory=False)
ROLES = pd.read_csv('/home/user/SAbR/register_error_analysis/retest/deeper/antigen_competition/tables/pdbwide_ab_chain_roles.csv.gz')
need = set(cases.pdb) | set(ctrls.ctrl_pdb)
REC = {}
for fn in ['new.jsonl', 'snapshot_rest.jsonl', 'validation.jsonl']:
    for line in open(f'{SP}/sabr/out/{fn}'):
        if line[9:13] not in need and json.loads(line)['pdb'] not in need: continue
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
AC = json.load(open(f'{SP}/sabr/anarci_cache.json'))

def numbering(seq, ct):
    doms = AC.get(seq)
    if doms is None:
        from anarci import run_anarci
        res = run_anarci([('q', seq)], scheme='imgt', assign_germline=False)
        doms = []
        for (num, st, en), info in zip(res[1][0] or [], res[2][0] or []):
            pos, k = [], st
            for (p, ins), aa in num:
                if aa != '-': pos.append([k, p, ins.strip()]); k += 1
            doms.append(dict(chain_type=info['chain_type'], pos=pos))
        AC[seq] = doms
    for d in doms:
        if (d['chain_type'] in 'KL') == (ct in 'KL'): return [((p, i), seq[s]) for s, p, i in d['pos']]
    return None

def read_atoms(pdb):
    """minimal mmCIF atom_site reader (model 1, heavy atoms)"""
    f = f'{CA}/cif/{pdb}.cif.gz'
    cols, rows, in_loop = [], [], False
    with gzip.open(f, 'rt') as fh:
        for line in fh:
            if line.startswith('_atom_site.'):
                cols.append(line.strip().split('.', 1)[1]); in_loop = True; continue
            if in_loop:
                if line.startswith(('ATOM', 'HETATM')): rows.append(line.split())
                elif cols and (line.startswith('#') or line.startswith('loop_')): break
    df = pd.DataFrame(rows, columns=cols[:len(rows[0])] if rows else cols)
    if 'pdbx_PDB_model_num' in df: df = df[df.pdbx_PDB_model_num == df.pdbx_PDB_model_num.iloc[0]]
    df = df[(df.group_PDB == 'ATOM') & (df.type_symbol != 'H')]
    ic = df.pdbx_PDB_ins_code.replace('?', '').replace('.', '') if 'pdbx_PDB_ins_code' in df else ''
    out = pd.DataFrame(dict(chain=df.auth_asym_id.values, resid=(df.auth_seq_id + ic).values,
                            x=df.Cartn_x.astype(float).values, y=df.Cartn_y.astype(float).values, z=df.Cartn_z.astype(float).values,
                            b=df.B_iso_or_equiv.astype(float).values, atom=df.auth_atom_id.values))
    return out

def read_vrpt(pdb):
    f = f'{CA}/vrpt/{pdb}_validation.xml.gz'
    if not os.path.exists(f): return {}
    out = {}
    try:
        for ev, el in ET.iterparse(gzip.open(f)):
            if el.tag == 'ModelledSubgroup' and el.get('model', '1') == '1':
                k = (el.get('chain'), el.get('resnum') + (el.get('icode') or '').strip())
                out[k] = dict(rama_out=el.get('rama') == 'OUTLIER', rota_out=el.get('rota') == 'OUTLIER',
                              clash=len(el.findall('clash')), q=float(el.get('Q_score')) if el.get('Q_score') else np.nan,
                              rsrz=float(el.get('rsrz')) if el.get('rsrz') else np.nan)
                el.clear()
    except Exception as e:
        print('vrpt parse', pdb, e)
    return out

def read_va(pdb):
    try: e = json.load(open(f'{CA}/entry/{pdb}.json'))
    except Exception: return None, {}, {}
    em = ((e.get('rcsb_entry_container_identifiers') or {}).get('emdb_ids') or [None])[0]
    if not em or not os.path.exists(f'{CA}/emdb_va/{em}.json'): return None, {}, {}
    raw = json.load(open(f'{CA}/emdb_va/{em}.json')); num = em.split('-')[1]
    x = raw.get(num) or (list(raw.values())[0] if raw else {})
    lr, smoc = {}, {}
    for kk, v in (x.get('residue_local_resolution') or {}).items():
        if isinstance(v, dict) and 'data' in v and (pdb in (v.get('name') or '').lower() or True):
            for r, val in zip(v['data']['residue'], v['data']['localResolution']):
                c, rest = r.split(':', 1); lr[(c, rest.split()[0])] = val
            break
    try:
        for kk, v in (x.get('smoc') or {}).items():
            if isinstance(v, dict) and 'data' in v:
                d = v['data']
                for c, w in d.items():
                    if isinstance(w, dict) and 'residue' in w:
                        for r, val in zip(w['residue'], w.get('smoc_scores') or w.get('score') or []):
                            smoc[(c, str(r).split()[0])] = val
                break
    except Exception: pass
    res = (x.get('resolution') or {}).get('value')
    return res, lr, smoc

ab_chains = ROLES.groupby('pdb').apply(lambda g: dict(zip(g.chain, g.ab_group))).to_dict()

def site(pdb, ch, ct, lo, hi, atoms, vr, va):
    row = dict(pdb=pdb, chain=ch, lo=lo, hi=hi)
    cr = C[(C.pdb == pdb) & (C.chain == ch)]
    if not len(cr): row['note'] = 'chain not in scan'; return row
    seq = cr.seq.iloc[0]; num = numbering(seq, ct); rec = REC.get((pdb, ch))
    if not num or not rec: row['note'] = 'no numbering/records'; return row
    modelled = {(x[4], x[5] or ''): f'{x[0]}{x[1] or ""}' for x in rec if x[4] is not None}
    vdom = [modelled[p] for p, _ in num if p in modelled]
    win = [(p, a) for p, a in num if lo <= p[0] <= hi]
    fl = [(p, a) for p, a in num if lo - 3 <= p[0] <= hi + 3]
    row['win_seq'] = ''.join(a for _, a in win)
    row['flank_seq'] = ''.join((a if lo <= p[0] <= hi else a.lower()) for p, a in fl)
    row['unbuilt_win5'] = sum(p not in modelled for p, _ in num if lo - 5 <= p[0] <= hi + 5)
    row['unbuilt_win'] = sum(p not in modelled for p, _ in win)
    row['seqres_first_imgt'] = num[0][0][0]
    row['first_built_imgt'] = min(p[0] for p in modelled)
    wauth = [modelled[p] for p, _ in win if p in modelled]
    row['n_win_built'] = len(wauth)
    lq = LQ.get((pdb, ch)) or {}
    v = pd.Series({a: lq[a] for a in vdom if a in lq})
    if len(v) >= 30 and any(a in v for a in wauth):
        rk = v.rank(pct=True); w = [a for a in wauth if a in v]
        row['fit_win'] = v[w].mean(); row['fit_pct'] = rk[w].mean(); row['fit_dom'] = v.mean()
    # validation outliers
    V = pd.DataFrame([dict(a=a, inwin=a in wauth, **vr[(ch, a)]) for a in vdom if (ch, a) in vr])
    if len(V):
        for k in ['rama_out', 'rota_out', 'clash']:
            row[f'{k}_win'] = V[V.inwin][k].mean(); row[f'{k}_rest'] = V[~V.inwin][k].mean()
        row['vq_win'] = V[V.inwin].q.mean(); row['vq_rest'] = V[~V.inwin].q.mean()
        row['rsrz_win'] = V[V.inwin].rsrz.mean(); row['rsrz_rest'] = V[~V.inwin].rsrz.mean()
    # local resolution
    gres, lr, smoc = va
    if lr:
        lw = [lr[(ch, a)] for a in wauth if (ch, a) in lr]; ld = [lr[(ch, a)] for a in vdom if (ch, a) in lr]
        allv = list(lr.values())
        if lw:
            row['lr_win'] = np.mean(lw); row['lr_dom'] = np.median(ld); row['lr_map_median'] = np.median(allv); row['global_res'] = gres
            row['lr_win_pct_dom'] = np.mean([np.mean(np.array(ld) <= x) for x in lw])  # high = worse than rest of domain
    if smoc:
        sw = [smoc[(ch, a)] for a in wauth if (ch, a) in smoc]; sd = [smoc[(ch, a)] for a in vdom if (ch, a) in smoc]
        if sw: row['smoc_win'] = np.mean(sw); row['smoc_dom'] = np.mean(sd)
    # atoms: B-factor and contacts
    A = atoms
    me = A[(A.chain == ch)]
    dom = me[me.resid.isin(vdom)]; w = me[me.resid.isin(wauth)]
    if len(dom) and len(w):
        rb = dom.groupby('resid').b.mean()
        row['b_win_z'] = (rb[rb.index.isin(wauth)].mean() - rb.mean()) / (rb.std() or 1)
        abmap = ab_chains.get(pdb, {})
        grp = abmap.get(ch)
        others = A[A.chain != ch]
        tree = cKDTree(others[['x', 'y', 'z']].values)
        def contacts(sel):
            hits = tree.query_ball_point(sel[['x', 'y', 'z']].values, 4.5)
            idx = sorted({i for h in hits for i in h})
            return others.iloc[idx]
        cw = contacts(w); cd = contacts(dom)
        def cls(c):
            if c not in abmap: return 'nonab'
            return 'partner' if abmap[c] == grp else 'otherab'
        for lab, cc in [('win', cw), ('dom', cd)]:
            k = cc.chain.map(cls)
            row[f'{lab}_contact_partner'] = int((k == 'partner').sum() > 0)
            row[f'{lab}_contact_otherab'] = int((k == 'otherab').sum() > 0)
            row[f'{lab}_contact_nonab'] = int((k == 'nonab').sum() > 0)
            row[f'{lab}_nonab_chains'] = ';'.join(sorted(set(cc.chain[k == 'nonab'])))
            row[f'{lab}_otherab_chains'] = ';'.join(sorted(set(cc.chain[k == 'otherab'])))
        # residue-level antigen contact count in window
        hits = cKDTree(others[others.chain.map(cls) == 'nonab'][['x', 'y', 'z']].values) if (others.chain.map(cls) == 'nonab').any() else None
        if hits is not None:
            cnt = 0
            for a, g in w.groupby('resid'):
                if any(len(h) for h in hits.query_ball_point(g[['x', 'y', 'z']].values, 4.5)): cnt += 1
            row['win_res_contact_nonab'] = cnt
        else: row['win_res_contact_nonab'] = 0
    row['partner_in_model'] = int(any(abmap_c != ch and ab_chains.get(pdb, {}).get(abmap_c) == ab_chains.get(pdb, {}).get(ch)
                                      for abmap_c in ab_chains.get(pdb, {})))
    return row

rows = []
todo = [('case', r.mode, r.pdb, r.pdb, r.chain, r.ctype, int(r.imgt0), int(r.imgt1), '') for r in cases.itertuples()]
for r in ctrls.itertuples():
    c0 = cases[(cases.pdb == r.case_pdb) & (cases['mode'] == r.mode)].iloc[0]
    m = re.search(r'\((\w+)\)', r.why)
    if m: ch = m.group(1)
    else:
        cand = C[(C.pdb == r.ctrl_pdb) & (C.ctype == r.ctype)]
        if c0.ctype == 'H': cand = cand.sort_values('vhh', ascending=bool(C[(C.pdb == c0.pdb) & (C.chain == c0.chain)].vhh.iloc[0] == 0))
        ch = cand.chain.iloc[0]
    todo.append(('control', r.mode, r.case_pdb, r.ctrl_pdb, ch, r.ctype, int(c0.imgt0), int(c0.imgt1), r.why))
cache = {}
for grp, mode, case_pdb, pdb, ch, ct, lo, hi, why in todo:
    if pdb not in cache:
        try: atoms = read_atoms(pdb)
        except Exception as e: print('atoms fail', pdb, e); atoms = pd.DataFrame(columns=['chain', 'resid', 'x', 'y', 'z', 'b', 'atom'])
        cache = {pdb: (atoms, read_vrpt(pdb), read_va(pdb))}
    atoms, vr, va = cache[pdb]
    try: row = site(pdb, ch, ct, lo, hi, atoms, vr, va)
    except Exception as e: row = dict(pdb=pdb, chain=ch, lo=lo, hi=hi, note=f'error {e}')
    row.update(group=grp, mode=mode, case_pdb=case_pdb, why=why, ctype=ct)
    rows.append(row); print(grp, mode, pdb, ch, lo, hi, {k: row.get(k) for k in ['win_seq', 'fit_pct', 'unbuilt_win5', 'win_contact_nonab', 'lr_win', 'global_res']}, flush=True)
L = pd.DataFrame(rows)
L.to_csv(f'{OUT}/tables/local_sites.csv', index=False)
print(L.shape)
