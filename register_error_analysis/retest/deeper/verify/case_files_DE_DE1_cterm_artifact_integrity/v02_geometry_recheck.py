"""V02: independent re-implementation of the investigator's framework-superposition check, with different references and
a second (local) superposition core, for all 80 D/E units in tables/independent_check_DE_units.csv and a sample of
clean controls.

* Coordinates: per-chain CIF from the PDBe ModelServer (not the investigator's cache); numbering from the scan records
  (ANARCI = sequence-based; SAbR not used).
* References: three clean, complete template chains per chain type (SAbR == ANARCI at every residue, IMGT 1-128 built),
  none of them 5DT1/1N8Z. Reference slot = ANARCI IMGT position of the reference residue.
* Cores: 'global' = IMGT 4-25, 39-55, 66-104; 'local' = 39-55, 84-104 (strands C, C', E-end, F next to strand G).
  Positions within +-1 of any run (runs_all) are excluded from the core.
* For each ANARCI position 116-124 built in the model: nearest reference slot (rank shift) and its distance; consensus =
  median shift over the 3 references; d_own = distance to the reference slot of the same number.
* Verdict per unit (positions of the run inside 116-124): 'indeterminate' if median nearest distance > 2.5 A;
  'confirmed' if >= half the tested positions shift in the direction of the SAbR-ANARCI delta; else 'not confirmed'.
Output: tables/v02_geometry_units.csv, tables/v02_geometry_controls.csv
"""
import os, io, json, time, urllib.request, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from Bio.PDB import MMCIFParser
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
D = f'{SP}/deeper'
CA = f'{D}/cache/verify_DE1/ms'; os.makedirs(CA, exist_ok=True)
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/case_files_DE_DE1_cterm_artifact_integrity'
CORES = {'global': list(range(4, 26)) + list(range(39, 56)) + list(range(66, 105)),
         'local': list(range(39, 56)) + list(range(84, 105))}


def fetch_chain(p, c):
    f = f'{CA}/{p}_{c}.cif'
    if not (os.path.exists(f) and os.path.getsize(f) > 500):
        for url in (f'https://www.ebi.ac.uk/pdbe/model-server/v1/{p}/atoms?auth_asym_id={c}&encoding=cif',
                    f'https://models.rcsb.org/v1/{p}/atoms?auth_asym_id={c}&encoding=cif'):
            try:
                txt = urllib.request.urlopen(url, timeout=120).read().decode()
                if len(txt) > 500: open(f, 'w').write(txt); break
            except Exception:
                time.sleep(2)
    if not os.path.exists(f): return None
    st = MMCIFParser(QUIET=True).get_structure(p, f)
    m = next(iter(st))
    return m[c] if c in m else None


def ca_by_anarci(ch, residues):
    rm = {(r.id[1], r.id[2].strip()): r for r in ch if r.id[0] == ' '}
    out = {}
    for x in residues:
        if x[4] is None: continue
        r = rm.get((x[0], x[1]))
        if r is not None and 'CA' in r: out[(x[4], x[5])] = r['CA'].coord
    return out


def kabsch(P, Q):
    Pc, Qc = P.mean(0), Q.mean(0)
    H = (P - Pc).T @ (Q - Qc); U, S, Vt = np.linalg.svd(H); d = np.sign(np.linalg.det(Vt.T @ U.T))
    Rm = Vt.T @ np.diag([1, 1, d]) @ U.T
    return Rm, Qc - Pc @ Rm.T


def assess(model, ref, runpos, core):
    P, Q = [], []
    for n in CORES[core]:
        if n in runpos: continue
        if (n, '') in model and (n, '') in ref: P.append(model[(n, '')]); Q.append(ref[(n, '')])
    if len(P) < 30: return None
    P, Q = np.array(P), np.array(Q); Rm, t = kabsch(P, Q)
    rms = float(np.sqrt(((P @ Rm.T + t - Q) ** 2).sum(1).mean()))
    slots = sorted([k for k in ref if 105 <= k[0] <= 128], key=lambda k: (k[0], k[1]))
    S = np.array([ref[k] for k in slots]); rank = {k: i for i, k in enumerate(slots)}
    o = {'rms': rms}
    for n in range(116, 125):
        if (n, '') in model and (n, '') in rank:
            x = model[(n, '')] @ Rm.T + t; dd = np.linalg.norm(S - x, axis=1); j = int(dd.argmin())
            o[f'sh{n}'] = rank[slots[j]] - rank[(n, '')]; o[f'd{n}'] = float(dd[j]); o[f'own{n}'] = float(dd[rank[(n, '')]])
    return o


def load_records(keys):
    rec = {}
    for f in ['validation', 'snapshot_rest', 'new']:
        for l in open(f'{SP}/sabr/out/{f}.jsonl'):
            r = json.loads(l)
            k = (r['pdb'], r['chain'])
            if k in keys and r.get('status') == 'ok': rec[k] = r['residues']
    return rec


def load_template_records(keys):
    rec = {}
    for l in open(f'{SP}/retest/templates.jsonl'):
        r = json.loads(l)
        k = (r['pdb'], r['chain'])
        if k in keys and r.get('status') == 'ok': rec[k] = r['residues']
    return rec


if __name__ == '__main__':
    T = pd.read_csv('/home/user/SAbR/register_error_analysis/retest/deeper/case_files_DE/tables/independent_check_DE_units.csv')
    X = pd.read_pickle(f'{D}/cache/case_files_DE/X_struct.pkl')
    ctl = X[X.group == 'matched_clean'].drop_duplicates(['pdb', 'chain'])
    rng = np.random.default_rng(5)
    ctl = pd.concat([g.sample(min(len(g), 40), random_state=int(rng.integers(1e9))) for _, g in ctl.groupby('ctype')])
    REFS = {'H': [('1k4c', 'A'), ('1yy8', 'B'), ('4m48', 'H')], 'K': [('1k4c', 'B'), ('1yy8', 'A'), ('4m48', 'L')]}
    # lambda references: first three clean, complete lambda templates
    lam = []
    for l in open(f'{SP}/retest/templates.jsonl'):
        r = json.loads(l)
        if r.get('status') != 'ok' or r.get('chain_type') != 'L': continue
        v = r['residues']; nums = [x[4] for x in v if x[4] is not None]
        if nums and all((x[4], x[5]) == (x[6], x[7]) for x in v if x[4] is not None) and min(nums) <= 2 and max(nums) >= 127:
            lam.append((r['pdb'], r['chain']))
        if len(lam) == 3: break
    REFS['L'] = lam
    print('references', REFS, flush=True)
    trec = load_template_records({k for v in REFS.values() for k in v})
    REF = {}
    for ct, lst in REFS.items():
        REF[ct] = []
        for p, c in lst:
            ch = fetch_chain(p, c)
            REF[ct].append(ca_by_anarci(ch, trec[(p, c)]))
    RA = pd.read_csv(f'{D}/runs_all.csv')
    runpos = {}
    for r in RA.itertuples(): runpos.setdefault((r.pdb, r.chain), set()).update(range(int(r.imgt0) - 1, int(r.imgt1) + 2))
    keys = set(zip(T.pdb, T.chain)) | set(zip(ctl.pdb, ctl.chain))
    rec = load_records(keys)
    out = []
    jobs = [('error', r.pdb, r.chain, r.ctype) for r in T.itertuples()] + [('clean', r.pdb, r.chain, r.ctype) for r in ctl.itertuples()]
    t0 = time.time()
    for i, (grp, p, c, ct) in enumerate(jobs):
        row = dict(group=grp, pdb=p, chain=c, ctype=ct)
        if (p, c) not in rec: row['note'] = 'no record'; out.append(row); continue
        ch = fetch_chain(p, c)
        if ch is None: row['note'] = 'no coords'; out.append(row); continue
        m = ca_by_anarci(ch, rec[(p, c)])
        for core in CORES:
            res = [assess(m, ref, runpos.get((p, c), set()), core) for ref in REF['H' if ct == 'H' else ct]]
            res = [r for r in res if r]
            if not res: continue
            row[f'{core}_rms'] = float(np.median([r['rms'] for r in res]))
            for n in range(116, 125):
                sh = [r[f'sh{n}'] for r in res if f'sh{n}' in r]
                if sh:
                    row[f'{core}_sh{n}'] = float(np.median(sh)); row[f'{core}_d{n}'] = float(np.median([r[f'd{n}'] for r in res if f'd{n}' in r]))
                    row[f'{core}_own{n}'] = float(np.median([r[f'own{n}'] for r in res if f'own{n}' in r]))
        out.append(row)
        if i % 25 == 0: print(i, len(jobs), round(time.time() - t0), flush=True)
    O = pd.DataFrame(out)
    E = O[O.group == 'error'].merge(T[['pdb', 'chain', 'mode', 'delta', 'imgt0', 'imgt1', 'last_built_imgt', 'model_ends_le121', 'confirm']], on=['pdb', 'chain'])

    def verdict(r, core):
        lo, hi = max(116, int(r.imgt0)), min(124, int(r.imgt1))
        pos = [n for n in range(lo, hi + 1) if pd.notna(r.get(f'{core}_sh{n}'))]
        if not pos: return 'not testable'
        if np.median([r[f'{core}_d{n}'] for n in pos]) > 2.5: return 'indeterminate'
        k = sum(np.sign(r[f'{core}_sh{n}']) == np.sign(r.delta) for n in pos)
        return 'confirmed' if k >= max(1, len(pos) / 2) else 'not confirmed'
    for core in CORES: E[f'v_{core}'] = E.apply(lambda r: verdict(r, core), axis=1)
    E.to_csv(f'{OUT}/tables/v02_geometry_units.csv', index=False)
    O[O.group == 'clean'].to_csv(f'{OUT}/tables/v02_geometry_controls.csv', index=False)
    print('done', round(time.time() - t0))
