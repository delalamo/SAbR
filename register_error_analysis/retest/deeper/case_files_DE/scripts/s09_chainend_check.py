"""Side check (outside the D/E lens, prompted by the C-terminal artifact found in s02): are N-terminal FR1 'errors'
(mode A_FR1Nterm) reproduced by an independent framework superposition?  Same method as s02: superpose the model's
framework CA (ANARCI IMGT 39-55, 66-104, outside any run) on a clean high-resolution reference (5dt1 H/L, 1n8z A),
then assign each ANARCI 1-12 residue to the nearest reference CA slot (IMGT 1-15). A run is 'confirmed' if most of its
residues in 1-12 sit one or more slots away in the direction of the SAbR-ANARCI shift.
Controls: clean chains of the s02 sample (false-positive rate of the method at the N-terminus)."""
import sys, json
sys.path.insert(0, '/home/user/SAbR/register_error_analysis/retest/deeper/case_files_DE/scripts')
from common import *
import s02_struct as S2
from Bio.PDB import MMCIFParser
from concurrent.futures import ProcessPoolExecutor
FWN = list(range(39, 56)) + list(range(66, 105))

def ref_load():
    tp = {}
    for l in open(f'{SP}/retest/templates.jsonl'):
        r = json.loads(l)
        if r.get('status') == 'ok' and (r['pdb'], r['chain']) in {('1n8z', 'A'), ('5dt1', 'H'), ('5dt1', 'L')}: tp[(r['pdb'], r['chain'])] = r['residues']
    REF = {}
    for (p, c), ct in {('5dt1', 'H'): 'H', ('1n8z', 'A'): 'K', ('5dt1', 'L'): 'L'}.items():
        parser = MMCIFParser(QUIET=True); st = parser.get_structure(p, S2.open_cif(S2.cif_path(p))); ch = next(iter(st))[c]
        num = {(x[0], x[1]): (x[6], x[7]) for x in tp[(p, c)] if x[6] is not None}
        REF[ct] = {num[(r.id[1], r.id[2].strip())]: r['CA'].coord for r in ch if (r.id[1], r.id[2].strip()) in num and 'CA' in r}
    return REF

def job(a):
    p, reqs, REF = a
    out = []
    try:
        parser = MMCIFParser(QUIET=True); s = parser.get_structure(p, S2.open_cif(S2.cif_path(p))); m = next(iter(s))
        for c, ct, res, runpos, label in reqs:
            if c not in m: continue
            ref = REF[ct]; ch = m[c]
            v = [x for x in res if x[4] is not None]
            rm = {(r.id[1], r.id[2].strip()): r for r in ch if r.id[0] == ' '}
            an = {(x[4], x[5]): rm.get((x[0], x[1])) for x in v}
            P, Q = [], []
            for n in FWN:
                r = an.get((n, ''))
                if r is not None and 'CA' in r and (n, '') in ref and n not in runpos: P.append(r['CA'].coord); Q.append(ref[(n, '')])
            if len(P) < 30: continue
            P, Q = np.array(P), np.array(Q); Rm, t = S2.kabsch(P, Q)
            slots = sorted([k for k in ref if 1 <= k[0] <= 15 and not k[1]]); Sx = np.array([ref[k] for k in slots])
            row = dict(pdb=p, chain=c, label=label, first_imgt=min(x[4] for x in v))
            for n in range(1, 13):
                r = an.get((n, ''))
                if r is not None and 'CA' in r and (n, '') in ref:
                    x = r['CA'].coord @ Rm.T + t; d = np.linalg.norm(Sx - x, axis=1); j = int(d.argmin())
                    row[f'shift_{n}'] = slots[j][0] - n; row[f'd_{n}'] = float(d[j])
            out.append(row)
    except Exception as e:
        out.append(dict(pdb=p, chain='*', label=f'error {e}'))
    return out

if __name__ == '__main__':
    CH = chains(); R = runs(); RA = runs_all()
    REF = ref_load()
    A = R[R['mode'] == 'A_FR1Nterm']
    A = A.merge(CH[['pdb', 'chain', 'unit']], on=['pdb', 'chain']).sort_values(['pdb', 'chain']).drop_duplicates('unit')
    S = pd.read_csv(f'{CACHE}/struct_sample.csv'); Sc = S[S.group == 'matched_clean']
    runpos = {}
    for r in RA.itertuples(): runpos.setdefault((r.pdb, r.chain), set()).update(range(int(r.imgt0) - 1, int(r.imgt1) + 2))
    keys = set(zip(A.pdb, A.chain)) | set(zip(Sc.pdb, Sc.chain))
    rec = sabr_records(keys)
    ctype = dict(zip(zip(CH.pdb, CH.chain), CH.ctype))
    lab = {(p, c): 'A error' for p, c in zip(A.pdb, A.chain)}; lab.update({(p, c): 'clean' for p, c in zip(Sc.pdb, Sc.chain)})
    jobs = {}
    for (p, c), l in lab.items():
        if (p, c) in rec: jobs.setdefault(p, []).append((c, ctype[(p, c)], rec[(p, c)], runpos.get((p, c), set()), l))
    res = []
    with ProcessPoolExecutor(3) as ex:
        for o in ex.map(job, [(p, r, REF) for p, r in jobs.items()], chunksize=3): res += o
    O = pd.DataFrame(res)
    O = O.merge(A[['pdb', 'chain', 'imgt0', 'imgt1', 'delta', 'method']], on=['pdb', 'chain'], how='left')
    def conf(r):
        if r.label != 'A error': return None
        lo, hi = int(max(1, r.imgt0)), int(min(12, r.imgt1))
        vals = [r.get(f'shift_{n}') for n in range(lo, hi + 1)]; vals = [v for v in vals if pd.notna(v)]
        if not vals: return 'not testable'
        k = sum(np.sign(v) == np.sign(r.delta) for v in vals)
        return 'confirmed' if k >= max(1, len(vals) / 2) else 'not confirmed'
    O['confirm'] = O.apply(conf, axis=1)
    O['start_late'] = O.first_imgt > 1
    O.to_csv(f'{OUT}/tables/modeA_nterm_check.csv', index=False)
    sc = [f'shift_{n}' for n in range(1, 13)]
    cl = O[O.label == 'clean']
    print('clean chains', len(cl), 'any nonzero shift at IMGT 1-12:', round(((cl[sc].abs() > 0).sum(axis=1) > 0).mean(), 3),
          '>=2 residues nonzero:', round(((cl[sc].abs() > 0).sum(axis=1) >= 2).mean(), 3))
    a = O[O.label == 'A error']
    print(pd.crosstab([a.method, a.start_late], a.confirm, margins=True))
