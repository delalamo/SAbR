"""Independent, expanded re-run of the high-resolution check (s09) using the scan's own protocol.

Differences from s09: ANARCI numbering comes from the full SEQRES (as in the scan, not the built-residue sequence);
the chain is trimmed to the ANARCI heavy V domain by author number (+-3, as the scan's FAST mode), so scFv chains
(VL-linker-VH) work; the full MMCIF2Dict is passed so unobserved residues are known to SAbR.
Covers every X-ray entry <= 3.3 A in the PDB-wide census (v01) with a claimed artifact pattern or a control pattern,
including the ones s09 skipped (7kbk, 9f18, 4o51, 4rzc, 5trp, 30bh, 5dhx, 6cf2, 1ghf, 5aaw) and 10 more FR3-gap-81
controls. One chain per entry (first heavy chain in the census).
Output: tables/v03_hires_sabr.csv (ANARCI default species = human+mouse HMMs, as in the scan);
        run with argv[2]=all for tables/v03b_hires_sabr_allspeciesHMM.csv (ANARCI allowed_species=None: rabbit/alpaca HMMs too).
"""
import os, sys, json, urllib.request, warnings, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[_v] = "1"
warnings.filterwarnings('ignore')
import logging; logging.disable(logging.WARNING)
import pandas as pd, numpy as np
from Bio.PDB import MMCIFParser
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
CA = f'{SP}/deeper/cache/verify_artifact_indel_conf/cif'
OLD = f'{SP}/deeper/cache/case_files_ABCF/hires_cif'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/case_files_ABCF_ARTIFACT_INDEL_confounding'
os.makedirs(CA, exist_ok=True)
A3 = dict(ALA='A', ARG='R', ASN='N', ASP='D', CYS='C', GLN='Q', GLU='E', GLY='G', HIS='H', ILE='I', LEU='L', LYS='K', MET='M', PHE='F',
          PRO='P', SER='S', THR='T', TRP='W', TYR='Y', VAL='V', MSE='M')
WIN = {'FR1_26': (20, 40)}
SPECIES = sys.argv[2] if len(sys.argv) > 2 else 'default'

def get_cif(p):
    for d in (CA, OLD):
        f = f'{d}/{p}.cif'
        if os.path.exists(f): return f
    f = f'{CA}/{p}.cif'
    data = urllib.request.urlopen(f'https://files.rcsb.org/download/{p.upper()}.cif', timeout=180).read()
    open(f, 'wb').write(data)
    return f

def job(args):
    pat, p, c, res = args
    from anarci import run_anarci
    from sabr import renumber_structure
    out = dict(pattern=pat, pdb=p, chain=c, res=res)
    try:
        f = get_cif(p)
        md = MMCIF2Dict(f)
        st = MMCIFParser(QUIET=True).get_structure(p, f)
        model = next(iter(st)); ch = model[c]
        # SEQRES with author numbering for this strand
        sc = list(zip(md['_pdbx_poly_seq_scheme.pdb_strand_id'], md['_pdbx_poly_seq_scheme.mon_id'],
                      md['_pdbx_poly_seq_scheme.pdb_seq_num'], md['_pdbx_poly_seq_scheme.pdb_ins_code']))
        sc = [(m, int(n), (i if i not in '.?' else '')) for s, m, n, i in sc if s == c]
        seq = ''.join(A3.get(m, 'X') for m, n, i in sc)
        an = run_anarci([('q', seq)], scheme='imgt', **({'allowed_species': None} if SPECIES == 'all' else {}))
        doms = an[1][0]; hd = [k for k, d in enumerate(an[2][0]) if d['chain_type'] == 'H']
        if not hd: out['note'] = 'no H domain'; return out
        k = hd[0]; numb, s0, s1 = doms[k]
        lab2an = {}
        j = s0
        for (pp, ii), aa in numb:
            if aa == '-': continue
            lab2an[(sc[j][1], sc[j][2])] = (pp, ii.strip()); j += 1
        nums = [sc[t][1] for t in range(s0, s1 + 1)]
        lo, hi = min(nums) - 3, max(nums) + 3
        tst = Structure(p); tst.header = dict(getattr(st, 'header', {}) or {}); tm = Model(model.id, model.serial_num); tst.add(tm); tc = ch.copy()
        for r in list(tc):
            if not (lo <= r.id[1] <= hi) or r.id[0] != ' ': tc.detach_child(r.id)
        tm.add(tc)
        t0 = time.time()
        new = renumber_structure(tst, chain=c, chain_type='H', mmcif_dict=md)
        out['sabr_s'] = round(time.time() - t0, 1)
        wlo, whi = WIN.get(pat, (74, 100))
        dis, nwin = [], 0
        for o, n in zip(list(tc), list(next(iter(new))[c])):
            a = lab2an.get((o.id[1], o.id[2].strip()))
            if a is None or not (wlo <= a[0] <= whi): continue
            nwin += 1
            if (a[0], a[1]) != (n.id[1], n.id[2].strip()):
                dis.append(f'{A3.get(o.get_resname(), "X")}{a[0]}{a[1]}/{n.id[1]}{n.id[2].strip()}')
        pos = {pp for (pp, ii), aa in numb if aa != '-' and ii.strip() == ''}
        miss = sorted(set(range(min(pos), 105)) - pos)
        out.update(n_win=nwin, n_dis=len(dis), detail=' '.join(dis),
                   an_miss_fr1=','.join(str(x) for x in miss if 11 <= x <= 26), an_miss_fr3=','.join(str(x) for x in miss if 74 <= x <= 100),
                   fr3_seq=''.join(A3.get(sc[t][0], 'X') for t in range(s0, s1 + 1))[70:100])
    except Exception as ex:
        out['note'] = f'{type(ex).__name__}: {str(ex)[:150]}'
    return out

if __name__ == '__main__':
    from multiprocessing import Pool
    CEN = pd.read_csv(f'{OUT}/tables/v01_pattern_census.csv')
    CEN = CEN[CEN.method.str.startswith('X-RAY') & (CEN.res <= 3.3)]
    CEN = CEN.sort_values(['pattern', 'res']).drop_duplicates(['pattern', 'pdb'])
    jobs = [(t, p, c, r) for t, p, c, r in zip(CEN.pattern, CEN.pdb, CEN.chain, CEN.res)]
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    rows = []
    with Pool(n) as pool:
        for r in pool.imap_unordered(job, jobs):
            print(r, flush=True); rows.append(r)
    pd.DataFrame(rows).sort_values(['pattern', 'res']).to_csv(f'{OUT}/tables/' + ('v03b_hires_sabr_allspeciesHMM.csv' if SPECIES == 'all' else 'v03_hires_sabr.csv'), index=False)
