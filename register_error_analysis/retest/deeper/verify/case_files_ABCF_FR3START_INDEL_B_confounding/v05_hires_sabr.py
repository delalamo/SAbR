"""SAbR vs ANARCI on high-resolution X-ray structures (<2.6 A) carrying the SAME ANARCI 66-75 indel pattern (same chain type)
as the B error units (candidates from v04): 9lad_B (70A, like 8bw0), 9pyn_B (kappa del 66, like 9pyh), 6oca_C (67A-C, like 8r4d),
3se9_H (67A, like the clean Ward bnAb controls). Counts residues in IMGT 55-80 where the two disagree.
Output: tables/v05_hires_sabr.csv"""
import os, json, urllib.request, warnings
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[_v] = "1"
warnings.filterwarnings('ignore')
import logging; logging.disable(logging.WARNING)
import pandas as pd
from Bio.PDB import MMCIFParser
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
from anarci import run_anarci
from sabr import renumber_structure
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
CA = f'{SP}/deeper/cache/verify_fr3start_confounding/cif'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/case_files_ABCF_FR3START_INDEL_B_confounding'
os.makedirs(CA, exist_ok=True)
A3 = dict(ALA='A', ARG='R', ASN='N', ASP='D', CYS='C', GLN='Q', GLU='E', GLY='G', HIS='H', ILE='I', LEU='L', LYS='K', MET='M', PHE='F',
          PRO='P', SER='S', THR='T', TRP='W', TYR='Y', VAL='V')
picks = [('9lad', 'B', 'H', '8bw0 70A'), ('9pyn', 'B', 'K', '9pyh kappa del66'), ('6oca', 'C', 'H', '8r4d 67A-C'), ('3se9', 'H', 'H', 'Ward bnAb 67A (clean controls)')]
rows = []
for p, c, ct, why in picks:
    f = f'{CA}/{p}.cif'
    if not os.path.exists(f):
        open(f, 'wb').write(urllib.request.urlopen(f'https://files.rcsb.org/download/{p.upper()}.cif', timeout=120).read())
    try:
        st = MMCIFParser(QUIET=True).get_structure(p, f)
        model = next(iter(st)); ch = model[c]
        res = [x for x in ch if x.id[0] == ' ' and x.get_resname() in A3][:130]
        tst = Structure(p); tst.header = dict(getattr(st, 'header', {}) or {}); tm = Model(0); tst.add(tm); tc = ch.copy()
        keep = {x.id for x in res}
        for x in list(tc):
            if x.id not in keep: tc.detach_child(x.id)
        tm.add(tc)
        seq = ''.join(A3[x.get_resname()] for x in res)
        an = run_anarci([('q', seq)], scheme='imgt')
        numb, s0, s1 = an[1][0][0]
        num = [(pp, ii.strip()) for (pp, ii), aa in numb if aa != '-']
        new = renumber_structure(tst, chain=c, chain_type=ct)
        sab = [(x.id[1], x.id[2].strip()) for x in next(iter(new))[c] if x.id[0] == ' ']
        pairs = list(zip(num, sab[s0:s0 + len(num)]))
        # unbuilt residues inside the window (gap in author numbering of modelled residues)
        win = [x for x, (a, b) in zip(res[s0:s0 + len(num)], pairs) if 55 <= a[0] <= 80]
        miss = tuple(k for k in range(66, 76) if k not in {a[0] for a, b in pairs if a[1] == ''} and k != 73)
        ins = tuple(f'{a[0]}{a[1]}' for a, b in pairs if 66 <= a[0] <= 75 and a[1])
        dis = [f'{seq[s0+i]}{a[0]}{a[1]}/{b[0]}{b[1]}' for i, (a, b) in enumerate(pairs) if 55 <= a[0] <= 80 and a != b]
        rows.append(dict(pdb=p, chain=c, ct=ct, compare_to=why, miss66=miss, ins66=ins, n_dis_55_80=len(dis), detail=' '.join(dis),
                         window_seq=''.join(seq[s0 + i] for i, (a, b) in enumerate(pairs) if 60 <= a[0] <= 78)))
    except Exception as ex:
        rows.append(dict(pdb=p, chain=c, compare_to=why, note=f'{type(ex).__name__}: {str(ex)[:150]}'))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(f'{OUT}/tables/v05_hires_sabr.csv', index=False)
