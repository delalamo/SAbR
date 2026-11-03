"""Run SAbR vs ANARCI (same protocol as the investigator's s09: observed residues, V domain trimmed) on the same-V reference
structures used in v05b, to tell 'copy of an error' from 'numbering artifact': 7m3n H (2.4 A EM, same Fab as 7m3l),
8tb7 N (2.94 A EM, same nanobody as 9yfu), 7y0n C (clean twin of 7y0o), 7te4 B (2.46 A X-ray, same Fab as 7te9),
and the query chains themselves for comparison. Output tables/v07_sabr_refs.csv"""
import os, warnings, pandas as pd
warnings.filterwarnings('ignore')
from Bio.PDB import MMCIFParser
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
from anarci import run_anarci
from sabr import renumber_structure
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/case_files_ABCF_FR3_INDEL_C_integrity'
DIRS = [f'{SP}/sabr/dl', f'{SP}/deeper/cache/case_files_ABCF/hires_cif', f'{SP}/deeper/cache/verify_fr3_integrity/cif']
A3 = dict(ALA='A', ARG='R', ASN='N', ASP='D', CYS='C', GLN='Q', GLU='E', GLY='G', HIS='H', ILE='I', LEU='L', LYS='K', MET='M', PHE='F',
          PRO='P', SER='S', THR='T', TRP='W', TYR='Y', VAL='V')
def path(p):
    for d in DIRS:
        if os.path.exists(f'{d}/{p}.cif'): return f'{d}/{p}.cif'
rows = []
for p, c in [('7m3n', 'H'), ('7m3l', 'H2'), ('8tb7', 'N'), ('9yfu', 'N'), ('7y0n', 'C'), ('7y0o', 'H'), ('7te4', 'B'), ('7te9', 'H')]:
    st = MMCIFParser(QUIET=True).get_structure(p, path(p)); ch = next(iter(st))[c]
    res = [x for x in ch if x.id[0] == ' ' and x.get_resname() in A3]
    seq = ''.join(A3[x.get_resname()] for x in res)
    for spc in [['human', 'mouse'], None]:
        an = run_anarci([('q', seq)], scheme='imgt', allowed_species=spc) if spc else run_anarci([('q', seq)], scheme='imgt')
        num = [(pp, ii.strip()) for (pp, ii), aa in an[1][0][0][0] if aa != '-']
        st0, en = an[1][0][0][1], an[1][0][0][2]
        keep = {x.id for x in res[st0:en + 1]}
        tst = Structure(p); tst.header = dict(getattr(st, 'header', {}) or {}); tm = Model(0); tst.add(tm); tc = ch.copy()
        for x in list(tc):
            if x.id not in keep: tc.detach_child(x.id)
        tm.add(tc)
        new = renumber_structure(tst, chain=c, chain_type='H')
        sab = [(x.id[1], x.id[2].strip()) for x in next(iter(new))[c] if x.id[0] == ' ']
        pairs = list(zip(num, sab))
        dis = [(seq[st0 + i], a, b) for i, (a, b) in enumerate(pairs) if 66 <= a[0] <= 117 and a != b]
        rows.append(dict(pdb=p, chain=c, anarci_species='human+mouse' if spc else 'all', n_obs=len(num), n_dis_66_117=len(dis),
                         window=f'{min(a[0] for _, a, _ in dis)}-{max(a[0] for _, a, _ in dis)}' if dis else '',
                         detail=' '.join(f'{s}{a[0]}{a[1]}/{b[0]}{b[1]}' for s, a, b in dis)[:250]))
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(f'{OUT}/tables/v07_sabr_refs.csv', index=False)
