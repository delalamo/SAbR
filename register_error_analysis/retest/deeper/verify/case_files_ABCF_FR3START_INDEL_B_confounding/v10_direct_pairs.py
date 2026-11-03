"""Numbering-free pairwise register checks with experimental references (same sequence):
 9om5 chain c (B run) vs chains a/e/i (same sequence, same map, no run); 9om5 g,k vs a.
 9pyh L (EM, B run 67-69) vs 9pyn B (same antibody, X-ray 2.34 A).
 8jag H vs 8jam H / 8jap H (same antibody, same paper; all carry the run) - consistency only.
Same metric as v09: framework superposition, per-residue CA distance to own residue vs to the i+-1 neighbour in the reference.
Output: tables/v10_direct_pairs.csv"""
import os, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from Bio.PDB import MMCIFParser
from Bio import Align
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/case_files_ABCF_FR3START_INDEL_B_confounding'
A3 = dict(ALA='A', ARG='R', ASN='N', ASP='D', CYS='C', GLN='Q', GLU='E', GLY='G', HIS='H', ILE='I', LEU='L', LYS='K', MET='M', PHE='F',
          PRO='P', SER='S', THR='T', TRP='W', TYR='Y', VAL='V', MSE='M')
def path(p):
    for f in [f'{SP}/sabr/dl/{p}.cif', f'{SP}/deeper/cache/case_files_ABCF/cif/{p}.cif', f'{SP}/deeper/cache/verify_fr3start_confounding/cif/{p}.cif']:
        if os.path.exists(f): return f
S = {}
def chain(p, c):
    if p not in S: S[p] = next(iter(MMCIFParser(QUIET=True).get_structure(p, path(p))))
    return [r for r in S[p][c] if r.id[0] == ' ' and r.get_resname() in A3 and 'CA' in r][:125]
al = Align.PairwiseAligner(); al.mode = 'global'; al.open_gap_score = -8; al.extend_gap_score = -1
al.substitution_matrix = Align.substitution_matrices.load('BLOSUM62')
def kabsch(P, Q):
    Pc, Qc = P.mean(0), Q.mean(0); H = (P - Pc).T @ (Q - Qc); U, s, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T)); R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return lambda X: (R @ (X - Pc).T).T + Qc
def compare(p, c, rp, rc, motif_start, n, label):
    a, b = chain(p, c), chain(rp, rc)
    sa = ''.join(A3[r.get_resname()] for r in a); sb = ''.join(A3[r.get_resname()] for r in b)
    aln = al.align(sa, sb)[0]; pairs = {}
    for (a0, a1), (b0, b1) in zip(*aln.aligned):
        for k in range(a1 - a0): pairs[a0 + k] = b0 + k
    i0 = sa.find(motif_start); win = [i for i in range(i0, i0 + n) if i in pairs]
    fw = [i for i in pairs if not (i0 - 12 <= i <= i0 + n + 3)]
    P = np.array([a[i]['CA'].coord for i in fw]); Q = np.array([b[pairs[i]]['CA'].coord for i in fw]); tf = kabsch(P, Q)
    rm = float(np.sqrt(((tf(P) - Q) ** 2).sum(1).mean()))
    own, nb = [], []
    for i in win:
        x = tf(a[i]['CA'].coord[None])[0]; j = pairs[i]
        own.append(float(np.linalg.norm(x - b[j]['CA'].coord)))
        nb.append(min(float(np.linalg.norm(x - b[j + s]['CA'].coord)) for s in (-1, 1) if 0 <= j + s < len(b)))
    r = dict(label=label, query=f'{p}_{c}', reference=f'{rp}_{rc}', window=sa[i0:i0 + n], rmsd_fw=rm, med_d_own=np.median(own),
             med_d_nb=np.median(nb), frac_closer_to_neighbour=np.mean(np.array(nb) < np.array(own)),
             per_res=' '.join(f'{sa[i]}{o:.1f}/{m:.1f}' for i, o, m in zip(win, own, nb)))
    print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()}); return r
rows = []
# 9om5 heavy V: run 63-72 = 'LNYKYNPSVR' (IMGT 63-72 per ANARCI)
for c in ['c', 'g', 'k', 'e', 'i']:
    rows.append(compare('9om5', c, '9om5', 'a', 'LNYKYNPSVR', 10, f'9om5 {c} vs a (same map); c/g/k carry the run, e/i do not'))
rows.append(compare('9om5', 'c', '9om5', 'e', 'LNYKYNPSVR', 10, '9om5 c vs e'))
# 9pyh L vs 9pyn B (X-ray 2.34 A same antibody); window 64-72 region 'STLNGVPSR'
rows.append(compare('9pyh', 'L', '9pyn', 'B', 'STLNGVPSR', 9, '9pyh EM vs 9pyn X-ray 2.34 A, same antibody'))
# 8jag vs 8jam/8jap (same antibody, all carry the run): expect agreement
for q in ['8jam', '8jap']:
    if path(q): rows.append(compare('8jag', 'H', q, 'H', 'PTYAGFTGRF', 10, f'8jag vs {q} (same antibody, same paper)'))
pd.DataFrame(rows).to_csv(f'{OUT}/tables/v10_direct_pairs.csv', index=False)
