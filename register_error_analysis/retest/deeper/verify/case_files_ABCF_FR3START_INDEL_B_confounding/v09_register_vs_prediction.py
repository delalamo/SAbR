"""Numbering-free register test. Superpose each deposited V chain carrying a first-occurrence B run onto the map-free
ImmuneBuilder / IgFold model of the same sequence (framework CA outside the run window), with residue correspondence from the
sequence alignment (i.e. the true register). For each run residue: d_own = CA distance to the same residue in the prediction,
d_nb = distance to the nearest sequence neighbour (i-1 / i+1, or i-k / i+k for |delta| = k). A genuine register shift puts the
deposited residue next to a neighbour's position (d_nb < d_own for most of the run); a numbering artifact leaves it on its own
position (d_own small). Runs in EM (feature and non-feature) plus 7wvl (X-ray, documented template threading) as a positive control.
Output: tables/v09_register_vs_prediction.csv"""
import os, json, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from Bio.PDB import MMCIFParser, PDBParser
from Bio import Align
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/case_files_ABCF_FR3START_INDEL_B_confounding'
A3 = dict(ALA='A', ARG='R', ASN='N', ASP='D', CYS='C', GLN='Q', GLU='E', GLY='G', HIS='H', ILE='I', LEU='L', LYS='K', MET='M', PHE='F',
          PRO='P', SER='S', THR='T', TRP='W', TYR='Y', VAL='V', MSE='M')
CH = pd.read_pickle(f'{OUT}/tables/chain_features.pkl').drop_duplicates(['pdb', 'chain']).set_index(['pdb', 'chain'])
U = pd.DataFrame(json.load(open(f'{SP}/pred/full/units.json'))); U = U[U.status == 'ok']
RF = pd.read_csv(f'{SP}/deeper/runs_first_occurrence.csv')
RF = RF[(RF['mode'] == 'B_CDR2FR3') & ((RF.method == 'EM') | (RF.pdb == '7wvl'))]
M = RF.merge(U[['pdb', 'chain', 'fv_id', 'mode']].drop_duplicates(['pdb', 'chain']).rename(columns={'mode': 'fv_mode'}), on=['pdb', 'chain'])
M = M.drop_duplicates(['fv_id', 'imgt0', 'imgt1'])   # one chain per antibody/run
recs = {}
need = set(zip(M.pdb, M.chain))
for fn in ['new', 'snapshot_rest', 'validation']:
    for line in open(f'{SP}/sabr/out/{fn}.jsonl'):
        x = json.loads(line)
        if (x['pdb'], x['chain']) in need and x.get('residues'): recs[(x['pdb'], x['chain'])] = x['residues']
def cif_path(p):
    for f in [f'{SP}/sabr/dl/{p}.cif', f'{SP}/deeper/cache/case_files_ABCF/cif/{p}.cif']:
        if os.path.exists(f): return f
aligner = Align.PairwiseAligner(); aligner.mode = 'global'; aligner.open_gap_score = -8; aligner.extend_gap_score = -1
aligner.substitution_matrix = Align.substitution_matrices.load('BLOSUM62')
def kabsch(P, Q):
    Pc, Qc = P.mean(0), Q.mean(0); H = (P - Pc).T @ (Q - Qc); Uu, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ Uu.T)); D = np.diag([1, 1, d]); R = Vt.T @ D @ Uu.T
    return lambda X: (R @ (X - Pc).T).T + Qc
rows = []
pcache = {}
for r in M.itertuples():
    f = cif_path(r.pdb); rec = recs.get((r.pdb, r.chain))
    if f is None or rec is None: rows.append(dict(pdb=r.pdb, chain=r.chain, note='missing cif/rec')); continue
    try:
        if f not in pcache: pcache[f] = next(iter(MMCIFParser(QUIET=True).get_structure(r.pdb, f)))
        ch = pcache[f][r.chain]
        imgt = {(int(x[0]), (x[1] or ' ')): (x[4], x[5] or '') for x in rec}
        dep = [(res, imgt.get((res.id[1], res.id[2]))) for res in ch if res.id[0] == ' ' and res.get_resname() in A3 and 'CA' in res]
        dep = [(res, im) for res, im in dep if im is not None and im[0] is not None]
        dseq = ''.join(A3[res.get_resname()] for res, im in dep)
        pch = 'H' if (r.ctype == 'H' or r.fv_mode == 'nanobody') else 'L'
        for tool in ['immunebuilder', 'igfold']:
            pf = f'{SP}/pred/full/{tool}/{r.fv_id}.pdb'
            if not os.path.exists(pf): continue
            pc = [res for res in next(iter(PDBParser(QUIET=True).get_structure('m', pf)))[pch] if res.id[0] == ' ' and 'CA' in res]
            pseq = ''.join(A3.get(res.get_resname(), 'X') for res in pc)
            aln = aligner.align(dseq, pseq)[0]
            pairs = {}
            for (a0, a1), (b0, b1) in zip(*aln.aligned):
                for k in range(a1 - a0): pairs[a0 + k] = b0 + k
            win = [i for i, (res, im) in enumerate(dep) if r.imgt0 <= im[0] <= r.imgt1 and i in pairs]
            fw = [i for i, (res, im) in enumerate(dep) if i in pairs and not (r.imgt0 - 3 <= im[0] <= r.imgt1 + 3)
                  and not (27 <= im[0] <= 38 or 56 <= im[0] <= 65 or 105 <= im[0] <= 117) and im[0] <= 128]
            P = np.array([dep[i][0]['CA'].coord for i in fw]); Q = np.array([pc[pairs[i]]['CA'].coord for i in fw])
            tf = kabsch(P, Q); rmsd_fw = float(np.sqrt(((tf(P) - Q) ** 2).sum(1).mean()))
            k = int(abs(r.delta)) if r.delta == r.delta and r.delta != 0 else 1
            own, nb, closer = [], [], 0
            for i in win:
                x = tf(dep[i][0]['CA'].coord[None])[0]; j = pairs[i]
                do = float(np.linalg.norm(x - pc[j]['CA'].coord))
                dn = min([float(np.linalg.norm(x - pc[j + s]['CA'].coord)) for s in (-k, k) if 0 <= j + s < len(pc)] or [np.nan])
                own.append(do); nb.append(dn); closer += int(dn < do)
            rows.append(dict(pdb=r.pdb, chain=r.chain, fv_id=r.fv_id, tool=tool, method=r.method, imgt0=r.imgt0, imgt1=r.imgt1, delta=r.delta,
                             feat=int(np.nan_to_num(CH.fr3start_indel.get((r.pdb, r.chain), 0))), n_win=len(win), rmsd_fw=rmsd_fw,
                             med_d_own=np.median(own) if own else np.nan, med_d_nb=np.median(nb) if nb else np.nan,
                             frac_closer_to_neighbour=closer / len(win) if win else np.nan, origin=r.origin))
            print({k2: (round(v, 2) if isinstance(v, float) else v) for k2, v in rows[-1].items() if k2 != 'origin'}, flush=True)
    except Exception as ex:
        rows.append(dict(pdb=r.pdb, chain=r.chain, note=f'{type(ex).__name__}: {str(ex)[:150]}')); print(rows[-1])
T = pd.DataFrame(rows); T.to_csv(f'{OUT}/tables/v09_register_vs_prediction.csv', index=False)
