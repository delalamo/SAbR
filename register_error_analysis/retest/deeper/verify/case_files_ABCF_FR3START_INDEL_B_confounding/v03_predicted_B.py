"""Map-free artifact test for mode-B runs: does SAbR also disagree with ANARCI inside the run window on de novo PREDICTED
models (ImmuneBuilder, IgFold; pred/full) of the same antibody? A sequence-driven numbering artifact should reappear in a
prediction; a genuine register error of the deposited model should not.
All first-occurrence B_CDR2FR3 runs (any method) whose chain has a prediction; feature = ANARCI indel at IMGT 66-75 (SEQRES).
Output: tables/v03_predicted_B.csv"""
import os, json, warnings
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[_v] = "1"
warnings.filterwarnings('ignore')
import logging; logging.disable(logging.WARNING)
import pandas as pd
from Bio.PDB import PDBParser
from anarci import run_anarci
from sabr import renumber_structure
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
PR = f'{SP}/pred/full'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/case_files_ABCF_FR3START_INDEL_B_confounding'
A3 = dict(ALA='A', ARG='R', ASN='N', ASP='D', CYS='C', GLN='Q', GLU='E', GLY='G', HIS='H', ILE='I', LEU='L', LYS='K', MET='M', PHE='F',
          PRO='P', SER='S', THR='T', TRP='W', TYR='Y', VAL='V')
C = pd.read_pickle(f'{OUT}/tables/chain_features.pkl').drop_duplicates(['pdb', 'chain']).set_index(['pdb', 'chain'])
U = pd.DataFrame(json.load(open(f'{PR}/units.json'))); U = U[U.status == 'ok']
RF = pd.read_csv(f'{SP}/deeper/runs_first_occurrence.csv')
RF = RF[RF['mode'] == 'B_CDR2FR3']
M = RF.merge(U[['pdb', 'chain', 'fv_id', 'mode']].drop_duplicates(['pdb', 'chain']).rename(columns={'mode': 'fv_mode'}), on=['pdb', 'chain'])
M['ct2'] = M.ctype.map(lambda t: 'H' if t == 'H' else 'L')
M['feat'] = [C.fr3start_indel.get((p, c), float('nan')) for p, c in zip(M.pdb, M.chain)]
print('B runs', len(RF), 'with prediction', len(M), 'fv', M.fv_id.nunique())
cache = {}
def sabr_vs_anarci(fn, ch, ctype):
    k = (fn, ch)
    if k in cache: return cache[k]
    st = PDBParser(QUIET=True).get_structure('m', fn)
    chain = next(iter(st))[ch]
    res = [r for r in chain if r.id[0] == ' ']
    seq = ''.join(A3.get(r.get_resname(), 'X') for r in res)
    an = run_anarci([('q', seq)], scheme='imgt')
    numb, s0, s1 = an[1][0][0]
    nums = [(p, i.strip()) for (p, i), a in numb if a != '-']
    ct = 'H' if ctype == 'H' else an[2][0][0]['chain_type']
    new = renumber_structure(st, chain=ch, chain_type=ct)
    sab = [(r.id[1], r.id[2].strip()) for r in next(iter(new))[ch] if r.id[0] == ' ']
    pairs = [(seq[s0 + j], nums[j], sab[s0 + j]) for j in range(len(nums))]
    cache[k] = pairs
    return pairs
rows = []
for (fv, ct2, i0, i1), g in M.groupby(['fv_id', 'ct2', 'imgt0', 'imgt1']):
    r0 = g.iloc[0]
    for tool in ['immunebuilder', 'igfold']:
        fn = f'{PR}/{tool}/{fv}.pdb'
        base = dict(fv_id=fv, tool=tool, pdbs=','.join(sorted(set(g.pdb))), chain_type=r0.ctype, imgt0=i0, imgt1=i1, delta=r0.delta,
                    method=r0.method, last_author=r0.last_author, feat=g.feat.max(), origin=r0.origin)
        if not os.path.exists(fn): rows.append(dict(base, note='missing')); continue
        try:
            ch = 'H' if (ct2 == 'H' or r0.fv_mode == 'nanobody') else 'L'
            pairs = sabr_vs_anarci(fn, ch, 'H' if ct2 == 'H' else 'L')
            dis = [f'{a}{n[0]}{n[1]}/{s[0]}{s[1]}' for a, n, s in pairs if i0 <= n[0] <= i1 and n != s]
            dis_w = [1 for a, n, s in pairs if 55 <= n[0] <= 80 and n != s]
            rows.append(dict(base, n_dis_pred=len(dis), n_dis_55_80=len(dis_w), reproduced=len(dis) >= 3, detail=' '.join(dis)))
        except Exception as ex:
            rows.append(dict(base, note=f'{type(ex).__name__}: {str(ex)[:120]}'))
        print(rows[-1], flush=True)
D = pd.DataFrame(rows)
D.to_csv(f'{OUT}/tables/v03_predicted_B.csv', index=False)
print(D.groupby(['feat', 'tool']).agg(n=('reproduced', 'size'), reproduced=('reproduced', 'sum')))
