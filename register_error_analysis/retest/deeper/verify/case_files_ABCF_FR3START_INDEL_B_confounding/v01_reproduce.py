"""Reproduce T7 (indel at IMGT 66-75 -> mode B) and build a chain/unit-level feature table for confounding checks.
Writes tables/chain_features.pkl (all chains, EM + X-ray, amyloid excluded) and tables/units_em.pkl."""
import pickle, re, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
DE = '/home/user/SAbR/register_error_analysis/retest/deeper'
OUT = f'{DE}/verify/case_files_ABCF_FR3START_INDEL_B_confounding'
C = pd.read_csv(f'{SP}/deeper/chains.csv.gz', low_memory=False)
amy = set(pd.read_csv(f'{DE}/pdb_meta/tables/amyloid_fibril_chains.csv').pdb.str.lower())
C = C[~C.pdb.isin(amy)].copy()
RF = pd.read_csv(f'{SP}/deeper/runs_first_occurrence.csv'); RF = RF[~RF.pdb.isin(amy)]
RA = pd.read_csv(f'{SP}/deeper/runs_all.csv'); RA = RA[~RA.pdb.isin(amy)]
AN = pickle.load(open(f'{SP}/deeper/cache/seq_novelty/anarci_num.pkl', 'rb'))
def number(seq, light):
    for d in AN.get(seq) or []:
        if (d[0] in 'KL') == light: return [(int(p), str(i).strip(), a) for p, i, a in d[4]]
    return None
def feats(seq, vseq, ct):
    nm = number(seq, ct in 'KL') or number(vseq, ct in 'KL')
    if not nm: return dict(numbered=0)
    pos = {p for p, i, a in nm if i == ''}
    ins66 = [f'{p}{i}' for p, i, a in nm if 66 <= p <= 75 and i != '']
    miss66 = sorted({k for k in range(66, 76) if k not in pos} - {73})
    cdr2 = sum(56 <= p <= 65 for p, i, a in nm)
    fr3 = sum(66 <= p <= 104 for p, i, a in nm)
    miss_fr3 = tuple(k for k in range(66, 105) if k not in pos)
    miss_cdr2 = tuple(k for k in range(56, 66) if k not in pos)
    ins_cdr2 = tuple(f'{p}{i}' for p, i, a in nm if 56 <= p <= 65 and i != '')
    s6680 = ''.join(a for p, i, a in nm if 62 <= p <= 80)
    return dict(numbered=1, fr3start_indel=int(bool(miss66) or bool(ins66)), miss66=tuple(miss66), ins66=tuple(ins66),
                cdr2_len_an=cdr2, fr3_len_an=fr3, miss_fr3=miss_fr3, miss_cdr2=miss_cdr2, ins_cdr2=ins_cdr2, win=s6680)
F = pd.DataFrame([feats(s, v, ct) for s, v, ct in zip(C.seq, C.vseq, C.ctype)], index=C.index)
C = pd.concat([C, F], axis=1)
C['cclass'] = np.where(C.ctype == 'H', np.where(C.vhh == 1, 'VHH', 'H'), C.ctype)
first_modes = RF.groupby(['pdb', 'chain'])['mode'].agg(set).to_dict()
all_modes = RA.groupby(['pdb', 'chain'])['mode'].agg(set).to_dict()
C['modes_first'] = [first_modes.get((p, c), set()) for p, c in zip(C.pdb, C.chain)]
C['modes_all'] = [all_modes.get((p, c), set()) for p, c in zip(C.pdb, C.chain)]
C['B_first'] = C.modes_first.map(lambda s: 'B_CDR2FR3' in s).astype(int)
C['B_any'] = C.modes_all.map(lambda s: 'B_CDR2FR3' in s).astype(int)
C.to_pickle(f'{OUT}/tables/chain_features.pkl')
print(C.groupby(['method', 'cclass']).fr3start_indel.agg(['sum', 'count', 'mean']))

E = C[C.method == 'EM'].copy()
E['ybin'] = pd.cut(E.year, [0, 2019, 2021, 2023, 2100], labels=['<=2019', '2020-21', '2022-23', '>=2024']).astype(str)
U = E.groupby('unit').agg(paper=('paper', 'first'), last_author=('last_author', 'first'), year=('year', 'first'), ybin=('ybin', 'first'),
                          res=('res', 'mean'), cclass=('cclass', 'first'), err_any=('err_any', 'max'), err_first=('err_first', 'max'),
                          modes=('modes_first', lambda s: set().union(*s)), modes_all=('modes_all', lambda s: set().union(*s)),
                          sars2=('sars2', 'max'), L2_atyp=('L2_atyp', 'max'), vhh=('vhh', 'max'),
                          vseq=('vseq', 'first'), ctype=('ctype', 'first'), seq=('seq', 'first'), pdb=('pdb', 'first'), chain=('chain', 'first'),
                          n_chains=('chain', 'size'), species=('species', 'first'), v_gene=('v_gene', 'first')).reset_index()
# investigator's feature uses the first chain's seq of the unit
U['fr3start_indel'] = [feats(s, v, ct).get('fr3start_indel', 0) for s, v, ct in zip(U.seq, U.vseq, U.ctype)]
fx = {k: feats(s, v, ct) for k, s, v, ct in zip(U.unit, U.seq, U.vseq, U.ctype)}
for col in ['miss66', 'ins66', 'cdr2_len_an', 'miss_cdr2', 'ins_cdr2', 'miss_fr3', 'win']:
    U[col] = [fx[k].get(col) for k in U.unit]
U['yB'] = U.modes.map(lambda s: 'B_CDR2FR3' in s).astype(int)
U['yB_all'] = U.modes_all.map(lambda s: 'B_CDR2FR3' in s).astype(int)
U.to_pickle(f'{OUT}/tables/units_em.pkl')
d = U[(U.yB == 1) | (U.err_any == False)]
d = d[d.cclass.isin(set(d[d.yB == 1].cclass))]
print('B error units', d.yB.sum(), 'with feature', d[d.yB == 1].fr3start_indel.sum())
print('clean units', (d.yB == 0).sum(), 'with feature', d[d.yB == 0].fr3start_indel.sum())
print(d[d.yB == 1].groupby('cclass').size())
cols = ['unit', 'pdb', 'chain', 'cclass', 'year', 'res', 'sars2', 'last_author', 'v_gene', 'species', 'L2_atyp', 'cdr2_len_an', 'miss66', 'ins66', 'miss_cdr2', 'ins_cdr2', 'win', 'err_first', 'err_any', 'modes', 'modes_all']
print(d[d.fr3start_indel == 1][cols].to_string())
# all EM units with the feature, by outcome
print('\nALL EM units with feature, by outcome:')
fe = U[U.fr3start_indel == 1].copy()
fe['outcome'] = np.where(fe.yB == 1, 'B_first', np.where(fe.err_first, 'other_first:' + fe.modes.astype(str), np.where(fe.err_any, 'copied/artifact only:' + fe.modes_all.astype(str), 'clean')))
print(fe[cols + ['outcome']].to_string())
fe[cols + ['outcome']].to_csv(f'{OUT}/tables/em_units_with_feature.csv', index=False)
