"""Verifier (confounding lens) for finding FR3_INDEL_C (case_files_ABCF).

Step 1: rebuild the investigator's per-chain features (ANARCI IMGT numbering of SEQRES, FR3 = IMGT 66-104 incl.
insertions) and per-unit table, independently of their script but with the same definitions, and add:
  - fr3_gap_pattern: positions 66-104 absent from the SEQRES numbering (minus chain-type norm gaps) + insertions
  - sabr pattern info from the SAbR per-residue output (modelled residues): where SAbR and ANARCI disagree
  - unit-level flags: first-occurrence C error, any C-region run (incl. copied/artifact), any run at all, other modes
Outputs: tables/chain_features.csv.gz, tables/units_em.csv.gz
"""
import json, pickle, numpy as np, pandas as pd
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
DE = '/home/user/SAbR/register_error_analysis/retest/deeper'
OUT = f'{DE}/verify/case_files_ABCF_FR3_INDEL_C_confounding'

C = pd.read_csv(f'{SP}/deeper/chains.csv.gz', low_memory=False)
amy = set(pd.read_csv(f'{DE}/pdb_meta/tables/amyloid_fibril_chains.csv').pdb.str.lower())
C = C[~C.pdb.isin(amy)].copy()
RF = pd.read_csv(f'{SP}/deeper/runs_first_occurrence.csv'); RF = RF[~RF.pdb.isin(amy)]
RA = pd.read_csv(f'{SP}/deeper/runs_all.csv'); RA = RA[~RA.pdb.isin(amy)]
AN = pickle.load(open(f'{SP}/deeper/cache/seq_novelty/anarci_num.pkl', 'rb'))


def number(seq, light):
    for d in AN.get(seq) or []:
        if (d[0] in 'KL') == light:
            return [(int(p), str(i).strip(), a) for p, i, a in d[4]]
    return None


NORMGAP = {'H': {73}, 'VHH': {73}, 'K': {73, 81, 82}, 'L': {73}}
C['cclass'] = np.where(C.ctype == 'H', np.where(C.vhh == 1, 'VHH', 'H'), C.ctype)
feat = []
for r in C.itertuples():
    nm = number(r.seq, r.ctype in 'KL') or number(r.vseq, r.ctype in 'KL')
    if not nm:
        feat.append({}); continue
    pos = {p for p, i, a in nm if i == ''}
    ins = sorted({f'{p}{i}' for p, i, a in nm if i != '' and 66 <= p <= 104})
    fr3 = sum(66 <= p <= 104 for p, i, a in nm)
    gaps = tuple(k for k in range(66, 105) if k not in pos and k not in NORMGAP[r.cclass])
    feat.append(dict(FR3=fr3, fr3_gaps=str(gaps), fr3_ins=','.join(ins), fr3_seq=''.join(a for p, i, a in nm if 66 <= p <= 104)))
F = pd.DataFrame(feat, index=C.index)
C = pd.concat([C, F], axis=1)
norm_fr3 = {'H': {38}, 'VHH': {38}, 'K': {36}, 'L': {36, 38}}
C['fr3_atyp'] = [int(f == f and int(f) not in norm_fr3[c]) for f, c in zip(C.FR3, C.cclass)]
C['fr3_short'] = [int(f == f and int(f) < min(norm_fr3[c])) for f, c in zip(C.FR3, C.cclass)]
C['fr3_pattern'] = [(g if isinstance(g, str) else '') + ('+' + i if isinstance(i, str) and i else '') for g, i in zip(C.fr3_gaps, C.fr3_ins)]

# run-level flags per chain
key = lambda d: list(zip(d.pdb, d.chain))
rf_modes = RF.groupby(['pdb', 'chain'])['mode'].agg(set).to_dict()
ra_modes = RA.groupby(['pdb', 'chain'])['mode'].agg(set).to_dict()
C['first_modes'] = [rf_modes.get(k, set()) for k in key(C)]
C['all_modes'] = [ra_modes.get(k, set()) for k in key(C)]
# runs overlapping the FR3 window 66-104 (any mode), first-occurrence and any
def overl(R, lo=66, hi=104):
    s = R[(R.imgt0 <= hi) & (R.imgt1 >= lo)]
    return set(zip(s.pdb, s.chain))
fo_fr3 = overl(RF); any_fr3 = overl(RA)
C['first_run_in_FR3win'] = [k in fo_fr3 for k in key(C)]
C['any_run_in_FR3win'] = [k in any_fr3 for k in key(C)]

# SAbR vs ANARCI on modelled residues within FR3 (74-100) and unbuilt residues in the FR3 window
sab = {}
for fn in ['new.jsonl', 'snapshot_rest.jsonl']:
    for line in open(f'{SP}/sabr/out/{fn}'):
        x = json.loads(line)
        if x.get('status') != 'ok' or not x.get('residues'):
            continue
        rec = x['residues']
        w = [r for r in rec if r[4] is not None and 66 <= r[4] <= 104]
        dis = [r for r in w if (r[4], r[5] or '') != (r[6], r[7] or '')]
        si = [r[3] for r in w]
        unb = (max(si) - min(si) + 1 - len(si)) if si else np.nan
        sab[(x['pdb'], x['chain'])] = dict(n_fr3_model=len(w), n_fr3_dis=len(dis), fr3_unbuilt_internal=unb,
                                          dis_pos=','.join(str(r[4]) + (r[5] or '') for r in dis))
S = pd.DataFrame([dict(pdb=k[0], chain=k[1], **v) for k, v in sab.items()])
C = C.merge(S, on=['pdb', 'chain'], how='left')
C.drop(columns=['seq']).to_csv(f'{OUT}/tables/chain_features.csv.gz', index=False)

# ---------- EM units (antibody V region in one paper)
E = C[C.method == 'EM'].copy()
E['ybin'] = pd.cut(E.year, [0, 2019, 2021, 2023, 2100], labels=['<=2019', '2020-21', '2022-23', '>=2024']).astype(str)
U = E.groupby('unit').agg(paper=('paper', 'first'), lab=('last_author', 'first'), year=('year', 'first'), ybin=('ybin', 'first'),
                          res=('res', 'mean'), cclass=('cclass', 'first'), sars2=('sars2', 'max'), err_any=('err_any', 'max'),
                          err_first=('err_first', 'max'), err_cao=('err_copied_or_artifact_only', 'max'),
                          first_modes=('first_modes', lambda s: set().union(*s)), all_modes=('all_modes', lambda s: set().union(*s)),
                          fr3_atyp=('fr3_atyp', 'max'), fr3_short=('fr3_short', 'max'), FR3=('FR3', 'first'),
                          fr3_pattern=('fr3_pattern', 'first'), fr3_seq=('fr3_seq', 'first'), v_gene=('v_gene', 'first'),
                          species=('species', 'first'), vhh=('vhh', 'max'),
                          first_run_in_FR3win=('first_run_in_FR3win', 'max'), any_run_in_FR3win=('any_run_in_FR3win', 'max'),
                          n_fr3_dis=('n_fr3_dis', 'max'), fr3_unbuilt_internal=('fr3_unbuilt_internal', 'max'),
                          n_chains=('chain', 'size'), pdb=('pdb', 'first'), chain=('chain', 'first')).reset_index()
U['C_first'] = U.first_modes.map(lambda s: 'C_FR3' in s).astype(int)
U['C_any'] = U.all_modes.map(lambda s: 'C_FR3' in s).astype(int)
U['first_modes'] = U.first_modes.map(lambda s: ';'.join(sorted(s)))
U['all_modes'] = U.all_modes.map(lambda s: ';'.join(sorted(s)))
U.to_csv(f'{OUT}/tables/units_em.csv.gz', index=False)
print('EM units', len(U), 'C_first', U.C_first.sum(), 'clean (err_any False)', (~U.err_any.astype(bool)).sum())
d = U[(U.C_first == 1) | (~U.err_any.astype(bool))]
d = d[d.cclass.isin(set(d[d.C_first == 1].cclass))]
print('investigator frame: C units', d.C_first.sum(), 'fr3_atyp', d[d.C_first == 1].fr3_atyp.sum(),
      '| controls', (d.C_first == 0).sum(), 'fr3_atyp', d[d.C_first == 0].fr3_atyp.sum())
