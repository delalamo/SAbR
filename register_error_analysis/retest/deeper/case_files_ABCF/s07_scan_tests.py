"""Scan-wide tests of the specifics that recurred in the case files.

Unit = antibody V region in one paper (chains.csv 'unit'), cryo-EM only, AL amyloid fibrils excluded.
  error unit (mode m): some chain of the unit carries a first-occurrence run of mode m
  control unit: no chain of the unit has a disagreement run of any kind (err_any False)
Model: logit(error) ~ feature + year bin + resolution + chain class (H / VHH / K / L), cluster-robust SE by paper.

Features (all from the SEQRES-based ANARCI IMGT numbering, i.e. independent of the deposited model):
  T1  C_FR3   atypical FR3 length (66-104): heavy != 38, kappa != 36, lambda not in {36, 38}
  T2  C_FR3   FR3 shorter than the chain-type norm (deletion)
  T3  F_CDR1  internal FR1 gap at IMGT 26 or elsewhere 11-26 beyond the chain-type norm ("FR1/CDR1 deletion")
  T4  F_CDR1  same, F runs with an undefined shift only (the 27-35 "NaN-delta" pattern)
  T5  A_FR1Nterm  rabbit-like heavy N-terminus (ANARCI all-species call = rabbit, or SEQRES starting at IMGT 3 with
                   QS[VL]EESGG / [QE]S[VL]EE motif)
  T6  any mode  "near-copy": an earlier-deposited chain with 90-99.9% V identity (not identical) carrying a run at the
                 same IMGT location with the same shift (A,B,C,F) - exposure counted for error and control units alike
                 as "an earlier near-identical V carries a run at this mode's location"
Outputs: tables/scan_tests.csv, tables/near_copies.csv
"""
import pickle, re, numpy as np, pandas as pd, statsmodels.formula.api as smf, warnings
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/case_files_ABCF'
DE = '/home/user/SAbR/register_error_analysis/retest/deeper'
C = pd.read_csv(f'{SP}/deeper/chains.csv.gz', low_memory=False)
amy = set(pd.read_csv(f'{DE}/pdb_meta/tables/amyloid_fibril_chains.csv').pdb.str.lower())
C = C[~C.pdb.isin(amy)].copy()
RF = pd.read_csv(f'{SP}/deeper/runs_first_occurrence.csv'); RF = RF[~RF.pdb.isin(amy)]
RA = pd.read_csv(f'{SP}/deeper/runs_all.csv'); RA = RA[~RA.pdb.isin(amy)]
AN = pickle.load(open(f'{SP}/deeper/cache/seq_novelty/anarci_num.pkl', 'rb'))
NOV = pd.read_csv(f'{DE}/seq_novelty/tables/chain_novelty.csv.gz', usecols=['pdb', 'chain', 'q_species_all'], low_memory=False)
C = C.merge(NOV, on=['pdb', 'chain'], how='left')

def number(seq, light):
    for d in AN.get(seq) or []:
        if (d[0] in 'KL') == light: return [(int(p), str(i).strip(), a) for p, i, a in d[4]]
    return None
feat = []
for r in C.itertuples():
    nm = number(r.seq, r.ctype in 'KL') or number(r.vseq, r.ctype in 'KL')
    if not nm: feat.append({}); continue
    ps = {p for p, i, a in nm}
    first = min(ps)
    f = dict(FR3=sum(66 <= p <= 104 for p, i, a in nm), first=first,
             fr1_gaps=tuple(k for k in range(max(first, 11), 27) if k not in ps),
             nterm=''.join(a for p, i, a in nm if p <= 12))
    feat.append(f)
F = pd.DataFrame(feat, index=C.index); C = pd.concat([C, F], axis=1)
C['cclass'] = np.where(C.ctype == 'H', np.where(C.vhh == 1, 'VHH', 'H'), C.ctype)
# chain-type norms
norm_fr3 = {'H': {38}, 'VHH': {38}, 'K': {36}, 'L': {36, 38}}
C['fr3_atyp'] = [int(f == f and int(f) not in norm_fr3[c]) for f, c in zip(C.FR3, C.cclass)]
C['fr3_short'] = [int(f == f and int(f) < min(norm_fr3[c])) for f, c in zip(C.FR3, C.cclass)]
modal_gaps = C.groupby('cclass').fr1_gaps.agg(lambda s: s.dropna().value_counts().index[0]).to_dict()
C['fr1_gap_atyp'] = [int(isinstance(g, tuple) and g != modal_gaps[c]) for g, c in zip(C.fr1_gaps, C.cclass)]
C['rabbit_nterm'] = [int((c in ('H',)) and ((sp == 'rabbit') or (f == 3 and bool(re.match(r'[QE]?S[VL]EE', str(n) or ''))))) for c, sp, f, n in zip(C.cclass, C.q_species_all, C['first'], C.nterm)]
print('modal FR1 gaps', modal_gaps)
print(C.groupby('cclass')[['fr3_atyp', 'fr3_short', 'fr1_gap_atyp', 'rabbit_nterm']].mean().round(4))

# ---------- units
E = C[C.method == 'EM'].copy()
E['ybin'] = pd.cut(E.year, [0, 2019, 2021, 2023, 2100], labels=['<=2019', '2020-21', '2022-23', '>=2024']).astype(str)
first_modes = RF.groupby(['pdb', 'chain'])['mode'].agg(set).to_dict()
nan_F = set(zip(RF[(RF['mode'] == 'F_CDR1') & RF.delta.isna()].pdb, RF[(RF['mode'] == 'F_CDR1') & RF.delta.isna()].chain))
E['modes'] = [first_modes.get((p, c), set()) for p, c in zip(E.pdb, E.chain)]
E['nanF'] = [(p, c) in nan_F for p, c in zip(E.pdb, E.chain)]
U = E.groupby('unit').agg(paper=('paper', 'first'), year=('year', 'first'), ybin=('ybin', 'first'), res=('res', 'mean'),
                          cclass=('cclass', 'first'), err_any=('err_any', 'max'), modes=('modes', lambda s: set().union(*s)),
                          nanF=('nanF', 'max'), fr3_atyp=('fr3_atyp', 'max'), fr3_short=('fr3_short', 'max'),
                          fr1_gap_atyp=('fr1_gap_atyp', 'max'), rabbit_nterm=('rabbit_nterm', 'max'), vseq=('vseq', 'first'),
                          ctype=('ctype', 'first'), seq=('seq', 'first'), deposit=('deposit', 'min'), pdb=('pdb', 'first'), chain=('chain', 'first')).reset_index()
rows = []
def test(name, mode, x, sub=None):
    d = U.copy()
    d['y'] = d.modes.map(lambda s: mode in s).astype(int)
    if sub is not None: d['y'] = d['y'] & sub(d)
    d = d[(d.y == 1) | (d.err_any == False)]
    d = d[d.cclass.isin(set(d[d.y == 1].cclass))]
    ne, nc = int(d.y.sum()), int((d.y == 0).sum())
    xe, xc = d[d.y == 1][x].mean(), d[d.y == 0][x].mean()
    zero_cell = min(int(d[d.y == 1][x].sum()), ne - int(d[d.y == 1][x].sum()), int(d[d.y == 0][x].sum()), nc - int(d[d.y == 0][x].sum())) == 0
    try:
        if zero_cell: raise ValueError('zero cell: Fisher exact instead')
        m = smf.logit(f'y ~ {x} + ybin + res + cclass', data=d).fit(disp=0, cov_type='cluster', cov_kwds={'groups': pd.factorize(d.paper)[0]})
        orr, lo, hi, p = np.exp(m.params[x]), *np.exp(m.conf_int().loc[x]), m.pvalues[x]
    except Exception as ex:
        if not zero_cell: print('logit failed', name, repr(ex)[:200])
        from scipy.stats import fisher_exact
        t = [[int(d[d.y == 1][x].sum()), ne - int(d[d.y == 1][x].sum())], [int(d[d.y == 0][x].sum()), nc - int(d[d.y == 0][x].sum())]]
        orr, p = fisher_exact(t); lo = hi = np.nan; name += ' (Fisher exact, unadjusted)'
    rows.append(dict(test=name, mode=mode, feature=x, n_error=ne, n_control=nc, error_value=f'{xe:.1%} ({int(round(xe*ne))}/{ne})',
                     control_value=f'{xc:.2%} ({int(round(xc*nc))}/{nc})', OR=orr, lo=lo, hi=hi, p=p))
test('T1 atypical FR3 length -> C_FR3', 'C_FR3', 'fr3_atyp')
test('T2 FR3 shorter than norm -> C_FR3', 'C_FR3', 'fr3_short')
test('T1b atypical FR3 length -> B_CDR2FR3', 'B_CDR2FR3', 'fr3_atyp')
test('T3 atypical FR1 gap (IMGT 11-26) -> F_CDR1', 'F_CDR1', 'fr1_gap_atyp')
test('T4 atypical FR1 gap -> F_CDR1 runs with undefined shift', 'F_CDR1', 'fr1_gap_atyp', sub=lambda d: d.nanF.astype(int))
test('T4b atypical FR1 gap -> A_FR1Nterm', 'A_FR1Nterm', 'fr1_gap_atyp')
test('T5 rabbit-like heavy N-terminus -> A_FR1Nterm', 'A_FR1Nterm', 'rabbit_nterm')

# ---------- T6 near copies
POS = list(range(1, 129))
def vec(seq, light):
    nm = number(seq, light)
    v = np.zeros(128, dtype=np.uint8)
    if nm:
        for p, i, a in nm:
            if i == '' and 1 <= p <= 128: v[p - 1] = ord(a)
    return v
Cc = C.drop_duplicates(['pdb', 'chain']).set_index(['pdb', 'chain'])
RA['dep'] = pd.to_datetime(RA.deposit)
modes = ['A_FR1Nterm', 'B_CDR2FR3', 'C_FR3', 'F_CDR1', 'Dplus_CDR3short', 'Dminus_CDR3long', 'E_FR4bulge']
RA2 = RA[RA['mode'].isin(modes)].copy()
RA2['vseq'] = [Cc.vseq.get((p, c)) for p, c in zip(RA2.pdb, RA2.chain)]
RA2 = RA2.dropna(subset=['vseq'])
pred = RA2.drop_duplicates(['vseq', 'mode', 'delta', 'imgt0'])
pred = pred.assign(seq=[Cc.seq.get((p, c)) for p, c in zip(pred.pdb, pred.chain)])
pv = np.stack([vec(s, ct in 'KL') if (number(s, ct in 'KL')) else vec(v, ct in 'KL') for s, v, ct in zip(pred.seq, pred.vseq, pred.ctype)])
pl = pred.ctype.isin(['K', 'L']).values
U['dep'] = pd.to_datetime(U.deposit)
uv = np.stack([vec(s, ct in 'KL') if (number(s, ct in 'KL')) else vec(v, ct in 'KL') for s, v, ct in zip(U.seq, U.vseq, U.ctype)])
print('vectors with residues: pred', (pv > 0).sum(1).mean(), 'units', (uv > 0).sum(1).mean())
ul = U.ctype.isin(['K', 'L']).values
near = []
for i in range(len(U)):
    same = pl == ul[i]
    a = uv[i]; b = pv[same]
    both = (a > 0) & (b > 0)
    idn = ((a == b) & both).sum(1) / np.maximum(both.sum(1), 1)
    cand = pred[same][(idn >= 0.90) & (idn < 0.999) & (both.sum(1) >= 80)]
    cand = cand[(cand.dep < U.dep.iloc[i]) & (cand.vseq != U.vseq.iloc[i])]
    for c in cand.itertuples():
        near.append(dict(unit=U.unit.iloc[i], pdb=U.pdb.iloc[i], pred_pdb=c.pdb, pred_chain=c.chain, mode=c.mode, delta=c.delta, imgt0=c.imgt0,
                         imgt1=c.imgt1, ident=float(idn[(pred[same].index == c.Index)][0]), pred_lab=c.last_author))
NC = pd.DataFrame(near); NC.to_csv(f'{OUT}/tables/near_copies.csv', index=False)
# does the error unit's own run match the predecessor's (same mode, same shift, overlapping)?
own = RF.copy(); own['unit'] = [Cc.unit.get((p, c)) for p, c in zip(own.pdb, own.chain)]
match = []
for r in own.itertuples():
    x = NC[(NC.unit == r.unit) & (NC['mode'] == r.mode)]
    x = x[(x.imgt0 <= r.imgt1) & (x.imgt1 >= r.imgt0) & ((x.delta == r.delta) | (x.delta.isna() & pd.isna(r.delta)))]
    if len(x): match.append(dict(unit=r.unit, pdb=r.pdb, chain=r.chain, mode=r.mode, imgt0=r.imgt0, imgt1=r.imgt1, delta=r.delta,
                                 pred=';'.join(sorted(set(x.pred_pdb))), ident=x.ident.max(), pred_lab=x.pred_lab.iloc[0], lab=r.last_author))
MT = pd.DataFrame(match); MT.to_csv(f'{OUT}/tables/near_copy_matches.csv', index=False)
for m in ['A_FR1Nterm', 'B_CDR2FR3', 'C_FR3', 'F_CDR1', 'any']:
    if m == 'any':
        U['x'] = U.unit.isin(set(NC.unit)).astype(int); sub = None
    else:
        U['x'] = U.unit.isin(set(NC[NC['mode'] == m].unit)).astype(int)
    U[f'nearpred_{m}'] = U.x
for m in ['A_FR1Nterm', 'B_CDR2FR3', 'C_FR3', 'F_CDR1']:
    test(f'T6 earlier near-identical V (90-99.9%) carries a {m} run -> {m}', m, f'nearpred_{m}')
T = pd.DataFrame(rows)
T['q'] = multipletests(T.p.fillna(1), method='fdr_bh')[1]
T.to_csv(f'{OUT}/tables/scan_tests.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 70)
print(T.round(4).to_string())
print('near-copy matched first-occurrence runs:', len(MT), 'units', MT.unit.nunique() if len(MT) else 0)
print(MT.to_string() if len(MT) else '')

# ---------- follow-up after s08-s10: remove artifact-explained units, and test FR3-start indels for B
try:
    AR = pd.read_csv(f'{OUT}/tables/artifact_runs.csv')
    art_units = {Cc.unit.get((p, c)) for p, c in zip(AR.pdb, AR.chain)}
    def fr3start_indel(seq, vseq, ct):
        nm = number(seq, ct in 'KL') or number(vseq, ct in 'KL')
        if not nm: return 0
        pos = {p for p, i, a in nm if i == ''}
        ins = any(66 <= p <= 75 and i != '' for p, i, a in nm)
        norm = {73}
        miss = {k for k in range(66, 76) if k not in pos} - norm
        return int(bool(miss) or ins)
    U['fr3start_indel'] = [fr3start_indel(s, v, ct) for s, v, ct in zip(U.seq, U.vseq, U.ctype)]
    U0 = U.copy()
    U = U0[~U0.unit.isin(art_units)].copy()
    test('T1-excl atypical FR3 length -> C_FR3, artifact-explained units removed', 'C_FR3', 'fr3_atyp')
    test('T2-excl FR3 shorter than norm -> C_FR3, artifact-explained units removed', 'C_FR3', 'fr3_short')
    test('T7 indel at the CDR2/FR3 boundary (IMGT 66-75) -> B_CDR2FR3', 'B_CDR2FR3', 'fr3start_indel')
    test('T7b indel at IMGT 66-75 -> C_FR3 (location control)', 'C_FR3', 'fr3start_indel')
    test('T7c indel at IMGT 66-75 -> A_FR1Nterm (location control)', 'A_FR1Nterm', 'fr3start_indel')
    U = U0
    T = pd.DataFrame(rows); T['q'] = multipletests(T.p.fillna(1), method='fdr_bh')[1]
    T.to_csv(f'{OUT}/tables/scan_tests.csv', index=False)
    print(T[T.test.str.contains('excl|T7')].round(4).to_string())
except FileNotFoundError:
    pass
