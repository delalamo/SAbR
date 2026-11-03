"""Reproduce T1/T1-excl (atypical FR3 length -> C_FR3) from the shared data, with the investigator's feature definition
(seq_novelty ANARCI cache), and list the positive error units with gap positions and run windows.
Outputs tables/v01_units.pkl (unit table), tables/v01_positives.csv"""
import pickle, numpy as np, pandas as pd, statsmodels.formula.api as smf, warnings
from scipy.stats import fisher_exact
warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
DE = '/home/user/SAbR/register_error_analysis/retest/deeper'
OUT = f'{DE}/verify/case_files_ABCF_FR3_INDEL_C_integrity'
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
fr3, gaps, src = [], [], []
for r in C.itertuples():
    light = r.ctype in 'KL'
    nm = number(r.seq, light); s = 'seq'
    if not nm: nm = number(r.vseq, light); s = 'vseq'
    if not nm: fr3.append(np.nan); gaps.append(None); src.append(None); continue
    ps = {p for p, i, a in nm if i == ''}
    fr3.append(sum(66 <= p <= 104 for p, i, a in nm))
    gaps.append(tuple(k for k in range(66, 105) if k not in ps) + tuple(f'{p}{i}' for p, i, a in nm if i and 66 <= p <= 104))
    src.append(s)
C['FR3'] = fr3; C['fr3_gaps'] = gaps; C['num_src'] = src
C['cclass'] = np.where(C.ctype == 'H', np.where(C.vhh == 1, 'VHH', 'H'), C.ctype)
norm = {'H': {38}, 'VHH': {38}, 'K': {36}, 'L': {36, 38}}
C['fr3_atyp'] = [int(f == f and int(f) not in norm[c]) for f, c in zip(C.FR3, C.cclass)]
fm = RF.groupby(['pdb', 'chain'])['mode'].agg(set).to_dict()
C['modes'] = [fm.get((p, c), set()) for p, c in zip(C.pdb, C.chain)]
anym = RA.groupby(['pdb', 'chain'])['mode'].agg(set).to_dict()
C['modes_all'] = [anym.get((p, c), set()) for p, c in zip(C.pdb, C.chain)]
C.to_pickle(f'{SP}/deeper/cache/verify_fr3_integrity/chains_feat.pkl')
def units(D):
    D = D.copy()
    D['ybin'] = pd.cut(D.year, [0, 2019, 2021, 2023, 2100], labels=['<=2019', '2020-21', '2022-23', '>=2024']).astype(str)
    return D.groupby('unit').agg(paper=('paper', 'first'), year=('year', 'first'), ybin=('ybin', 'first'), res=('res', 'mean'),
        cclass=('cclass', 'first'), err_any=('err_any', 'max'), err_first=('err_first', 'max'), modes=('modes', lambda s: set().union(*s)),
        modes_all=('modes_all', lambda s: set().union(*s)), fr3_atyp=('fr3_atyp', 'max'), FR3=('FR3', 'first'), fr3_gaps=('fr3_gaps', 'first'),
        pdb=('pdb', 'first'), chain=('chain', 'first'), last_author=('last_author', 'first'), v_gene=('v_gene', 'first'),
        species=('species', 'first'), vseq=('vseq', 'first'), seq=('seq', 'first'), ctype=('ctype', 'first'), method=('method', 'first')).reset_index()
U = units(C[C.method == 'EM']); U.to_pickle(f'{SP}/deeper/cache/verify_fr3_integrity/units_em.pkl')
UX = units(C[C.method == 'X-ray']); UX.to_pickle(f'{SP}/deeper/cache/verify_fr3_integrity/units_xray.pkl')
U['yC'] = U.modes.map(lambda s: 'C_FR3' in s).astype(int)
def fit(d, x='fr3_atyp', y='yC', label=''):
    d = d[d.cclass.isin(set(d[d[y] == 1].cclass))]
    a = int(d[(d[y] == 1)][x].sum()); ne = int(d[y].sum()); c = int(d[d[y] == 0][x].sum()); nc = int((d[y] == 0).sum())
    orf, pf = fisher_exact([[a, ne - a], [c, nc - c]])
    try:
        m = smf.logit(f'{y} ~ {x} + ybin + res + cclass', data=d).fit(disp=0, cov_type='cluster', cov_kwds={'groups': pd.factorize(d.paper)[0]})
        o, lo, hi, p = np.exp(m.params[x]), *np.exp(m.conf_int().loc[x]), m.pvalues[x]
    except Exception as e:
        o = lo = hi = p = np.nan
    return dict(test=label, err=f'{a}/{ne}', err_pct=a / ne * 100, ctl=f'{c}/{nc}', ctl_pct=c / nc * 100, OR_fisher=orf, p_fisher=pf, OR_adj=o, lo=lo, hi=hi, p_adj=p)
rows = []
d = U[(U.yC == 1) | (U.err_any == False)]
rows.append(fit(d, label='T1 reproduce: C_FR3 vs clean (err_any False)'))
AR = pd.read_csv(f'{DE}/case_files_ABCF/tables/artifact_runs.csv')
Cu = C.set_index(['pdb', 'chain']).unit.to_dict()
art = {Cu.get((p, c)) for p, c in zip(AR.pdb, AR.chain)}
d2 = d[~d.unit.isin(art)]
rows.append(fit(d2, label='T1-excl reproduce: artifact units removed'))
T = pd.DataFrame(rows); pd.set_option('display.width', 250); print(T.round(5).to_string())
T.to_csv(f'{OUT}/tables/v01_reproduce.csv', index=False)
# positives
P = d[(d.yC == 1)].copy()
RFc = RF[RF['mode'] == 'C_FR3'].copy(); RFc['unit'] = [Cu.get((p, c)) for p, c in zip(RFc.pdb, RFc.chain)]
win = RFc.groupby('unit').apply(lambda x: ';'.join(f'{a}-{b}({dd:+.0f})' if dd == dd else f'{a}-{b}(nan)' for a, b, dd in zip(x.imgt0, x.imgt1, x.delta))).to_dict()
P['C_runs'] = P.unit.map(win); P['artifact_excl'] = P.unit.isin(art)
P = P.sort_values(['fr3_atyp', 'year'], ascending=[False, True])
cols = ['pdb', 'chain', 'cclass', 'year', 'res', 'last_author', 'v_gene', 'species', 'FR3', 'fr3_gaps', 'fr3_atyp', 'C_runs', 'artifact_excl', 'modes']
print(P[cols].to_string())
P[cols + ['unit', 'paper']].to_csv(f'{OUT}/tables/v01_C_error_units.csv', index=False)
