# Further adversarial checks on enzyme_target:
#  (a) near-duplicate copy leak (same antibody, 1-3 residue SEQRES difference, identical run) -> recount first-occurrence errors
#  (b) within-lab conditional logit (enzyme vs non-enzyme papers of the same lab)
#  (c) paper-level fragility index; permutation test within year
#  (d) "tool antibody" comparison: auxiliary antibodies against enzymes vs against transporters / GPCRs / receptors
#  (e) lab experience adjustment at paper level; EC-number audit of the enzyme class
import json, re, warnings, unicodedata, difflib
import numpy as np, pandas as pd, statsmodels.formula.api as smf, statsmodels.api as sm
from statsmodels.discrete.conditional_models import ConditionalLogit
from scipy import stats
warnings.filterwarnings('ignore')
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 60)
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
AC = '/home/user/SAbR/register_error_analysis/retest/deeper/antigen_competition/tables'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/antigen_competition_enzyme_target_integrity'
TESTS = []
def norm_lab(s):
    if not isinstance(s, str) or not s: return 'NA'
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode().lower()
    sur, _, ini = s.partition(','); return re.sub(r'[^a-z]', '', sur) + '_' + re.sub(r'[^a-z]', '', ini)[:1]

e = pd.read_csv(f'{SP}/deeper/entries.csv')
A = pd.read_csv(f'{AC}/pdbwide_ab_entries_antigen.csv.gz')
amy = set(pd.read_csv('/home/user/SAbR/register_error_analysis/retest/deeper/pdb_meta/tables/amyloid_fibril_chains.csv').pdb)
ch = pd.read_csv(f'{SP}/deeper/chains.csv.gz', usecols=['pdb', 'chain', 'method', 'deposit', 'unit', 'paper', 'err_first', 'err_any', 'vseq', 'aux_reagent', 'role', 'vhh', 'res', 'last_author'])

# ---------------- (a) near-duplicate copy leak
print('== (a) near-duplicate copy leak among first-occurrence runs')
ra = pd.read_csv(f'{SP}/deeper/runs_all.csv')
fo = pd.read_csv(f'{SP}/deeper/runs_first_occurrence.csv')
vs = ch.set_index(['pdb', 'chain']).vseq
ra['vseq'] = [vs.get((p, c)) for p, c in zip(ra.pdb, ra.chain)]; fo['vseq'] = [vs.get((p, c)) for p, c in zip(fo.pdb, fo.chain)]
ra['dep'] = pd.to_datetime(ra.deposit); fo['dep'] = pd.to_datetime(fo.deposit)
leak = []
for i, r in fo.iterrows():
    if not isinstance(r.vseq, str): continue
    cand = ra[(ra.dep < r.dep) & (ra.paper != r.paper) & (ra.imgt0 == r.imgt0) & (ra.imgt1 == r.imgt1) & (ra.delta == r.delta) & (ra.ctype == r.ctype)]
    for _, c in cand.iterrows():
        if not isinstance(c.vseq, str) or c.vseq == r.vseq: continue
        sm_ = difflib.SequenceMatcher(None, r.vseq, c.vseq, autojunk=False)
        nd = max(len(r.vseq), len(c.vseq)) - sum(b.size for b in sm_.get_matching_blocks())
        if nd <= 3:
            leak.append(dict(pdb=r.pdb, chain=r.chain, paper=r.paper, imgt0=r.imgt0, imgt1=r.imgt1, delta=r.delta, earlier_pdb=c.pdb, earlier_paper=c.paper, n_diff=nd,
                             same_lab=norm_lab(r.last_author) == norm_lab(c.last_author))); break
L = pd.DataFrame(leak).drop_duplicates(['pdb', 'chain'])
print(L.to_string() if len(L) else 'none')
L.to_csv(f'{OUT}/near_duplicate_copy_leaks.csv', index=False)
# entries whose ALL first-occurrence runs are near-duplicate copies
fo_ent = fo.groupby('pdb').size(); leak_ent = L.groupby('pdb').size() if len(L) else pd.Series(dtype=int)
fo['leak'] = [(p, c) in set(zip(L.pdb, L.chain)) for p, c in zip(fo.pdb, fo.chain)] if len(L) else False
ent_all_leak = set(fo.groupby('pdb').leak.all().loc[lambda s: s].index)
print('entries whose first-occurrence runs are all near-duplicate copies:', sorted(ent_all_leak))

# ---------------- build analysis frame
m = e.merge(A[['pdb', 'tgt_cls', 'tgt_keys', 'tgt_name']], on='pdb', how='left')
m = m[~m.pdb.isin(amy) & (m.err_first | ~m.err_any)].copy()
m['y'] = m.err_first.astype(int); m['dyear'] = pd.to_datetime(m.deposit).dt.year; m['ycat'] = m.dyear.clip(lower=2017).astype(str)
m['paper'] = m.doi.fillna(m.last_author.fillna('') + '_' + m.dyear.astype(str)); m['labn'] = m.last_author.map(norm_lab)
m['host_enz'] = (m.tgt_cls == 'enzyme').astype(int); m['vir_oth'] = (m.tgt_cls == 'viral other').astype(int); m['enz_comb'] = m.host_enz | m.vir_oth
lab = pd.read_csv(f'{AC}/lab_experience.csv'); m = m.merge(lab[['pdb', 'lab_prior_ab']], on='pdb', how='left')
EM = m[m.method == 'EM'].copy()
EMc = EM[~EM.pdb.isin(ent_all_leak)].copy()  # leak-corrected: near-duplicate copies are copied, so excluded like other copied-only entries
def two(d, t):
    return [int(((d[t] == 1) & (d.y == 1)).sum()), int(((d[t] == 1) & (d.y == 0)).sum()), int(((d[t] == 0) & (d.y == 1)).sum()), int(((d[t] == 0) & (d.y == 0)).sum())]
def fis(d, t, lab_, st):
    a, b, c, dd = two(d, t); o, p = stats.fisher_exact([[a, b], [c, dd]])
    r = dict(test=lab_, stratum=st, OR=o, p=p, cases_in=f'{a}/{a+b}', cases_out=f'{c}/{c+dd}'); TESTS.append(r)
    print(f'  {lab_:70s} in {a}/{a+b} out {c}/{c+dd} OR {o:.2f} p={p:.2g}'); return r
def paper_tab(D):
    P = D.groupby('paper').agg(y=('y', 'max'), host_enz=('host_enz', 'max'), enz_comb=('enz_comb', 'max'), vir_oth=('vir_oth', 'max'), n=('y', 'size'),
                               ycat=('ycat', lambda s: s.mode().iat[0]), dyear=('dyear', 'min'), res=('res', 'median'), labn=('labn', 'first'), exp=('lab_prior_ab', 'min')).reset_index()
    P['ln'] = np.log2(P.n); P['lexp'] = np.log2(1 + P.exp.fillna(0)); return P
P = paper_tab(EM); Pc = paper_tab(EMc)
print('\n== leak-corrected counts (EM)')
for t in ['host_enz', 'vir_oth', 'enz_comb']:
    fis(EMc, t, f'entry, leak-corrected: {t}', 'EM'); fis(Pc, t, f'paper, leak-corrected: {t}', 'EM papers')

# ---------------- (b) within-lab conditional logit at paper level
print('\n== (b) within-lab comparison (paper level, labs with both enzyme and non-enzyme EM antibody papers)')
for t in ['host_enz', 'enz_comb']:
    g = P.groupby('labn')[t].agg(['min', 'max']); labs_mixed = g[(g['min'] == 0) & (g['max'] == 1)].index
    D = P[P.labn.isin(labs_mixed)].copy()
    inf = D.groupby('labn').y.agg(['min', 'max']); informative = inf[inf['min'] != inf['max']].index
    print(f'  {t}: labs with both kinds of paper {len(labs_mixed)}, papers {len(D)}, informative labs (both error and clean papers) {len(informative)}')
    print(D.groupby(t).y.agg(['sum', 'size']).to_string())
    try:
        cl = ConditionalLogit(D.y.values, D[[t]].values.astype(float), groups=pd.factorize(D.labn)[0]).fit(disp=0)
        o = np.exp(cl.params[0]); p = cl.pvalues[0]; ci = np.exp(cl.conf_int()[0])
        print(f'  conditional logit (lab strata) OR {o:.2f} ({ci[0]:.2f}-{ci[1]:.2f}) p={p:.2g}')
        TESTS.append(dict(test=f'paper: {t}, conditional on lab', stratum='EM papers, mixed labs', OR=o, p=p, cases_in=f"{int(D[D[t]==1].y.sum())}/{int((D[t]==1).sum())}", cases_out=f"{int(D[D[t]==0].y.sum())}/{int((D[t]==0).sum())}"))
    except Exception as ex: print('  conditional logit failed', ex)
    print(D[D.labn.isin(informative)].sort_values(['labn', t])[['labn', 'paper', t, 'y', 'n', 'dyear']].to_string())

# ---------------- (c) fragility index at paper level and within-year permutation
print('\n== (c) paper-level fragility and within-year permutation')
for t in ['host_enz', 'enz_comb']:
    a, b, c, dd = two(P, t); k = 0
    while stats.fisher_exact([[a - k, b + k], [c, dd]])[1] < 0.05 and k < a: k += 1
    print(f'  {t}: paper-level Fisher fragility index = {k} (flipping {k} of {a} enzyme error papers to clean gives p>=0.05)')
    rng = np.random.default_rng(7); obs = P[P[t] == 1].y.sum(); sims = []
    for _ in range(20000):
        perm = P.groupby('ycat')[t].transform(lambda s: rng.permutation(s.values))
        sims.append(P.y[perm == 1].sum())
    sims = np.array(sims); pp = (1 + (sims >= obs).sum()) / (1 + len(sims))
    print(f'  {t}: within-year permutation of class labels among papers: obs {obs} error papers, null mean {sims.mean():.1f}, p={pp:.2g}')
    TESTS.append(dict(test=f'paper: {t}, within-year permutation', stratum='EM papers', OR=np.nan, p=pp, cases_in=f'{obs}/{int(P[t].sum())}'))

# ---------------- (d) tool-antibody comparison (unit level, auxiliary antibodies by contacted antigen class)
print('\n== (d) auxiliary (tool) antibodies by contacted antigen class, EM units')
U = pd.read_csv(f'{OUT}/units_em_enzyme.csv.gz')
aux = ch[ch.method == 'EM'].groupby('unit').agg(aux=('aux_reagent', 'max'), role=('role', lambda s: s.mode().iat[0])).reset_index()
U = U.merge(aux, on='unit', how='left')
print(U[U.aux == True].groupby('agcls').y.agg(['sum', 'size', 'mean']).sort_values('size', ascending=False).round(3).to_string())
Ua = U[(U.aux == True) & U.agcls.isin(['enzyme', 'viral other', 'channel/transporter', 'GPCR', 'receptor/membrane protein', 'other protein'])].copy()
Ua['enz'] = Ua.agcls.isin(['enzyme', 'viral other']).astype(int)
fis(Ua, 'enz', 'aux units: enzyme(+viral other) vs transporter/GPCR/receptor/other targets', 'EM aux units')
Ua['host'] = (Ua.agcls == 'enzyme').astype(int)
fis(Ua[Ua.agcls != 'viral other'], 'host', 'aux units: host enzyme vs transporter/GPCR/receptor/other targets', 'EM aux units')

# ---------------- (e) lab experience + year at paper level; nanobody at paper level
print('\n== (e) paper-level models with lab experience')
for t in ['host_enz', 'enz_comb']:
    for f in [f'y ~ {t} + C(ycat) + res + ln', f'y ~ {t} + lexp + C(ycat) + res + ln']:
        mod = smf.glm(f, P, family=sm.families.Binomial()).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(P.labn)[0]})
        ci = np.exp(mod.conf_int().loc[t]); print(f'  {f:50s} OR {np.exp(mod.params[t]):.2f} ({ci.iloc[0]:.2f}-{ci.iloc[1]:.2f}) p={mod.pvalues[t]:.2g}')
        TESTS.append(dict(test=f'paper: {f}', stratum='EM papers, lab-clustered', OR=np.exp(mod.params[t]), p=mod.pvalues[t]))
# ---------------- EC audit of enzyme class
uni = json.load(open(f'{SP}/deeper/cache/antigen_competition/uniprot_info.json'))
EM['ec'] = EM.tgt_keys.fillna('').map(lambda s: ';'.join(sum([(uni.get(k) or {}).get('ec') or [] for k in s.split('|')], [])))
print('\n== host-enzyme EM entries: error status, EC numbers')
x = EM[EM.host_enz == 1].groupby(['tgt_name']).agg(n=('y', 'size'), err=('y', 'sum'), labs=('labn', lambda s: ','.join(sorted(set(s)))), ec=('ec', 'first')).sort_values(['err', 'n'], ascending=False)
print(x.to_string())
print('\n== EC-bearing targets in other classes, EM (name, n, errors)')
y = EM[(EM.ec != '') & ~EM.tgt_cls.isin(['enzyme', 'viral other', 'viral capsid', 'viral surface glycoprotein'])].groupby(['tgt_cls', 'tgt_name']).y.agg(['size', 'sum']).sort_values('size', ascending=False)
print(y.head(40).to_string())
pd.DataFrame(TESTS).to_csv(f'{OUT}/verify_tests_v02.csv', index=False)
