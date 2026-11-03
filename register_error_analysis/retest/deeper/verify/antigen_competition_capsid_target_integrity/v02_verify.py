# Adversarial integrity/statistics check of the antigen_competition "capsid_target" finding.
# Independently rebuilds case/control sets, reproduces the headline numbers, then tests
# (1) lab -> consortium clustering, (2) numbering-artifact-prone error types (D0 / within-CDR1 gap placement /
# bovine antibodies numbered with non-bovine HMMs / very large shifts), (3) leave-one-out fragility,
# (4) class misassignment, (5) species confounding, (6) period, (7) the within-capsid lq_z claim.
import re, json, warnings, unicodedata, collections
import numpy as np, pandas as pd, statsmodels.formula.api as smf, statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings('ignore')
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 60)
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
INV = '/home/user/SAbR/register_error_analysis/retest/deeper/antigen_competition/tables'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/antigen_competition_capsid_target_integrity'
TESTS = []

# ---------------------------------------------------------------- 0. rebuild case/control independently
ent = pd.read_csv(f'{SP}/deeper/entries.csv')
amy = set(pd.read_csv('/home/user/SAbR/register_error_analysis/retest/deeper/pdb_meta/tables/amyloid_fibril_chains.csv').pdb)
foc = pd.read_csv(f'{INV}/focal_entries_antigen_features.csv.gz', usecols=['pdb', 'tgt_cls', 'tgt_name', 'tgt_desc', 'tgt_org', 'lab'])
e = ent.copy()
e['amyloid'] = e.pdb.isin(amy)
e = e[~e.amyloid & (e.err_first | ~e.err_any)].copy()
e = e.merge(foc, on='pdb', how='left')
print('rebuilt analysis set', len(e), 'errors', int(e.err_first.sum()), '| missing class rows', int(e.tgt_cls.isna().sum()))
e['y'] = e.err_first.astype(int)
e['dyear'] = pd.to_datetime(e.deposit).dt.year
e['ycat'] = e.dyear.clip(lower=2017).astype(str)
e['paper'] = e.doi.fillna(e.last_author.fillna('') + '_' + e.dyear.astype(str))
e['period'] = pd.cut(e.dyear, [0, 2019, 2021, 2100], labels=['<=2019', '2020-21', '2022+']).astype(str)
e['cap'] = (e.tgt_cls == 'viral capsid').astype(int)
EM = e[e.method == 'EM'].copy()
print('EM', len(EM), 'errors', int(EM.y.sum()), '| capsid', int(EM.cap.sum()), 'capsid errors', int(EM[EM.cap == 1].y.sum()))

def rate(d, col='y'): return f'{100*d[col].mean():.1f}% ({int(d[col].sum())}/{len(d)})'

def glm(d, f, term, groups):
    mod = smf.glm(f, d, family=sm.families.Binomial()).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(d[groups])[0]})
    ci = mod.conf_int().loc[term]
    return np.exp(mod.params[term]), np.exp(ci[0]), np.exp(ci[1]), mod.pvalues[term]

def add(test, d, OR, lo, hi, p, note='', ycol='y', grp='cap'):
    a = d[d[grp] == 1]; b = d[d[grp] == 0]
    TESTS.append(dict(test=test, capsid=rate(a, ycol), other=rate(b, ycol), OR=OR, lo=lo, hi=hi, p=p, n=len(d), n_err=int(d[ycol].sum()), note=note))
    print(f'  {test:75s} capsid {rate(a, ycol):>16s} other {rate(b, ycol):>18s} OR {OR:5.2f} ({lo:.2f}-{hi:.2f}) p={p:.2g}')

print('\n== 1. Reproduce headline (entry level, EM, y ~ cap + C(ycat) + res, SE clustered by paper)')
add('R1 headline reproduction', EM, *glm(EM, 'y ~ cap + C(ycat) + res', 'cap', 'paper'))

# ---------------------------------------------------------------- 2. labs -> consortia (citation co-authorship)
aux = pd.read_csv(f'{OUT}/entry_aux.csv.gz')
EM = EM.merge(aux, on='pdb', how='left')
def nn(n):
    n = unicodedata.normalize('NFKD', n).encode('ascii', 'ignore').decode().lower().strip()
    sur, _, ini = n.partition(',')
    return re.sub(r'[^a-z]', '', sur) + '_' + re.sub(r'[^a-z]', '', ini)[:1]
def authors(r):
    s = set()
    for col in ['audit_authors', 'cit_authors']:
        if isinstance(r[col], str): s |= {nn(a) for a in r[col].split('; ') if a.strip()}
    return s
EM['last_n'] = EM.last_author.fillna('').map(lambda x: nn(x) if x else '')
# corresponding-author set of a paper: last audit author + last citation author (both normalised)
EM['cit_last'] = EM.cit_authors.fillna('').map(lambda s: nn(s.split('; ')[-1]) if s else '')
pap = EM.groupby('paper').agg(cap=('cap', 'max'), lasts=('last_n', lambda s: set(s) - {''}), clast=('cit_last', lambda s: set(s) - {''})).reset_index()
pap_auth = EM.groupby('paper').apply(lambda g: set().union(*[authors(r) for _, r in g.iterrows()])).to_dict()
# union-find: two papers share a consortium if a corresponding author of one is an author of the other.
# Applied among capsid papers (the claim is about capsid labs); every other paper keeps its own last-author lab.
par = {}
def f(x):
    par.setdefault(x, x)
    while par[x] != x: par[x] = par[par[x]]; x = par[x]
    return x
def u(a, b): par[f(a)] = f(b)
cp = pap[pap.cap == 1]
for _, a in cp.iterrows():
    f(a.paper)
    for _, b in cp.iterrows():
        if a.paper < b.paper and ((a.lasts | a.clast) & pap_auth[b.paper] or (b.lasts | b.clast) & pap_auth[a.paper]):
            u(a.paper, b.paper)
cons_of_paper = {p: ('CONS:' + f(p)) for p in cp.paper}
EM['consortium'] = [cons_of_paper.get(p, 'LAB:' + (l or p)) for p, l in zip(EM.paper, EM.last_n)]
# a non-capsid paper from a capsid consortium's corresponding author joins that consortium too (same lab)
lab2cons = {}
for p, c in cons_of_paper.items():
    for l in pap.set_index('paper').loc[p, 'lasts']: lab2cons.setdefault(l, c)
EM['consortium'] = [c if c.startswith('CONS:') else lab2cons.get(l, c) for c, l in zip(EM.consortium, EM.last_n)]
ce = EM[(EM.cap == 1) & (EM.y == 1)]
tab = ce.groupby('consortium').agg(err_entries=('pdb', 'size'), papers=('paper', 'nunique'), last_authors=('last_author', lambda s: ', '.join(sorted(set(s)))),
                                   pdbs=('pdb', lambda s: ' '.join(sorted(s)))).sort_values('err_entries', ascending=False)
print('\n== 2. Capsid error entries by consortium (citation co-authorship links)')
print(tab.to_string())
print('distinct last-author strings among capsid error entries:', ce.last_author.nunique(), '| consortia:', ce.consortium.nunique())
tab.to_csv(f'{OUT}/capsid_error_consortia.csv')
add('R2 headline, SE clustered by consortium', EM, *glm(EM, 'y ~ cap + C(ycat) + res', 'cap', 'consortium'))

# leave-one-consortium-out and drop top-k
print('\n  leave-one-consortium-out (entry level):')
lo_rows = []
for c in tab.index:
    d = EM[EM.consortium != c]
    r = glm(d, 'y ~ cap + C(ycat) + res', 'cap', 'paper'); lo_rows.append((c, tab.loc[c, 'last_authors'], *r))
    print(f'    drop {tab.loc[c,"last_authors"][:40]:40s} OR {r[0]:.2f} ({r[1]:.2f}-{r[2]:.2f}) p={r[3]:.2g}')
pd.DataFrame(lo_rows, columns=['consortium', 'labs', 'OR', 'lo', 'hi', 'p']).to_csv(f'{OUT}/leave_one_consortium_out.csv', index=False)
for k in [2, 3]:
    d = EM[~EM.consortium.isin(tab.index[:k])]
    add(f'R3 drop top-{k} capsid error consortia (entry level)', d, *glm(d, 'y ~ cap + C(ycat) + res', 'cap', 'paper'))

# ---------------------------------------------------------------- 3. questionable error types
runs = pd.read_csv(f'{SP}/deeper/runs_first_occurrence.csv')
runs = runs[runs.pdb.isin(EM.pdb)].copy()
bov = set(EM[EM.ab_orgs.fillna('').str.contains('Bos taurus') | EM.title.fillna('').str.contains(r'bovine|cattle|\bcow\b', case=False, regex=True)].pdb)
runs['q_D0'] = runs['mode'] == 'D0'                                   # no net shift, inside CDR3 / at a truncated chain end
runs['q_F_nan'] = (runs['mode'] == 'F_CDR1') & runs.delta.isna()      # gap placement inside CDR1, frameworks agree
runs['q_bovine'] = runs.pdb.isin(bov)                                 # bovine V domains numbered with non-bovine HMMs
runs['q_bigshift'] = runs.delta.abs() >= 5                            # >=5-position shifts: tool failure or ultralong-CDR3 convention
runs['q_any'] = runs.q_D0 | runs.q_F_nan | runs.q_bovine
runs['q_any2'] = runs.q_any | runs.q_bigshift
ent_q = runs.groupby('pdb').agg(all_q=('q_any', 'all'), all_q2=('q_any2', 'all'), n_runs=('q_any', 'size'))
EM = EM.merge(ent_q, on='pdb', how='left')
EM['all_q'] = EM.all_q.fillna(False).astype(bool); EM['all_q2'] = EM.all_q2.fillna(False).astype(bool)
print('\n== 3. Error entries whose only first-occurrence runs are numbering-artifact-prone types')
x = EM[EM.y == 1]
print(pd.crosstab(x.cap, x.all_q, margins=True).to_string())
print(pd.crosstab(x.cap, x.all_q2, margins=True).to_string())
print('  bovine EM entries:', len(bov & set(EM.pdb)), '| errors among them:', int(EM[EM.pdb.isin(bov)].y.sum()), '| capsid among them:', int(EM[EM.pdb.isin(bov)].cap.sum()))
print('  capsid error entries flagged questionable:', ' '.join(sorted(x[(x.cap == 1) & x.all_q].pdb)))
print('  non-capsid error entries flagged questionable:', ' '.join(sorted(x[(x.cap == 0) & x.all_q].pdb)))
# check the D0/within-CDR1 runs are concentrated in capsids (the claim's "what differs" signature)
r2 = runs.merge(EM[['pdb', 'cap']], on='pdb')
print('  runs by flag and capsid:'); print(r2.groupby('cap')[['q_D0', 'q_F_nan', 'q_bovine', 'q_bigshift']].sum().to_string())
# robust definitions: drop entries whose error rests only on questionable runs (neither case nor control)
D1 = EM[~((EM.y == 1) & EM.all_q)].copy()
add('R4 robust errors (drop D0-only, within-CDR1-gap-only, bovine-only error entries)', D1, *glm(D1, 'y ~ cap + C(ycat) + res', 'cap', 'paper'))
D2 = D1[~D1.pdb.isin(bov)].copy()
add('R5 R4 + drop all bovine-antibody entries (cases and controls)', D2, *glm(D2, 'y ~ cap + C(ycat) + res', 'cap', 'paper'))
D3 = EM[~((EM.y == 1) & EM.all_q2) & ~EM.pdb.isin(bov)].copy()
add('R6 R5 + also drop error entries resting only on >=5-position shifts', D3, *glm(D3, 'y ~ cap + C(ycat) + res', 'cap', 'paper'))
add('R7 R5, SE clustered by consortium', D2, *glm(D2, 'y ~ cap + C(ycat) + res', 'cap', 'consortium'))
for k in [1, 2]:
    tabr = D2[(D2.cap == 1) & (D2.y == 1)].groupby('consortium').size().sort_values(ascending=False)
    d = D2[~D2.consortium.isin(tabr.index[:k])]
    add(f'R8 R5 + drop top-{k} remaining capsid error consortia', d, *glm(d, 'y ~ cap + C(ycat) + res', 'cap', 'paper'))
print('  remaining capsid error entries by consortium (R5):'); print(D2[(D2.cap == 1) & (D2.y == 1)].groupby('consortium').agg(n=('pdb', 'size'), papers=('paper', 'nunique'), labs=('last_author', lambda s: ', '.join(sorted(set(s))))).sort_values('n', ascending=False).to_string())

# ---------------------------------------------------------------- 4. paper- and consortium-level units
def paper_level(D, label, unit='paper'):
    P = D.groupby(unit).agg(y=('y', 'max'), cap=('cap', 'max'), capmode=('cap', 'mean'), ycat=('ycat', lambda s: s.mode().iat[0]), res=('res', 'median'), n=('y', 'size')).reset_index()
    P['cap'] = (P.capmode >= 0.5).astype(int); P['ln'] = np.log2(P.n)
    OR, lo, hi, p = glm(P, 'y ~ cap + C(ycat) + res + ln', 'cap', unit)
    add(f'{label}: logit y~cap+year+res+log2(n)', P, OR, lo, hi, p)
    a = int(P[(P.cap == 1)].y.sum()); b = int((P.cap == 1).sum()) - a; c = int(P[P.cap == 0].y.sum()); d_ = int((P.cap == 0).sum()) - c
    orf, pf = stats.fisher_exact([[a, b], [c, d_]])
    add(f'{label}: Fisher exact (unadjusted)', P, orf, np.nan, np.nan, pf)
    # Poisson-binomial check: expected capsid error units from a model fitted on non-capsid units only
    mod = smf.glm('y ~ C(ycat) + res + ln', P[P.cap == 0], family=sm.families.Binomial()).fit()
    ph = mod.predict(P[P.cap == 1]).values; rng = np.random.default_rng(7)
    sims = rng.binomial(1, np.tile(ph, (20000, 1))).sum(1)
    pmc = (1 + (sims >= a).sum()) / (1 + len(sims))
    print(f'     {label}: observed capsid error units {a}/{a+b}, expected {ph.sum():.1f} (non-capsid model), one-sided simulation p={pmc:.2g}')
    TESTS.append(dict(test=f'{label}: observed vs expected (Poisson-binomial)', capsid=f'{a}/{a+b}', other=f'expected {ph.sum():.1f}', OR=a/ph.sum(), lo=np.nan, hi=np.nan, p=pmc, n=len(P), n_err=int(P.y.sum()), note='ratio obs/exp'))
    return P
print('\n== 4. Paper- and consortium-level')
paper_level(EM, 'R9 paper level, original errors')
paper_level(D2, 'R10 paper level, robust errors (R5)')
paper_level(EM, 'R11 consortium level, original errors', unit='consortium')
paper_level(D2, 'R12 consortium level, robust errors (R5)', unit='consortium')

# leave-one-paper-out on the robust set (entry level)
print('\n  leave-one-capsid-error-paper-out, robust set (R5):')
lp = []
for p in D2[(D2.cap == 1) & (D2.y == 1)].paper.unique():
    d = D2[D2.paper != p]; r = glm(d, 'y ~ cap + C(ycat) + res', 'cap', 'paper'); lp.append((p, *r))
lp = pd.DataFrame(lp, columns=['paper', 'OR', 'lo', 'hi', 'p']).sort_values('p', ascending=False)
print(lp.head(5).round(4).to_string())
lp.to_csv(f'{OUT}/leave_one_paper_out_robust.csv', index=False)

# ---------------------------------------------------------------- 5. class misassignment (capsids classified as surface glycoprotein)
txt = (EM.tgt_name.fillna('') + ' ' + EM.tgt_desc.fillna('') + ' ' + EM.title.fillna(''))
mis = (EM.cap == 0) & (EM.tgt_org == 'viral') & txt.str.contains(r'\bVP[1-4]\b|capsid|norovirus|\bVLP\b.*VP1', case=False, regex=True) & ~txt.str.contains(r'nucleocapsid|spike|envelope|glycoprotein E[12]|\bE2\b', case=False, regex=True)
print('\n== 5. Capsid entries classified elsewhere:', int(mis.sum()), 'entries,', int(EM[mis].y.sum()), 'errors')
print(EM[mis][['pdb', 'y', 'tgt_cls', 'tgt_name', 'title', 'last_author']].to_string())
EM['cap_fix'] = (EM.cap.astype(bool) | mis).astype(int)
d = EM.copy(); d['cap'] = d.cap_fix
add('R13 capsid class corrected (+ norovirus VP1 / rotavirus VP4 etc.), original errors', d, *glm(d, 'y ~ cap + C(ycat) + res', 'cap', 'paper'))
d2 = d[~((d.y == 1) & d.all_q) & ~d.pdb.isin(bov)]
add('R14 corrected class, robust errors (R5 filters)', d2, *glm(d2, 'y ~ cap + C(ycat) + res', 'cap', 'paper'))

# ---------------------------------------------------------------- 6. species: chain-level species of the antibody (ANARCI) vs source organism
ch = pd.read_csv(f'{SP}/deeper/chains.csv.gz', usecols=['pdb', 'chain', 'species', 'unit', 'err_first', 'err_any', 'v_id', 'vhh'])
sp = EM[['pdb', 'ab_orgs']].copy()
sp['src_mouse'] = sp.ab_orgs.fillna('').str.contains('Mus musculus|Mus sp', regex=True).astype(int)
sp['src_human'] = sp.ab_orgs.fillna('').str.contains('Homo sapiens').astype(int)
EM = EM.merge(sp[['pdb', 'src_mouse', 'src_human']], on='pdb', how='left')
mo = ch.groupby('pdb').species.apply(lambda s: (s == 'mouse').mean()).rename('anarci_mouse')
EM = EM.merge(mo, on='pdb', how='left')
print('\n== 6. Species. Capsid vs other: ANARCI-mouse fraction %.2f vs %.2f; source-organism mouse %.2f vs %.2f' % (
    EM[EM.cap == 1].anarci_mouse.mean(), EM[EM.cap == 0].anarci_mouse.mean(), EM[EM.cap == 1].src_mouse.mean(), EM[EM.cap == 0].src_mouse.mean()))
D2 = D2.merge(EM[['pdb', 'src_mouse', 'anarci_mouse']], on='pdb', how='left')
for lab_, dd in [('source organism mouse', D2[D2.src_mouse == 1]), ('source organism not mouse', D2[D2.src_mouse == 0])]:
    try:
        add(f'R15 robust errors, stratum: {lab_}', dd, *glm(dd, 'y ~ cap + C(ycat) + res', 'cap', 'paper'))
    except Exception as ex:
        print('  failed', lab_, ex)
add('R16 robust errors, adjusted for source-organism mouse', D2, *glm(D2, 'y ~ cap + src_mouse + C(ycat) + res', 'cap', 'paper'))

# ---------------------------------------------------------------- 7. period
print('\n== 7. Period')
for per in ['<=2019', '2020-21', '2022+']:
    dd = D2[D2.period == per]
    try:
        add(f'R17 robust errors, period {per}', dd, *glm(dd, 'y ~ cap + C(ycat) + res' if per != '<=2019' else 'y ~ cap + res', 'cap', 'paper'))
    except Exception as ex:
        print('  failed', per, ex)

# ---------------------------------------------------------------- 8. within-capsid lq_z claim
lq = pd.read_csv(f'{SP}/deeper/chains.csv.gz', usecols=['pdb', 'lq_z']).groupby('pdb').lq_z.mean()
EM['lq_z'] = EM.pdb.map(lq)
print('\n== 8. lq_z, error vs clean entries, within capsid and within non-capsid (EM)')
for nm, dd in [('capsid', EM[EM.cap == 1]), ('non-capsid', EM[EM.cap == 0]), ('capsid robust', D2.assign(lq_z=D2.pdb.map(lq))[lambda t: t.cap == 1])]:
    a = dd[dd.y == 1].lq_z.dropna(); b = dd[dd.y == 0].lq_z.dropna(); p = stats.mannwhitneyu(a, b).pvalue
    print(f'  {nm:14s} errors {a.mean():+.2f} (n={len(a)}) vs clean {b.mean():+.2f} (n={len(b)}) MWU p={p:.2g}')
    TESTS.append(dict(test=f'R18 lq_z error vs clean, {nm}', capsid=f'{a.mean():+.2f}', other=f'{b.mean():+.2f}', OR=np.nan, lo=np.nan, hi=np.nan, p=p, n=len(a) + len(b), n_err=len(a), note='mean lq_z err vs clean'))
# interaction: is the error-vs-clean lq_z gap larger in capsids?
dd = EM.dropna(subset=['lq_z']).copy()
mod = smf.ols('lq_z ~ y * cap + C(ycat) + res', dd).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(dd.paper)[0]})
print(f'  interaction y:cap on lq_z: {mod.params["y:cap"]:+.2f} p={mod.pvalues["y:cap"]:.2g}')
TESTS.append(dict(test='R19 lq_z error-vs-clean gap larger in capsids (interaction)', capsid='', other='', OR=mod.params['y:cap'], lo=np.nan, hi=np.nan, p=mod.pvalues['y:cap'], n=len(dd), n_err=int(dd.y.sum()), note='OLS coefficient, not OR'))

# ---------------------------------------------------------------- 9. X-ray
XR = e[e.method == 'X-ray']
print('\n== 9. X-ray capsid:', rate(XR[XR.cap == 1]), 'vs', rate(XR[XR.cap == 0]))

# ---------------------------------------------------------------- BH
T = pd.DataFrame(TESTS)
T['q_BH'] = np.nan
k = T.p.notna()
T.loc[k, 'q_BH'] = multipletests(T.loc[k, 'p'], method='fdr_bh')[1]
T.to_csv(f'{OUT}/verify_tests.csv', index=False)
print('\n== All tests (BH over this verification)')
print(T[['test', 'capsid', 'other', 'OR', 'lo', 'hi', 'p', 'q_BH', 'n', 'n_err']].round(4).to_string())
EM.drop(columns=['audit_authors', 'cit_authors']).to_csv(f'{OUT}/em_entries_verify.csv.gz', index=False)
