# Adversarial verification of the "enzyme_target" finding (antigen_competition lens).
# Independently rebuild case/control sets from the shared tables, recompute entry/paper/unit-level effects,
# check the error chains actually contact the enzyme, and run leave-one-paper/lab-out fragility checks.
import json, re, warnings, collections, unicodedata
import numpy as np, pandas as pd, statsmodels.formula.api as smf, statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.contingency_tables import StratifiedTable
warnings.filterwarnings('ignore')
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 70)
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
AC = '/home/user/SAbR/register_error_analysis/retest/deeper/antigen_competition/tables'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/antigen_competition_enzyme_target_integrity'
TESTS = []

def norm_lab(s):
    if not isinstance(s, str) or not s: return 'NA'
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode().lower()
    sur, _, ini = s.partition(',')
    return re.sub(r'[^a-z]', '', sur) + '_' + re.sub(r'[^a-z]', '', ini)[:1]

# ---------------------------------------------------------------- 1. rebuild case/control from raw shared tables
e = pd.read_csv(f'{SP}/deeper/entries.csv')
A = pd.read_csv(f'{AC}/pdbwide_ab_entries_antigen.csv.gz')
assert A.pdb.is_unique and e.pdb.is_unique
amy = set(pd.read_csv('/home/user/SAbR/register_error_analysis/retest/deeper/pdb_meta/tables/amyloid_fibril_chains.csv').pdb)
ch = pd.read_csv(f'{SP}/deeper/chains.csv.gz', usecols=['pdb', 'chain', 'ctype', 'method', 'res', 'deposit', 'unit', 'paper', 'err_first', 'err_any', 'err_modes',
                                                       'vhh', 'role', 'aux_reagent', 'v_id', 'species', 'chain_desc', 'fit', 'irm', 'lq_z', 'last_author'])
# cross-check entry flags against chain flags
cf = ch.groupby('pdb').agg(cf_first=('err_first', 'max'), cf_any=('err_any', 'max'))
chk = e.merge(cf, on='pdb', how='left')
print('entry err_first vs chain-derived mismatches:', int((chk.err_first != chk.cf_first).sum()), '| err_any mismatches:', int((chk.err_any != chk.cf_any).sum()))
m = e.merge(A[['pdb', 'tgt_cls', 'tgt_name', 'tgt_desc', 'tgt_keys', 'ab_status', 'lab']], on='pdb', how='left')
m = m[~m.pdb.isin(amy) & (m.err_first | ~m.err_any)].copy()
m['y'] = m.err_first.astype(int)
m['dyear'] = pd.to_datetime(m.deposit).dt.year
m['ycat'] = m.dyear.clip(lower=2017).astype(str)
m['period'] = pd.cut(m.dyear, [0, 2019, 2021, 2100], labels=['<=2019', '2020-21', '2022+']).astype(str)
m['paper'] = m.doi.fillna(m.last_author.fillna('') + '_' + m.dyear.astype(str))  # investigator's definition
m['paper2'] = m.doi.fillna(m.last_author.fillna(''))  # alt: DOI-less entries merged per lab (no year split)
m['labn'] = m.last_author.map(norm_lab)
m['host_enz'] = (m.tgt_cls == 'enzyme').astype(int)
m['vir_oth'] = (m.tgt_cls == 'viral other').astype(int)
m['enz_comb'] = m.host_enz | m.vir_oth
print('analysis entries', len(m), 'errors', m.y.sum(), '| EM', (m.method == 'EM').sum(), m[m.method == 'EM'].y.sum())
EM = m[m.method == 'EM'].copy(); XR = m[m.method == 'X-ray'].copy()

def rate(d, col='y'): return f'{d[col].mean()*100:.1f}% ({int(d[col].sum())}/{len(d)})'
def two(d, t, col='y'):
    a = int(((d[t] == 1) & (d[col] == 1)).sum()); b = int(((d[t] == 1) & (d[col] == 0)).sum())
    c = int(((d[t] == 0) & (d[col] == 1)).sum()); dd = int(((d[t] == 0) & (d[col] == 0)).sum())
    return a, b, c, dd
def fisher(d, t, label, stratum, col='y'):
    a, b, c, dd = two(d, t, col)
    orr, p = stats.fisher_exact([[a, b], [c, dd]])
    r = dict(test=label, stratum=stratum, model='Fisher exact (unadjusted)', OR=orr, p=p, n=len(d), cases_in=f'{a}/{a+b}', cases_out=f'{c}/{c+dd}')
    TESTS.append(r); return r
def logit(d, f, t, label, stratum, grp='paper'):
    mod = smf.glm(f, d, family=sm.families.Binomial()).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(d[grp])[0]})
    k = [x for x in mod.params.index if x == t][0]; ci = np.exp(mod.conf_int().loc[k])
    a, b, c, dd = two(d, t)
    r = dict(test=label, stratum=stratum, model=f + f' | cluster={grp}', OR=np.exp(mod.params[k]), lo=ci.iloc[0], hi=ci.iloc[1], p=mod.pvalues[k], n=len(d),
             cases_in=f'{a}/{a+b}', cases_out=f'{c}/{c+dd}')
    TESTS.append(r); return r
def show(r): print(f"  {r['test']:62s} [{r['stratum']}] in {r['cases_in']:>8s} out {r['cases_out']:>9s} OR {r['OR']:.2f}" + (f" ({r['lo']:.2f}-{r['hi']:.2f})" if 'lo' in r and pd.notna(r.get('lo')) else '') + f" p={r['p']:.2g}")

print('\n== 2. Entry level, EM (reproduce investigator)')
for t in ['host_enz', 'vir_oth', 'enz_comb']:
    show(fisher(EM, t, f'entry: {t}', 'EM'))
    show(logit(EM, f'y ~ {t} + C(ycat) + res', t, f'entry: {t}, year+res', 'EM'))
    show(logit(EM, f'y ~ {t} + C(ycat) + res', t, f'entry: {t}, year+res, lab-clustered', 'EM', grp='labn'))
show(fisher(XR, 'enz_comb', 'entry: enz_comb', 'X-ray'))
print('  X-ray expected errors among 50 enzyme entries at X-ray base rate:', round(50 * XR.y.mean(), 2), '| P(0 errors) =', round((1 - XR.y.mean()) ** 50, 3))

# ---------------------------------------------------------------- 3. paper level
print('\n== 3. Paper level, EM')
def paper_tab(D, key='paper'):
    P = D.groupby(key).agg(y=('y', 'max'), host_enz=('host_enz', 'max'), vir_oth=('vir_oth', 'max'), enz_comb=('enz_comb', 'max'),
                           host_mode=('tgt_cls', lambda s: int(s.mode().iat[0] == 'enzyme')), n=('y', 'size'), ycat=('ycat', lambda s: s.mode().iat[0]),
                           res=('res', 'median'), labn=('labn', 'first'), sars2=('sars2', 'max'), period=('period', lambda s: s.mode().iat[0])).reset_index()
    P['ln'] = np.log2(P.n); return P
P = paper_tab(EM)
print('  papers', len(P), 'error papers', int(P.y.sum()))
for t in ['host_enz', 'host_mode', 'vir_oth', 'enz_comb']:
    show(fisher(P, t, f'paper: {t} (any entry in class)' if t != 'host_mode' else 'paper: host enzyme (modal class)', 'EM papers'))
    try: show(logit(P, f'y ~ {t} + C(ycat) + res + ln', t, f'paper: {t}, year+res+log n', 'EM papers', grp='labn'))
    except Exception as ex: print('   logit failed', t, ex)
P2 = paper_tab(EM, 'paper2')
show(fisher(P2, 'host_enz', 'paper (alt def: DOI-less merged per lab): host_enz', 'EM papers alt'))
show(fisher(P2, 'enz_comb', 'paper (alt def: DOI-less merged per lab): enz_comb', 'EM papers alt'))
# papers that mix enzyme and non-enzyme targets
mix = EM.groupby('paper').host_enz.agg(['min', 'max']); print('  papers with mixed enzyme / non-enzyme entries:', int(((mix['min'] == 0) & (mix['max'] == 1)).sum()))
print('  error labs, host enzyme papers:', P[(P.host_enz == 1) & (P.y == 1)].labn.nunique(), '| combined:', P[(P.enz_comb == 1) & (P.y == 1)].labn.nunique(),
      '| error papers combined:', int(P[(P.enz_comb == 1)].y.sum()), 'of', int(P.enz_comb.sum()))

# ---------------------------------------------------------------- 4. period strata
print('\n== 4. By period (EM entries and papers)')
for per in ['<=2019', '2020-21', '2022+']:
    d = EM[EM.period == per]; pp = P[P.period == per]
    for t in ['host_enz', 'enz_comb']:
        r = fisher(d, t, f'entry: {t}', f'EM {per}'); r2 = fisher(pp, t, f'paper: {t}', f'EM papers {per}')
        print(f'  {per:8s} {t:9s} entries in {r["cases_in"]:>7s} out {r["cases_out"]:>9s} OR {r["OR"]:.2f} p={r["p"]:.2g} | papers in {r2["cases_in"]:>6s} out {r2["cases_out"]:>8s} OR {r2["OR"]:.2f} p={r2["p"]:.2g}')
for t in ['host_enz', 'enz_comb']:
    tabs = []
    for per, d in EM.groupby('period'):
        a, b, c, dd = two(d, t); tabs.append(np.array([[a, b], [c, dd]]) + 0.0)
    st = StratifiedTable(tabs); print(f'  Mantel-Haenszel {t}: OR {st.oddsratio_pooled:.2f} p={st.test_null_odds().pvalue:.2g} | homogeneity p={st.test_equal_odds().pvalue:.2g}')
    TESTS.append(dict(test=f'entry: {t}, Mantel-Haenszel by period', stratum='EM', model='MH', OR=st.oddsratio_pooled, p=st.test_null_odds().pvalue, n=len(EM)))
# non-SARS-CoV-2 EM only (SARS2 entries are never enzyme; removes the huge low-rate spike block)
show(logit(EM[~EM.sars2.astype(bool)], 'y ~ host_enz + C(ycat) + res', 'host_enz', 'entry: host_enz, year+res, non-SARS2', 'EM non-SARS2'))
show(logit(EM[~EM.sars2.astype(bool)], 'y ~ enz_comb + C(ycat) + res', 'enz_comb', 'entry: enz_comb, year+res, non-SARS2', 'EM non-SARS2'))

# ---------------------------------------------------------------- 5. fragility: leave-one-paper-out and drop-top-k
print('\n== 5. Fragility')
def lopo(D, t, key, f):
    errp = D[(D[t] == 1) & (D.y == 1)][key].value_counts()
    out = []
    for pp in errp.index:
        d = D[D[key] != pp]
        try:
            mod = smf.glm(f, d, family=sm.families.Binomial()).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(d.paper)[0]})
            out.append((pp, int(errp[pp]), np.exp(mod.params[t]), mod.pvalues[t]))
        except Exception: pass
    return pd.DataFrame(out, columns=[key, 'n_err_entries_removed', 'OR', 'p'])
for t in ['host_enz', 'enz_comb']:
    L = lopo(EM, t, 'paper', f'y ~ {t} + C(ycat) + res')
    print(f'  {t} leave-one-paper-out: OR range {L.OR.min():.2f}-{L.OR.max():.2f}, max p {L.p.max():.2g}')
    L2 = lopo(EM, t, 'labn', f'y ~ {t} + C(ycat) + res')
    print(f'  {t} leave-one-lab-out (normalised labs): OR range {L2.OR.min():.2f}-{L2.OR.max():.2f}, max p {L2.p.max():.2g}')
    # drop the k error papers with most error entries, cumulatively (entry level), and paper-level Fisher
    order = EM[(EM[t] == 1) & (EM.y == 1)].paper.value_counts().index.tolist()
    for k in range(0, len(order) + 1):
        d = EM[~EM.paper.isin(order[:k])]; pp = P[~P.paper.isin(order[:k])]
        a, b, c, dd = two(d, t); pf = stats.fisher_exact([[a, b], [c, dd]])[1]
        a2, b2, c2, d2 = two(pp, t); pf2 = stats.fisher_exact([[a2, b2], [c2, d2]])[1]
        if pf > 0.05 and pf2 > 0.05:
            print(f'  {t}: dropping the top {k} error papers (by error-entry count) makes entry Fisher p={pf:.2g} ({a}/{a+b}) and paper Fisher p={pf2:.2g} ({a2}/{a2+b2})'); break
    else:
        print(f'  {t}: still p<0.05 after dropping all but... check')
    # what is left after dropping to 1 error paper per lab? (paper level, each lab counted once)
    Pl = P.sort_values('y', ascending=False).groupby(['labn', t]).head(1)
    show(fisher(Pl, t, f'paper: {t}, one paper per lab-x-class (lab-level)', 'EM labs'))

# ---------------------------------------------------------------- 6. does the erroneous chain bind the enzyme? (unit level)
print('\n== 6. Chain/unit level: antigen class contacted by the antibody itself')
R = pd.read_csv(f'{AC}/pdbwide_ab_chain_roles.csv.gz')
CH = ch[~ch.pdb.isin(amy)].merge(R[['pdb', 'chain', 'ab_role_contact', 'ab_ag_cls', 'ab_ag_area']], on=['pdb', 'chain'], how='left')
CH = CH.merge(m[['pdb', 'tgt_cls', 'tgt_name']], on='pdb', how='left')
errC = CH[CH.err_first & CH.tgt_cls.isin(['enzyme', 'viral other']) & (CH.method == 'EM')]
print(errC[['pdb', 'chain', 'ctype', 'vhh', 'err_modes', 'ab_role_contact', 'ab_ag_cls', 'ab_ag_area', 'tgt_cls', 'role', 'chain_desc']].to_string())
print('  erroneous chains in enzyme-target entries whose own contact class is enzyme/viral other:', int(errC.ab_ag_cls.isin(['enzyme', 'viral other']).sum()), 'of', len(errC))
CH['dyear'] = pd.to_datetime(CH.deposit).dt.year
U = CH[CH.method == 'EM'].groupby('unit').agg(err=('err_first', 'max'), anyrun=('err_any', 'max'), res=('res', 'median'), dyear=('dyear', 'min'),
                                             agcls=('ab_ag_cls', lambda s: s.mode().iat[0] if s.notna().any() else 'unknown'), vhh=('vhh', 'max'),
                                             paper=('paper', 'first'), lab=('last_author', 'first'), v_id=('v_id', 'mean')).reset_index()
U = U[U.err | ~U.anyrun].copy(); U['y'] = U.err.astype(int); U['ycat'] = U.dyear.clip(lower=2017).astype(str)
U['host_enz'] = (U.agcls == 'enzyme').astype(int); U['enz_comb'] = U.agcls.isin(['enzyme', 'viral other']).astype(int); U['vhh'] = U.vhh.astype(int)
U['labn'] = U.lab.map(norm_lab)
for t in ['host_enz', 'enz_comb']:
    show(fisher(U, t, f'unit (antibody-in-paper): {t} by own contacts', 'EM units'))
    show(logit(U, f'y ~ {t} + C(ycat) + res', t, f'unit: {t}, year+res', 'EM units'))
    show(logit(U, f'y ~ {t} + vhh + C(ycat) + res', t, f'unit: {t}, year+res+VHH', 'EM units'))
    for v in [0, 1]:
        show(fisher(U[U.vhh == v], t, f'unit: {t} within VHH={v}', 'EM units'))
U.to_csv(f'{OUT}/units_em_enzyme.csv.gz', index=False)

# ---------------------------------------------------------------- 7. classification sensitivity: "target has an EC number" (non-viral)
print('\n== 7. Classification sensitivity')
uni = json.load(open(f'{SP}/deeper/cache/antigen_competition/uniprot_info.json'))
def has_ec(keys):
    if not isinstance(keys, str): return 0
    return int(any(bool((uni.get(k) or {}).get('ec')) for k in keys.split('|') if not k.startswith('g95_')))
EM['ec_any'] = EM.tgt_keys.map(has_ec)
print(pd.crosstab(EM.tgt_cls, EM.ec_any).to_string())
EM['ec_nonviral'] = ((EM.ec_any == 1) & (EM.tgt_cls != 'viral surface glycoprotein') & (EM.tgt_cls != 'viral capsid') & (EM.tgt_cls != 'viral other')).astype(int)
show(fisher(EM, 'ec_nonviral', 'entry: target has EC number (non-viral), any class', 'EM'))
show(logit(EM, 'y ~ ec_nonviral + C(ycat) + res', 'ec_nonviral', 'entry: target has EC (non-viral), year+res', 'EM'))
EM['ec_other_cls'] = ((EM.ec_nonviral == 1) & (EM.tgt_cls != 'enzyme')).astype(int)
print('  EC-bearing targets classed elsewhere (e.g. transporter/receptor):', rate(EM[EM.ec_other_cls == 1]))
print(EM[EM.ec_other_cls == 1].groupby('tgt_cls').y.agg(['sum', 'size']).to_string())

# ---------------------------------------------------------------- 8. error-mode composition
print('\n== 8. Error modes, EM first-occurrence error entries')
x = EM[EM.y == 1].copy(); x['grp'] = np.where(x.host_enz == 1, 'host enzyme', np.where(x.vir_oth == 1, 'viral other', 'other'))
x['A'] = x.modes.fillna('').str.contains('A_FR1Nterm').astype(int); x['B'] = x.modes.fillna('').str.contains('B_CDR2FR3').astype(int)
print(x.groupby('grp')[['A', 'B']].agg(['sum', 'size']).to_string())
print('  host-enzyme A_FR1Nterm entries by paper:', x[(x.grp == 'host enzyme') & (x.A == 1)].paper.nunique(), 'papers')
print('  B_CDR2FR3 entries in enzyme groups by paper:', x[(x.grp != 'other') & (x.B == 1)].paper.value_counts().to_dict())

# ---------------------------------------------------------------- 9. viral other: what is in it?
print('\n== 9. "viral other" (labelled viral enzymes) entries, EM')
print(EM[EM.vir_oth == 1][['pdb', 'y', 'last_author', 'dyear', 'ab_status', 'tgt_name', 'tgt_desc', 'modes']].to_string())

# ---------------------------------------------------------------- 10. full host-enzyme EM list (controls too), for manual audit
EM[EM.host_enz == 1][['pdb', 'y', 'last_author', 'dyear', 'res', 'paper', 'ab_status', 'tgt_name', 'tgt_desc', 'modes']].sort_values(['y', 'last_author']).to_csv(f'{OUT}/em_host_enzyme_entries.csv', index=False)
P.to_csv(f'{OUT}/em_papers.csv', index=False)

T = pd.DataFrame(TESTS); T['q_within_verify'] = multipletests(T.p.fillna(1), method='fdr_bh')[1]
T.to_csv(f'{OUT}/verify_tests.csv', index=False)
print('\nwrote', len(T), 'tests')
