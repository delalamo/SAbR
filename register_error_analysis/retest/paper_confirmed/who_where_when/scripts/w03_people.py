"""People, places and experience (people_network lens, deeper README 4.3 and 6) on the paper-confirmed case set.
Study level (citation DOI; unpublished entries grouped by last author), cryo-EM primary, X-ray secondary.
Reuses: people_network cache (studies.csv.gz, entries_people.csv.gz, audit_authors_keyed.csv.gz, study_hub_flags.csv,
study_carrier_persistence.csv.gz), the confounding verifier's study frame (new-team flag + covariates) and the integrity
verifier's frame (all-name and co-author-rule new-team flags), paper_meta paper_authors (affiliations, countries).
People are reported only as pseudonyms (same P#### scheme as the lens: index in the sorted list of person keys)."""
import sys, re, collections, unicodedata, hashlib
sys.path.insert(0, '.'); sys.path.insert(0, f'/home/user/SAbR/register_error_analysis/retest/deeper/people_network')
from ww_common import *
from common import strat_exact, bh
PN = f'{SP}/deeper/cache/people_network'
E = entry_flags()
EP = pd.read_csv(f'{PN}/entries_people.csv.gz')
ST = pd.read_csv(f'{PN}/studies.csv.gz', parse_dates=['dep0'])
U = unit_flags(E, EP[['pdb', 'study']], 'study')
ST = ST.merge(U, on='study', how='left')
ST = add_yc(ST, 'year')
print('studies by method x flag:'); print(ST.groupby('method')[['case', 'case_i', 'ctrl', 'rest_w', 'rest', 'clean']].sum())

# ------------------------------------------------------------------ 1. experience (new team), EM study frame of the verifier
SF = pd.read_csv(f'{SP}/deeper/cache/verify_people_newteam_conf/study_frame.csv.gz')
SV = pd.read_csv(f'{SP}/deeper/cache/verify_newteam/study_frame_verify2.csv.gz')[['study', 'nt_strict', 'nt_conf']]
SV['study'] = SV.study.str.replace(r'^doi:', '', regex=True)
SF = SF.merge(SV, on='study', how='left').merge(U, on='study', how='left')
SF = add_yc(SF, 'dep_year'); SF['lab'] = SF.lab_key
SF['l_team_abem_na'] = np.log2(1 + SF.team_max_abem_na); SF['l_team_abem_all'] = np.log2(1 + SF.team_max_abem)
SF['has_hub'] = SF.has_hub.astype(float)
for c in ['case', 'case_i', 'ctrl', 'rest', 'rest_w', 'win_all', 'gen', 'clean']:
    SF[c] = SF[c].fillna(False).astype(bool)
B = '{YC} + res_med + l2_nv'
for x, ref in [('new', 'study OR 1.99 (1.36-2.91); <=2021 0.96-2.23 n.s.; 2022-23 4.7-8.4 [people_network; verify integrity]'),
               ('nt_strict', 'all-name flag: pooled 2.06; <=2021 OR 0.96 [verify integrity]'),
               ('nt_conf', 'co-author identity rule: pooled 2.01; <=2021 1.15-1.26 [verify integrity]'),
               ('first_em_team_na', 'no prior cryo-EM entry: OR 1.77 (1.16-2.68) [people_network]'),
               ('l_team_abem_na', 'most experienced author prior ab cryo-EM, per doubling 0.89 (0.81-0.97) [people_network]'),
               ('l_team_abem_all', 'same, all names counted (collision-inflating); not reported in lens'),
               ('l_lab_prior_ab', 'last author prior antibody entries per doubling 0.90 (0.83-0.98) [people_network]; 0.88 [antigen_competition]'),
               ('has_hub', 'multi-lab hub person on study OR 0.67 (0.39-1.13) [people_network]')]:
    record(f'experience: {x}', 'study', 'EM', x, B + ', lab-clustered', run_groups(SF, x, B, 'lab'), ref)
record('experience: new (no nanobody studies)', 'study', 'EM', 'new', B, run_groups(SF[SF.vhh_any == 0], 'new', B, 'lab'), 'no-nanobody OR 2.58; <=2023 2.70 adj [verify conf]')
record('experience: new (nanobody studies)', 'study', 'EM', 'new', B, run_groups(SF[SF.vhh_any == 1], 'new', B, 'lab'), 'nanobody OR 0.82 [verify conf]')
record('experience: new, deposited <=2021 only', 'study', 'EM', 'new', B, run_groups(SF[SF.dep_year <= 2021], 'new', B, 'lab'), '<=2021: 0.96 (all names) / 2.23 (their flag) [verify integrity]')
for x in ['nt_strict', 'l_team_abem_na', 'l_team_abem_all', 'l_lab_prior_ab', 'first_em_team_na']:
    record(f'experience: {x} (+China, +SARS2)', 'study', 'EM', x, B + ' + china + sars2', run_groups(SF, x, B + ' + china + sars2', 'lab'), 'China-led: _na flag OR 6.1 vs all-name 1.10 [verify conf]')
    record(f'experience: {x} (non-China-led studies)', 'study', 'EM', x, B, run_groups(SF[SF.china == 0], x, B, 'lab'), 'outside China both flags agree: 1.69 / 2.36 [verify conf]')
record('experience: new (+China, +SARS2)', 'study', 'EM', 'new', B + ' + china + sars2', run_groups(SF, 'new', B + ' + china + sars2', 'lab'), '2020-21 uncommon surnames OR 1.81; +China/USA 2.16 [verify integrity]')
# one study per lab (500 draws), case vs window controls
rng = np.random.default_rng(1)
d0 = SF[SF.case | (SF.ctrl & ~SF.gen & ~SF.case_i)].copy(); d0['_y'] = d0.case.astype(int)
ors = []
for i in range(500):
    s = d0.sample(frac=1, random_state=int(rng.integers(1e9))).drop_duplicates('lab')
    r = mh(s, '_y', 'new', 'yc3')
    if 'OR' in r: ors.append((r['OR'], r['p']))
ors = np.array(ors)
print(f'one study per lab (500 draws), case vs window controls, MH by year: median OR {np.median(ors[:, 0]):.2f}, median p {np.median(ors[:, 1]):.2g}, '
      f'share p<0.05 {np.mean(ors[:, 1] < 0.05):.2f}; cases per draw ~{int(d0.drop_duplicates("lab")._y.sum())}')
ROWS.append(dict(trend='experience: new, one study per lab (500 draws)', level='study', method='EM', x='new', spec='MH by yc3', group='case',
                 OR=np.median(ors[:, 0]), p=np.median(ors[:, 1]), note=f'share of draws p<0.05: {np.mean(ors[:, 1] < 0.05):.2f}'))
# crude table in window
t = d0.groupby(['yc3', 'new'])._y.agg(['sum', 'size']).unstack(); print('case vs window controls, new team by year:\n', t)

# ------------------------------------------------------------------ 2. lab persistence and carrier (people_network s06)
CP = pd.read_csv(f'{PN}/study_carrier_persistence.csv.gz').drop(columns=['y']).merge(U, on='study', how='left')
CP = add_yc(CP, 'year'); CP['lab'] = CP.lab.fillna('?')
for c in ['case', 'case_i', 'ctrl', 'rest', 'rest_w', 'win_all', 'gen', 'clean']:
    CP[c] = CP[c].fillna(False).astype(bool)
record('lab persistence: same lab had an earlier error study', 'study', 'EM', 'exp_own', B + ' + l_n_prior_own (labs with a prior study)',
       run_groups(CP[CP.n_prior_own > 0], 'exp_own', '{YC} + res_med + l2_nv + l_n_prior_own', 'lab'), 'OR 2.19 p=0.007 -> 1.69 adj; 1.04 if earlier error on another antigen [people_network]')
record('carrier: co-author on an earlier error study of another lab', 'study', 'EM', 'exp_x', B + ' + l_n_prior_x',
       run_groups(CP[CP.n_prior_x > 0], 'exp_x', '{YC} + res_med + l2_nv + l_n_prior_x', 'lab'), 'OR 1.02 (0.52-1.99) [people_network]')
# lab of a paper-case study had an earlier paper-case study? (descriptive)
cs = CP[CP.case].sort_values('dep0')
print('\npaper-case EM studies:', len(cs), '| labs:', cs.lab.nunique(), '| labs with >1 paper-case study:', int((cs.lab.value_counts() > 1).sum()))

# ------------------------------------------------------------------ 3. entity scans (study level)
A = pd.read_csv(f'{PN}/audit_authors_keyed.csv.gz')
pseudo = {k: f'P{i:04d}' for i, k in enumerate(sorted(set(A.pkey)))}
labpseudo = lambda k: pseudo.get(k, 'L?')
ALL = pd.read_csv(f'{PN}/studies.csv.gz'); ALL['akeys'] = ALL.akeys.fillna('')
pair = collections.Counter()
for ks in ALL.akeys:
    ks = sorted(set(ks.split('|')) - {''})
    if len(ks) > 40: continue
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)): pair[(ks[i], ks[j])] += 1
strong = [(a, b) for (a, b), c in pair.items() if c >= 3]
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
nodes = sorted({x for e in strong for x in e}); ix = {k: i for i, k in enumerate(nodes)}
M = coo_matrix((np.ones(len(strong)), ([ix[a] for a, b in strong], [ix[b] for a, b in strong])), shape=(len(nodes), len(nodes)))
_, labc = connected_components(M, directed=False); grp_of = {k: f'G{labc[i]:04d}' for k, i in ix.items()}
PA = pd.read_csv(f'{D}/paper_meta/paper_authors.csv.gz')
PM = pd.read_csv(f'{D}/paper_meta/paper_meta.csv', usecols=['doi', 'last_country', 'countries'])
KW = re.compile(r'(universit|institut|cent(er|re)|college|hospital|school|laborator|academy|inc\b|ltd|gmbh|llc|corporation|'
                r'company|pharma|therapeutics|biosciences|foundation|facility|core|consortium|agency|council)', re.I)
FAC = re.compile(r'(cryo-?em (core|facility|center|centre)|electron microscopy (core|facility|center|centre|resource)|'
                 r'national (cryo|center for cryo|ccp-em)|pacific northwest cryo|pncc|national center for cryo-?em|ncef|'
                 r'new york structural biology center|nysbc|stanford-slac|slac|ebic|electron bio-imaging|'
                 r'diamond light|simons electron|(cryo|electron)[- ]microscopy (platform|unit|laboratory)|'
                 r'core facility|shared resource|structural biology core|imaging (core|facility|center))', re.I)
def norm_inst(seg):
    s = unicodedata.normalize('NFKD', seg).encode('ascii', 'ignore').decode().lower()
    s = re.sub(r'\b(the|of|and|for|&)\b', ' ', s); s = re.sub(r'[^a-z ]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()
inst_of = collections.defaultdict(set); fac_of = collections.defaultdict(set)
for dd, aff in zip(PA.doi, PA.aff.fillna('')):
    for a in re.split(r';|\|', aff):
        segs = [x.strip() for x in a.split(',') if x.strip()]
        found = [norm_inst(x) for x in segs if KW.search(x) and not re.search(r'department|dept|division|faculty|school of|program', x, re.I)]
        if found: inst_of[dd].add(found[-1])
        for mm in FAC.finditer(a): fac_of[dd].add(norm_inst(mm.group(0)).replace('centre', 'center'))
pmc = PM.set_index('doi')
hub = pd.read_csv(f'{PN}/study_hub_flags.csv').set_index('study').has_hub
inst_ids = {}
def inst_label(k):
    return inst_ids.setdefault(k, f'I{len(inst_ids):03d}')
ENT = []


def scan(Sx, tag, case='case'):
    Sx = Sx[Sx[case] | (Sx.ctrl & ~Sx.gen & ~Sx.case_i)].reset_index(drop=True).copy()
    y = Sx[case].astype(int).values; st = Sx.yc3.values
    sets_any = Sx.akeys.fillna('').map(lambda s: set(s.split('|')) - {''})
    sets_nl = Sx.akeys_nonlast.fillna('').map(lambda s: set(s.split('|')) - {''})
    pub = Sx.doi.notna().values
    cls = {'person (any position)': (sets_any, lambda k: pseudo.get(k, 'P?')),
           'person (non-last author)': (sets_nl, lambda k: pseudo.get(k, 'P?')),
           'lab (last author)': (Sx.lab_pkey.map(lambda k: {k} if isinstance(k, str) else set()), lambda k: pseudo.get(k, 'L?')),
           'co-author group (>=3 shared studies)': (sets_any.map(lambda s: {grp_of[k] for k in s if k in grp_of}), str),
           'institution (published)': (Sx.doi.map(lambda dd: inst_of.get(dd, set()) if isinstance(dd, str) else set()), inst_label),
           'facility mention (published)': (Sx.doi.map(lambda dd: fac_of.get(dd, set()) if isinstance(dd, str) else set()), str),
           'country, last author (published)': (Sx.doi.map(lambda dd: {pmc.last_country.get(dd)} if isinstance(dd, str) and dd in pmc.index and isinstance(pmc.last_country.get(dd), str) else set()), str),
           'country, any author (published)': (Sx.doi.map(lambda dd: set(str(pmc.countries.get(dd, '')).split('|')) - {'', 'nan'} if isinstance(dd, str) and dd in pmc.index else set()), str),
           'any facility mention (aggregate)': (Sx.doi.map(lambda dd: {'any_facility'} if isinstance(dd, str) and fac_of.get(dd) else set()), str),
           'any multi-lab hub person (aggregate)': (Sx.study.map(lambda s: {'has_hub'} if hub.get(s, False) else set()), str)}
    for cname, (sets, lab) in cls.items():
        inv = collections.defaultdict(list)
        for i, s in enumerate(sets):
            for k in s: inv[k].append(i)
        rows = []
        for k, ii in inv.items():
            m = np.zeros(len(Sx), bool); m[ii] = True
            if m.sum() < 3 or (y[m].sum() == 0): continue
            o, ex, pu, p2, orr = strat_exact(m, y, st)
            rows.append(dict(scan=tag, cls=cname, entity=lab(k), n=int(m.sum()), n_case=int(o), expected=ex, OR_MH=orr, p_upper=pu, p_two=p2,
                             n_case_labs=Sx.loc[m & (y == 1), 'lab'].nunique()))
        if rows:
            R = pd.DataFrame(rows); R['q_within_class'] = bh(R.p_upper.values); ENT.append(R)
            b = R.sort_values('p_upper').iloc[0]
            print(f'  [{tag}] {cname:40s} tested {len(R):4d} | entities in >=2 case studies {int((R.n_case >= 2).sum()):3d} | min q {R.q_within_class.min():.2f} '
                  f'(top {b.entity}: {b.n_case}/{b.n} studies, exp {b.expected:.2f}, case labs {b.n_case_labs})')


print('\n== entity scans: paper-case studies vs window control studies (exact stratified by year, >=3 studies, >=1 case)')
for meth in ['EM', 'all']:
    Sx = ST[ST.grp.isin(['err', 'clean'])].copy() if meth == 'all' else ST[(ST.method == 'EM') & ST.grp.isin(['err', 'clean'])].copy()
    if meth == 'all':   # one row per study across methods (a paper with EM and X-ray entries is one study)
        Sx = Sx.groupby('study').agg(akeys=('akeys', lambda v: '|'.join(v.fillna(''))), akeys_nonlast=('akeys_nonlast', lambda v: '|'.join(v.fillna(''))),
                                     lab=('lab', 'first'), lab_pkey=('lab_pkey', 'first'), doi=('doi', 'first'), yc3=('yc3', 'min'),
                                     case=('case', 'max'), case_i=('case_i', 'max'), ctrl=('ctrl', 'min'), gen=('gen', 'max')).reset_index()
    for c in ['case', 'case_i', 'ctrl', 'gen']: Sx[c] = Sx[c].fillna(False).astype(bool)
    print(f'{meth}: case studies {int(Sx.case.sum())}, case labs {Sx[Sx.case].lab.nunique()}, window control studies {int((Sx.ctrl & ~Sx.gen).sum())}')
    scan(Sx, f'{meth} strict'); scan(Sx, f'{meth} incl copies', case='case_i')
ENT = pd.concat(ENT); ENT.to_csv(f'{T}/w03_entity_enrichment_pseudonymised.csv', index=False)
s = ENT.groupby(['scan', 'cls']).agg(n_tested=('entity', 'size'), n_in_ge2_case_studies=('n_case', lambda x: int((x >= 2).sum())),
                                      min_p=('p_upper', 'min'), min_q=('q_within_class', 'min')).reset_index()
s.to_csv(f'{T}/w03_entity_enrichment_summary.csv', index=False)
print(s.round(3).to_string(index=False))

# ------------------------------------------------------------------ 4. co-authorship links among paper-case studies vs half-year matched window controls
print('\n== shared-author links between paper-case studies of different labs vs half-year matched samples of window control studies')
Sx = ST[(ST.method == 'EM') & ST.grp.isin(['err', 'clean'])].copy()
for c in ['case', 'ctrl', 'gen', 'case_i']: Sx[c] = Sx[c].fillna(False).astype(bool)
Sx['sets'] = Sx.akeys.fillna('').map(lambda s: set(s.split('|')) - {''})
Sx['sets_nl'] = Sx.akeys_nonlast.fillna('').map(lambda s: set(s.split('|')) - {''})
COMMON = set(A[A.common].sur.unique())
def edges(df, col='sets', na=False):
    rows = df[[col, 'lab']].values; n = 0
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            if rows[i][1] == rows[j][1]: continue
            sh = rows[i][0] & rows[j][0]
            if na: sh = {k for k in sh if k.split(':')[-1].split('_')[0] not in COMMON}
            if sh: n += 1
    return n
cases = Sx[Sx.case]; pool = Sx[Sx.ctrl & ~Sx.gen & ~Sx.case_i]
res = []
for col, na, lab in [('sets', False, 'any shared audit author, different labs'), ('sets_nl', False, 'shared non-last author, different labs'),
                     ('sets', True, 'any shared author, collision-prone surnames dropped')]:
    obs = edges(cases, col, na); null = []
    for i in range(1000):
        samp = pd.concat([pool[pool.half == h].sample(n=k, replace=len(pool[pool.half == h]) < k, random_state=i)
                          for h, k in cases.half.value_counts().items() if len(pool[pool.half == h]) > 0])
        null.append(edges(samp, col, na))
    null = np.array(null)
    res.append(dict(link=lab, observed=obs, null_mean=null.mean(), ratio=obs / max(null.mean(), 1e-9), p_more=float(np.mean(null >= obs)), p_less=float(np.mean(null <= obs)), n_case_studies=len(cases)))
R = pd.DataFrame(res); print(R.round(3).to_string(index=False)); R.to_csv(f'{T}/w03_network_links.csv', index=False)
save_rows(f'{T}/w03_people_tests.csv')
