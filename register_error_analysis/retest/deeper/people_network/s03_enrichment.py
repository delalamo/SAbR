#!/usr/bin/env python3
"""Are specific people, co-author groups, core-staff/facility hubs, labs, institutions or countries over-represented
among first-occurrence error studies (cryo-EM), relative to clean studies deposited in the same year category?

Test: exact stratified Fisher (conditional, hypergeometric per deposit-year stratum, convolved), one-sided for
enrichment; BH within each entity class and across all classes. Unit: study (paper); entry level as a secondary
analysis (inflated by multi-entry studies).  Entities need >= 3 cryo-EM studies (error or clean).
Person identities are pseudonymised in the repo tables (P#### ids).
"""
import sys, re, collections, unicodedata, hashlib
sys.path.insert(0, '/home/user/SAbR/register_error_analysis/retest/deeper/people_network')
import numpy as np, pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from common import OUT, SP, CACHE, load_studies, load_entries, strat_exact, bh

S = load_studies('EM')
E = load_entries('EM')
ALL = pd.read_csv(f'{CACHE}/studies.csv.gz')  # all studies, both methods, for group definitions
ALL['akeys'] = ALL.akeys.fillna('')
A = pd.read_csv(f'{CACHE}/audit_authors_keyed.csv.gz')
print('EM studies: error', S.y.sum(), 'clean', (1 - S.y).sum(), '| entries: error', E.y.sum(), 'clean', (1 - E.y).sum())

# pseudonyms
allkeys = sorted(set(A.pkey))
pseudo = {k: f'P{i:04d}' for i, k in enumerate(allkeys)}

# ------------------------------------------------------------------ co-author groups (strong ties, all studies)
pair = collections.Counter()
for ks in ALL.akeys:
    ks = sorted(set(ks.split('|')) - {''})
    if len(ks) > 40:   # consortium-size lists would create cliques
        continue
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            pair[(ks[i], ks[j])] += 1
strong = [(a, b) for (a, b), c in pair.items() if c >= 3]
nodes = sorted({x for e in strong for x in e}); ix = {k: i for i, k in enumerate(nodes)}
M = coo_matrix((np.ones(len(strong)), ([ix[a] for a, b in strong], [ix[b] for a, b in strong])), shape=(len(nodes), len(nodes)))
ncomp, lab = connected_components(M, directed=False)
grp_of = {k: f'G{lab[i]:04d}' for k, i in ix.items()}
gsize = collections.Counter(grp_of.values())
print('strong-tie co-author groups:', ncomp, 'people in groups', len(nodes), 'largest', gsize.most_common(3))

# ------------------------------------------------------------------ multi-lab hubs (core-staff / facility proxies)
IDX = pd.read_pickle(f'{CACHE}/idx_parsed.pkl')
em = IDX[IDX.em & (IDX.dep >= '2015-01-01')]
labs_of = collections.defaultdict(set); n_em = collections.Counter(); n_last = collections.Counter()
for lb, sk in zip(em.lab, em.skeys):
    ks = sk.split('|')
    for k in set(ks):
        n_em[k] += 1
    n_last[ks[-1]] += 1
    for k in set(ks[:-1]):
        labs_of[k].add(lb)
COMMON = set(pd.read_csv(f'{CACHE}/audit_authors_keyed.csv.gz').query('common').sur.unique())
hubs = {k for k, v in labs_of.items() if len(v) >= 5 and n_last[k] / max(1, n_em[k]) < 0.25 and k.split('_')[0] not in COMMON
        and len(k.split('_')[1]) >= 1}
print('PDB-wide cryo-EM multi-lab hubs (>=5 labs as non-last author since 2015, <25% as last author, uncommon surname):', len(hubs))
skeys_by_pdb = A.groupby('pdb').skey.apply(set).to_dict()
skey_last = A[A.is_last].set_index('pdb').skey.to_dict()


def study_hubs(pdbs):
    h = set()
    for p in pdbs:
        h |= {k for k in skeys_by_pdb.get(p, set()) if k in hubs and k != skey_last.get(p)}
    return h


Eall = pd.read_csv(f'{CACHE}/entries_people.csv.gz')
pdbs_of_study = Eall[Eall.method == 'EM'].groupby('study').pdb.apply(list).to_dict()
S['hubs'] = [study_hubs(pdbs_of_study.get(s, [])) for s in S.study]
E['hubs'] = [study_hubs([p]) for p in E.pdb]
S['has_hub'] = S.hubs.map(len) > 0
E['has_hub'] = E.hubs.map(len) > 0

# ------------------------------------------------------------------ affiliations: institutions, facilities, countries
PA = pd.read_csv(f'{OUT}/../paper_meta/paper_authors.csv.gz')
PM = pd.read_csv(f'{OUT}/../paper_meta/paper_meta.csv', usecols=['doi', 'last_country', 'countries'])
KW = re.compile(r'(universit|institut|cent(er|re)|college|hospital|school|laborator|academy|inc\b|ltd|gmbh|llc|corporation|'
                r'company|pharma|therapeutics|biosciences|foundation|facility|core|consortium|agency|council)', re.I)
FAC = re.compile(r'(cryo-?em (core|facility|center|centre)|electron microscopy (core|facility|center|centre|resource)|'
                 r'national (cryo|center for cryo|ccp-em)|pacific northwest cryo|pncc|national center for cryo-?em|ncef|'
                 r'new york structural biology center|nysbc|stanford-slac|slac|ebic|electron bio-imaging|'
                 r'diamond light|simons electron|(cryo|electron)[- ]microscopy (platform|unit|laboratory)|'
                 r'core facility|shared resource|structural biology core|imaging (core|facility|center))', re.I)


def norm_inst(seg):
    s = unicodedata.normalize('NFKD', seg).encode('ascii', 'ignore').decode().lower()
    s = re.sub(r'\b(the|of|and|for|&)\b', ' ', s)
    s = re.sub(r'[^a-z ]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()


inst_of = collections.defaultdict(set); fac_of = collections.defaultdict(set)
for d, aff in zip(PA.doi, PA.aff.fillna('')):
    for a in re.split(r';|\|', aff):
        segs = [x.strip() for x in a.split(',') if x.strip()]
        found = [norm_inst(x) for x in segs if KW.search(x) and not re.search(r'department|dept|division|faculty|school of|program', x, re.I)]
        if found:
            inst_of[d].add(found[-1])
        for m in FAC.finditer(a):
            fac_of[d].add(norm_inst(m.group(0)).replace('centre', 'center'))
S['inst'] = S.doi.map(lambda d: inst_of.get(d, set()) if isinstance(d, str) else set())
S['fac'] = S.doi.map(lambda d: fac_of.get(d, set()) if isinstance(d, str) else set())
pmc = PM.set_index('doi')
S['last_country'] = S.doi.map(pmc.last_country)
S['countries'] = S.doi.map(lambda d: set(str(pmc.countries.get(d, '')).split('|')) - {'', 'nan'} if isinstance(d, str) and d in pmc.index else set())
print('published EM studies with affiliations:', (S.inst.map(len) > 0).sum(), 'with a facility mention:', (S.fac.map(len) > 0).sum())

# ------------------------------------------------------------------ entity tests
T = []


def run(cls, unit, df, members, strata, min_n=3, label=None):
    """members: dict entity -> boolean mask"""
    for ent, mask in members.items():
        n = int(mask.sum())
        if n < min_n:
            continue
        o, ex, pu, p2, orr = strat_exact(mask, df.y.values, strata)
        labs = df.loc[mask, 'lab'].nunique() if 'lab' in df else np.nan
        elabs = df.loc[mask & (df.y == 1), 'lab'].nunique() if 'lab' in df else np.nan
        T.append(dict(cls=cls, unit=unit, entity=label(ent) if label else ent, n=n, n_err=o, expected=ex, OR_MH=orr,
                      p_upper=pu, p_two=p2, n_labs=labs, n_err_labs=elabs))


def masks_from_sets(series):
    inv = collections.defaultdict(list)
    for i, s in enumerate(series):
        for k in s:
            inv[k].append(i)
    out = {}
    for k, ii in inv.items():
        m = np.zeros(len(series), bool); m[ii] = True; out[k] = m
    return out


for unit, df in [('study', S), ('entry', E)]:
    st = df.ycat.values
    sets_any = df.akeys.map(lambda s: set(s.split('|')) - {''})
    sets_nl = df.akeys_nonlast.map(lambda s: set(s.split('|')) - {''})
    run('person (any position)', unit, df, masks_from_sets(sets_any), st, label=lambda k: pseudo.get(k, k))
    run('person (non-last author only)', unit, df, masks_from_sets(sets_nl), st, label=lambda k: pseudo.get(k, k))
    run('lab (last author)', unit, df, masks_from_sets(df.lab_pkey.map(lambda k: {k} if isinstance(k, str) else set())), st,
        label=lambda k: pseudo.get(k, k))
    run('co-author group (strong ties >=3 shared studies)', unit, df,
        masks_from_sets(sets_any.map(lambda s: {grp_of[k] for k in s if k in grp_of})), st)
    run('multi-lab cryo-EM hub person', unit, df, masks_from_sets(df.hubs), st, label=lambda k: 'H:' + hashlib.md5(k.encode()).hexdigest()[:6])
    run('any multi-lab hub on the study (aggregate)', unit, df, {'has_hub': df.has_hub.values}, st)
    if unit == 'study':
        P = df.published.values.astype(bool)
        dp = df[P].reset_index(drop=True)
        run('institution (any author affiliation)', unit, dp, masks_from_sets(dp.inst), dp.ycat.values)
        run('facility mention (any affiliation)', unit, dp, masks_from_sets(dp.fac), dp.ycat.values)
        run('any facility mention (aggregate)', unit, dp, {'any_facility': (dp.fac.map(len) > 0).values}, dp.ycat.values)
        run('country (last author)', unit, dp, masks_from_sets(dp.last_country.map(lambda c: {c} if isinstance(c, str) else set())), dp.ycat.values)
        run('country (any author)', unit, dp, masks_from_sets(dp.countries), dp.ycat.values)

T = pd.DataFrame(T)
T['q_within_class'] = np.nan
for (c, u), g in T.groupby(['cls', 'unit']):
    T.loc[g.index, 'q_within_class'] = bh(g.p_upper.values)
for u, g in T.groupby('unit'):
    T.loc[g.index, 'q_all_classes'] = bh(g.p_upper.values)
T = T.sort_values(['unit', 'p_upper'])
T.to_csv(f'{OUT}/tables/entity_enrichment.csv', index=False)

summ = T.groupby(['unit', 'cls']).agg(n_tested=('entity', 'size'), min_p=('p_upper', 'min'), min_q=('q_within_class', 'min'),
                                     n_q10=('q_within_class', lambda s: int((s < 0.1).sum())),
                                     n_p05=('p_upper', lambda s: int((s < 0.05).sum()))).reset_index()
# expected number of nominal p<0.05 under the null is < 0.05*n_tested for discrete tests
summ.to_csv(f'{OUT}/tables/entity_enrichment_summary.csv', index=False)
print(summ.to_string())
print('\nTop study-level hits:')
print(T[T.unit == 'study'].head(15)[['cls', 'entity', 'n', 'n_err', 'expected', 'OR_MH', 'p_upper', 'q_within_class', 'q_all_classes', 'n_labs', 'n_err_labs']].to_string())
print('\nTop entry-level hits:')
print(T[T.unit == 'entry'].head(10)[['cls', 'entity', 'n', 'n_err', 'expected', 'OR_MH', 'p_upper', 'q_within_class', 'q_all_classes', 'n_labs', 'n_err_labs']].to_string())
# aggregate rows for hubs/facilities
print('\nAggregate rows:')
print(T[T.cls.str.contains('aggregate')].to_string())
S[['study', 'has_hub']].assign(n_hubs=S.hubs.map(len)).to_csv(f'{CACHE}/study_hub_flags.csv', index=False)
pd.Series(sorted(hubs)).to_csv(f'{CACHE}/hubs_private.csv', index=False)

# entry-level hits: how many distinct studies do their error entries come from?
inv_e = collections.defaultdict(list)
for i, s in enumerate(E.akeys.map(lambda s: set(s.split('|')) - {''})):
    for k in s:
        inv_e[pseudo.get(k, k)].append(i)
H = T[(T.unit == 'entry') & (T.q_within_class < 0.1) & T.cls.str.startswith(('person', 'lab'))].copy()
H['n_err_studies'] = [E.iloc[inv_e[e]].loc[lambda d: d.y == 1].study.nunique() if e in inv_e else np.nan for e in H.entity]
H['n_studies'] = [E.iloc[inv_e[e]].study.nunique() if e in inv_e else np.nan for e in H.entity]
H[['cls', 'entity', 'n', 'n_err', 'expected', 'q_within_class', 'n_studies', 'n_err_studies']].to_csv(f'{OUT}/tables/entry_level_hits_by_study.csv', index=False)
print('\nentry-level person/lab hits (q<0.1): distinct error studies behind them')
print(H[['cls', 'entity', 'n', 'n_err', 'n_studies', 'n_err_studies']].drop_duplicates('entity').to_string())
