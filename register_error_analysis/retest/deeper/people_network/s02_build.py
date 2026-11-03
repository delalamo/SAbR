#!/usr/bin/env python3
"""Build the people/lab tables for the people_network lens.

Outputs (tables/):
  entries_people.csv.gz   one row per antibody entry: group (err/clean/excl), study key, period, res, n V chains,
                          author keys, experience measures (PDB-wide, before the study's first deposition)
  studies.csv.gz          one row per (study, method): error/clean/excl at the study level
Private (scratchpad cache, not in the repo): person key -> display name map.

Conventions
  * error entry  = err_first (first-occurrence run), AL amyloid fibrils removed (detector artifact)
  * clean entry  = no disagreement run of any kind (err_any False)
  * excluded     = runs that are all copied/artifact
  * study        = citation DOI; unpublished entries are grouped by last author + deposits <= 60 days apart
  * person key   = ORCID when recorded on the PDB audit-author record, or the strict name key (surname + all
                   initials) mapped to an ORCID when that name has exactly one ORCID in the antibody set;
                   otherwise the strict name key.  Loose key (surname + first initial) kept for sensitivity.
"""
import json, re, unicodedata, collections
import numpy as np, pandas as pd

SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad/deeper'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/people_network'
CACHE = f'{SP}/cache/people_network'

COMMON = set('''wang li zhang liu chen yang huang zhao wu zhou xu sun ma zhu hu guo he lin gao luo zheng liang xie song
tang han feng deng cao peng zeng xiao tian dong pan yuan cai jiang yu du ye cheng wei su lu ding ren shen yao zhong jin cui
tan fan fu qian xue zou kong bai shi mao qin yin wan gu hou shao meng long duan lei hao kim lee park choi jung kang cho yoon
jang lim shin han oh seo kwon hwang ahn song nguyen tran le pham singh kumar patel sharma gupta sato suzuki takahashi tanaka
watanabe ito nakamura kobayashi yamamoto smith johnson williams brown jones miller davis garcia rodriguez martinez wilson
anderson taylor thomas moore white martin lopez gonzalez hernandez'''.split())


def asc(s):
    return unicodedata.normalize('NFKD', str(s)).encode('ascii', 'ignore').decode().lower()


def keys(name):
    """'Smith, J.R.' -> ('smith_j', 'smith_jr', 'smith')"""
    n = asc(name)
    sur, _, ini = n.partition(',')
    sur = re.sub(r'[^a-z]', '', sur)
    ini = re.sub(r'[^a-z]', '', ini)
    if not sur:
        return None, None, None
    return f'{sur}_{ini[:1]}', f'{sur}_{ini}', sur


# ---------------------------------------------------------------- entries
E = pd.read_csv(f'{SP}/entries.csv')
C = pd.read_csv(f'{SP}/chains.csv.gz', usecols=['pdb', 'paper', 'sars2', 'ctype', 'vhh'])
am = set(pd.read_csv(f'{OUT}/../pdb_meta/tables/amyloid_fibril_chains.csv').pdb.str.lower())
E['amyloid'] = E.pdb.isin(am)
E['grp'] = np.where(E.err_first, 'err', np.where(~E.err_any, 'clean', 'excl'))
E.loc[E.amyloid, 'grp'] = 'amyloid'
paper = C.drop_duplicates('pdb').set_index('pdb').paper
E['paper'] = E.pdb.map(paper)
E['dep'] = pd.to_datetime(E.deposit)

# ---------------------------------------------------------------- audit authors (ORCID)
AA = {}
for l in open(f'{CACHE}/audit_author_orcid.jsonl'):
    r = json.loads(l)
    a = sorted(r['audit_author'] or [], key=lambda x: x.get('pdbx_ordinal') or 0)
    AA[r['rcsb_id'].lower()] = [(x['name'], x.get('identifier_ORCID')) for x in a]
rows = []
for p, lst in AA.items():
    for i, (nm, orc) in enumerate(lst):
        lk, sk, sur = keys(nm)
        if lk is None:
            continue
        rows.append(dict(pdb=p, pos=i, n=len(lst), name=nm, orcid=orc, lkey=lk, skey=sk, sur=sur))
A = pd.DataFrame(rows)
A['orcid'] = A.orcid.where(A.orcid.notna() & (A.orcid.astype(str).str.len() > 10))
one = A[A.orcid.notna()].groupby('skey').orcid.nunique()
o_map = A[A.orcid.notna() & A.skey.isin(one[one == 1].index)].groupby('skey').orcid.first()
A['pkey'] = np.where(A.orcid.notna(), 'o:' + A.orcid.astype(str),
                     np.where(A.skey.isin(o_map.index), 'o:' + A.skey.map(o_map).astype(str), 'n:' + A.skey))
A['is_last'] = A.pos == A.n - 1
A['common'] = A.sur.isin(COMMON)
A.to_csv(f'{CACHE}/audit_authors_keyed.csv.gz', index=False)
# private display-name map (kept out of the repo)
A.groupby('pkey').name.agg(lambda s: s.value_counts().index[0]).to_csv(f'{CACHE}/person_names_private.csv')
print('audit authors', len(A), 'entries', A.pdb.nunique(), 'person keys', A.pkey.nunique(),
      'with ORCID (direct or mapped)', A.pkey.str.startswith('o:').mean().round(3))

last = A[A.is_last].set_index('pdb')
E['lab'] = E.pdb.map(last.lkey)          # last author, loose key (as the other lenses)
E['lab_common'] = E.pdb.map(last.common).fillna(False)
E['n_auth'] = E.pdb.map(A.groupby('pdb').size()).fillna(0).astype(int)

# ---------------------------------------------------------------- study key
E = E.sort_values('dep').reset_index(drop=True)
E['study'] = E.paper.where(E.doi.notna())
unpub = E[E.study.isna()]
for lab, g in unpub.groupby(unpub.lab.fillna('?')):
    cur, last_d = None, None
    for i, d in zip(g.index, g.dep):
        if cur is None or (d - last_d).days > 60:
            cur = f'unpub:{lab}:{d.date()}'
        E.loc[i, 'study'] = cur
        last_d = d
E['published'] = E.doi.notna()

# ---------------------------------------------------------------- PDB-wide history (entries.idx)
abset = set(json.load(open(f'{SP}/cache/antigen_competition/ab_entries_pdbwide.json')))
hist = collections.defaultdict(lambda: {'all': [], 'em': [], 'ab': [], 'abem': []})
lab_hist = collections.defaultdict(lambda: {'all': [], 'em': [], 'ab': [], 'abem': []})
idx_rows = []
with open('/tmp/supp/pdbwide/entries.idx', encoding='utf-8', errors='replace') as fh:
    for l in fh.read().splitlines()[2:]:
        q = l.split('\t')
        if len(q) < 6:
            continue
        au = [a.strip() for a in re.split(r'(?<=\.),\s*', q[5]) if a.strip()]
        if not au:
            continue
        try:
            d = pd.Timestamp(pd.to_datetime(q[2], format='%m/%d/%y'))
        except Exception:
            continue
        if d.year > 2026:
            d = d.replace(year=d.year - 100)
        pid = q[0].lower()
        em = 'ELECTRON MICROSCOPY' in (q[7] if len(q) > 7 else '')
        ab = pid in abset
        ks = [keys(a) for a in au]
        ks = [k for k in ks if k[0]]
        if not ks:
            continue
        idx_rows.append((pid, d, em, ab, ks[-1][0], '|'.join(k[1] for k in ks)))
        d64 = np.datetime64(d, 'D')
        for k in {k[1] for k in ks}:
            h = hist[k]; h['all'].append(d64)
            if em: h['em'].append(d64)
            if ab: h['ab'].append(d64)
            if ab and em: h['abem'].append(d64)
        h = lab_hist[ks[-1][0]]; h['all'].append(d64)
        if em: h['em'].append(d64)
        if ab: h['ab'].append(d64)
        if ab and em: h['abem'].append(d64)
for H in (hist, lab_hist):
    for k in H:
        for t in H[k]:
            H[k][t] = np.sort(np.array(H[k][t], dtype='datetime64[D]'))
IDX = pd.DataFrame(idx_rows, columns=['pdb', 'dep', 'em', 'ab', 'lab', 'skeys'])
IDX.to_pickle(f'{CACHE}/idx_parsed.pkl')
print('entries.idx parsed', len(IDX), 'EM', IDX.em.sum(), 'antibody', IDX.ab.sum())


def prior(arr, d):
    return int(np.searchsorted(arr, np.datetime64(d, 'D'))) if arr is not None and len(arr) else 0


# experience before the study's first deposition (same-study entries excluded)
E['study_dep0'] = E.groupby('study').dep.transform('min')
akeys = A.groupby('pdb').skey.apply(list).to_dict()
lkeys_last = last.lkey.to_dict()
nonlast = A[~A.is_last].groupby('pdb').skey.apply(list).to_dict()
out = collections.defaultdict(list)
for p, d in zip(E.pdb, E.study_dep0):
    lab = lkeys_last.get(p)
    for t in ['all', 'em', 'ab', 'abem']:
        out[f'lab_prior_{t}'].append(prior(lab_hist[lab][t], d) if lab in lab_hist else 0)
        vals = [prior(hist[k][t], d) if k in hist else 0 for k in akeys.get(p, [])]
        out[f'team_max_{t}'].append(max(vals) if vals else 0)
        nl = [prior(hist[k][t], d) if k in hist else 0 for k in nonlast.get(p, [])]
        out[f'nonlast_max_{t}'].append(max(nl) if nl else 0)
    # union of distinct prior EM entries of any author is approximated by the max (above); count of authors with any EM
    out['team_n_em_people'].append(sum(1 for k in akeys.get(p, []) if k in hist and prior(hist[k]['em'], d) > 0))
for k, v in out.items():
    E[k] = v
E['first_em_lab'] = (E.lab_prior_em == 0).astype(int)
E['first_em_team'] = (E.team_max_em == 0).astype(int)
E['first_ab_lab'] = (E.lab_prior_ab == 0).astype(int)
E['first_abem_lab'] = (E.lab_prior_abem == 0).astype(int)

# ---------------------------------------------------------------- covariates
nv = C.groupby('pdb').size()
E['n_v'] = E.pdb.map(nv).fillna(1)
E['l2_nv'] = np.log2(E.n_v)
E['ycat'] = pd.cut(E.year, [0, 2019, 2020, 2021, 2022, 2023, 2024, 2100],
                   labels=['<=2019', '2020', '2021', '2022', '2023', '2024', '>=2025']).astype(str)
E['period'] = pd.cut(E.year, [0, 2019, 2021, 2100], labels=['<=2019', '2020-21', '2022+']).astype(str)
E['half'] = E.dep.dt.year.astype(str) + 'H' + np.where(E.dep.dt.month <= 6, '1', '2')
E['resbin'] = pd.cut(E.res, [0, 3.3, 3.7, 4.2, 10], labels=['<3.3', '3.3-3.7', '3.7-4.2', '>4.2']).astype(str)
E['sars2'] = E.sars2.fillna(False).astype(bool)
E['akeys'] = E.pdb.map(A.groupby('pdb').pkey.apply(lambda s: '|'.join(s)))
E['akeys_nonlast'] = E.pdb.map(A[~A.is_last].groupby('pdb').pkey.apply(lambda s: '|'.join(s)))
E['lab_pkey'] = E.pdb.map(last.pkey)
E.drop(columns=['title']).to_csv(f'{CACHE}/entries_people.csv.gz', index=False)

# ---------------------------------------------------------------- study level (per method)
S = []
for (st, m), g in E[E.grp != 'amyloid'].groupby(['study', 'method']):
    grp = 'err' if (g.grp == 'err').any() else ('clean' if (g.grp == 'clean').all() else 'excl')
    keys_ = set('|'.join(g.akeys.fillna('')).split('|')) - {''}
    keys_nl = set('|'.join(g.akeys_nonlast.fillna('')).split('|')) - {''}
    S.append(dict(study=st, method=m, grp=grp, n_entries=len(g), n_err_entries=int((g.grp == 'err').sum()),
                  dep0=g.dep.min(), year=g.dep.min().year, ycat=g.ycat.iloc[0], period=g.period.iloc[0],
                  half=g.half.iloc[0], res_med=g.res.median(), n_v=g.n_v.sum(), lab=g.lab.mode().iloc[0] if g.lab.notna().any() else None,
                  lab_pkey=g.lab_pkey.mode().iloc[0] if g.lab_pkey.notna().any() else None,
                  lab_common=bool(g.lab_common.iloc[0]), sars2=bool(g.sars2.any()), published=bool(g.published.iloc[0]),
                  doi=g.doi.iloc[0], n_auth=len(keys_), akeys='|'.join(sorted(keys_)), akeys_nonlast='|'.join(sorted(keys_nl)),
                  **{c: g[c].iloc[0] for c in E.columns if c.startswith(('lab_prior', 'team_max', 'nonlast_max', 'first_', 'team_n'))}))
S = pd.DataFrame(S)
S['l2_nv'] = np.log2(S.n_v)
S['resbin'] = pd.cut(S.res_med, [0, 3.3, 3.7, 4.2, 10], labels=['<3.3', '3.3-3.7', '3.7-4.2', '>4.2']).astype(str)
S['sizebin'] = pd.cut(S.n_v, [0, 2, 4, 8, 1e5], labels=['1-2', '3-4', '5-8', '>8']).astype(str)
S.to_csv(f'{CACHE}/studies.csv.gz', index=False)
print(pd.crosstab(S.method, S.grp))
print(pd.crosstab(E.method, E.grp))
