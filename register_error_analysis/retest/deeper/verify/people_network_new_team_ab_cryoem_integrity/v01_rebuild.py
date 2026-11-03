#!/usr/bin/env python3
"""Independent rebuild of the 'new team' exposure for the people_network finding (adversarial integrity check).

Differences from the investigator's pipeline (s02/s05), on purpose:
  * study-entry authors are taken from entries.idx itself (the same source as the history), so the study's own
    names and the history names are parsed identically (the investigator used RCSB GraphQL names for the study and
    entries.idx names for the history);
  * entries.idx author list is parsed by pairing "Surname, Initials" tokens (handles 'Jr.' etc.) instead of
    splitting on '.,';
  * several name keys: strict (surname+all initials), loose (surname+first initial), surname only (max collision);
  * three antibody-entry sets for 'prior antibody cryo-EM entry': ANARCI set (antigen_competition, as used by the
    investigator), a regex+molecule-name set built here from pdb_seqres, and their union;
  * study key rebuilt from entries.csv (DOI; unpublished: last author + 60-day gap, or each entry alone).
Outputs (cache): study frame with all exposure variants.
"""
import json, re, gzip, unicodedata, collections, pickle
import numpy as np, pandas as pd

SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad/deeper'
CV = f'{SP}/cache/verify_newteam'
PN = '/home/user/SAbR/register_error_analysis/retest/deeper/people_network'

COMMON = set('''wang li zhang liu chen yang huang zhao wu zhou xu sun ma zhu hu guo he lin gao luo zheng liang xie song
tang han feng deng cao peng zeng xiao tian dong pan yuan cai jiang yu du ye cheng wei su lu ding ren shen yao zhong jin cui
tan fan fu qian xue zou kong bai shi mao qin yin wan gu hou shao meng long duan lei hao kim lee park choi jung kang cho yoon
jang lim shin han oh seo kwon hwang ahn song nguyen tran le pham singh kumar patel sharma gupta sato suzuki takahashi tanaka
watanabe ito nakamura kobayashi yamamoto smith johnson williams brown jones miller davis garcia rodriguez martinez wilson
anderson taylor thomas moore white martin lopez gonzalez hernandez'''.split())
SUFFIX = re.compile(r'\b(jr|sr|ii|iii|iv)\b\.?', re.I)
INIT = re.compile(r"^(?:[A-Z][a-z]{0,2}\.[-\s]?)+$")


def asc(s):
    return unicodedata.normalize('NFKD', str(s)).encode('ascii', 'ignore').decode()


def parse_authors(s):
    """'Smith, J.R., Phillips Jr., G.N., XYZ Consortium' -> [(surname, initials), ...]"""
    toks = [t.strip() for t in asc(s).split(',') if t.strip()]
    out = []; i = 0
    while i < len(toks):
        if i + 1 < len(toks) and INIT.match(toks[i + 1]):
            out.append((toks[i], toks[i + 1])); i += 2
        else:
            out.append((toks[i], '')); i += 1
    return out


def key(sur, ini):
    s = re.sub(r'[^a-z]', '', SUFFIX.sub('', sur.lower()))
    ii = re.sub(r'[^a-z]', '', ini.lower())
    if not s:
        return None
    return dict(strict=f'{s}_{ii}', loose=f'{s}_{ii[:1]}', sur=s, common=s in COMMON)


# ------------------------------------------------------------- entries.idx
idx = {}
with open('/tmp/supp/pdbwide/entries.idx', encoding='ascii') as fh:
    for l in fh.read().splitlines()[2:]:
        q = l.split('\t')
        d = pd.to_datetime(q[2], format='%m/%d/%y')
        if d.year > 2026:
            d = d.replace(year=d.year - 100)
        au = [key(*a) for a in parse_authors(q[5])]
        au = [a for a in au if a]
        idx[q[0].lower()] = dict(dep=np.datetime64(d.date(), 'D'), em='ELECTRON MICROSCOPY' in q[7], au=au)
print('entries.idx', len(idx))

# ------------------------------------------------------------- antibody-entry sets
ab_anarci = set(json.load(open(f'{SP}/cache/antigen_competition/ab_entries_pdbwide.json')))
pat = re.compile(r'C.{6,22}W[VIFYLHWAS][RQKHEL][QKRHEL].{20,110}[WF][GA].G')
nm = re.compile(r'FAB|ANTIBOD|IMMUNOGLOBULIN|NANOBOD|VHH|SCFV|SYBOD|HEAVY CHAIN|LIGHT CHAIN|\bIGG|\bIGM|\bIGA|KAPPA|LAMBDA|'
                r'\bVH\b|\bVL\b|FV\b|MEGABOD', re.I)
ab_name = set(); ab_rx = set()
with gzip.open('/tmp/supp/pdbwide/pdb_seqres.txt.gz', 'rt') as f:
    cur = None; desc = ''
    for line in f:
        if line[0] == '>':
            h = line[1:].split()
            cur = h[0].split('_')[0] if 'mol:protein' in line else None
            desc = line.split('  ', 1)[1] if '  ' in line else ''
        elif cur:
            s = line.strip()
            if 60 <= len(s) <= 1500 and pat.search(s):
                ab_rx.add(cur)
                if nm.search(desc) and not re.search(r'T[- ]?CELL RECEPTOR|TCR', desc, re.I):
                    ab_name.add(cur)
ab_alt = ab_name  # regex AND antibody-like molecule name (no ANARCI)
print('antibody sets: anarci', len(ab_anarci), 'regex+name', len(ab_alt), 'overlap', len(ab_anarci & ab_alt),
      'union', len(ab_anarci | ab_alt))
pickle.dump(dict(ab_anarci=ab_anarci, ab_alt=ab_alt), open(f'{CV}/absets.pkl', 'wb'))

# ------------------------------------------------------------- history per key type
H = {}
for kt in ['strict', 'loose', 'sur']:
    h = collections.defaultdict(lambda: collections.defaultdict(list))
    for p, r in idx.items():
        tags = ['all'] + (['em'] if r['em'] else [])
        if r['em'] and p in ab_anarci: tags.append('abem')
        if r['em'] and p in ab_alt: tags.append('abem_alt')
        if r['em'] and (p in ab_alt or p in ab_anarci): tags.append('abem_union')
        if p in ab_anarci: tags.append('ab')
        for k in {a[kt] for a in r['au']}:
            for t in tags:
                h[k][t].append(r['dep'])
    for k in h:
        for t in h[k]:
            h[k][t] = np.sort(np.array(h[k][t]))
    H[kt] = h


def prior(kt, k, t, d):
    a = H[kt].get(k, {}).get(t)
    return 0 if a is None else int(np.searchsorted(a, d))  # strictly before d


# ------------------------------------------------------------- focal entries & studies
E = pd.read_csv(f'{SP}/entries.csv')
am = set(pd.read_csv(f'{PN}/../pdb_meta/tables/amyloid_fibril_chains.csv').pdb.str.lower())
E['grp'] = np.where(E.err_first, 'err', np.where(~E.err_any, 'clean', 'excl'))
E.loc[E.pdb.isin(am), 'grp'] = 'amyloid'
E['dep'] = pd.to_datetime(E.deposit)
E = E.sort_values(['dep', 'pdb']).reset_index(drop=True)
# integrity: entries.idx deposition date vs entries.csv deposit
E['idx_dep'] = E.pdb.map(lambda p: idx.get(p, {}).get('dep'))
print('focal entries missing from entries.idx:', E.idx_dep.isna().sum())
E['dep_mismatch_days'] = (pd.to_datetime(E.idx_dep) - E.dep).dt.days
print('deposit date mismatch (entries.idx vs entries.csv):', (E.dep_mismatch_days.abs() > 0).sum())
E['lastk'] = E.pdb.map(lambda p: idx[p]['au'][-1]['loose'] if p in idx and idx[p]['au'] else None)
E['study'] = np.where(E.doi.notna(), 'doi:' + E.doi.astype(str).str.lower(), None)
for lab, g in E[E.study.isna()].groupby(E.lastk.fillna('?')):
    cur = None; last_d = None
    for i, d in zip(g.index, g.dep):
        if cur is None or (d - last_d).days > 60:
            cur = f'unpub:{lab}:{d.date()}'
        E.loc[i, 'study'] = cur; last_d = d
E['study_single'] = np.where(E.doi.notna(), E.study, 'single:' + E.pdb)
C = pd.read_csv(f'{SP}/chains.csv.gz', usecols=['pdb', 'chain'])
E['n_v'] = E.pdb.map(C.groupby('pdb').size()).fillna(E.n_V_chains)
E.to_pickle(f'{CV}/entries_base.pkl')


def exposures(p, d):
    au = idx.get(p, {}).get('au', [])
    r = {'n_auth': len(au)}
    for kt in ['strict', 'loose', 'sur']:
        for t in ['abem', 'abem_alt', 'abem_union', 'em', 'ab']:
            v = [prior(kt, a[kt], t, d) for a in au]
            r[f'max_{t}_{kt}'] = max(v or [0])
            vn = [prior(kt, a[kt], t, d) for a in au if not a['common']]
            r[f'max_{t}_{kt}_na'] = max(vn or [0])
        r[f'last_abem_{kt}'] = prior(kt, au[-1][kt], 'abem', d) if au else 0
    r['n_noncommon'] = sum(not a['common'] for a in au)
    r['last_common'] = bool(au and au[-1]['common'])
    r['all_common'] = bool(au) and all(a['common'] for a in au)
    return r


rows = []
for skey in ['study', 'study_single']:
    E['dep0'] = E.groupby(skey).dep.transform('min')
    X = pd.DataFrame([exposures(p, np.datetime64(d.date(), 'D')) for p, d in zip(E.pdb, E.dep0)])
    EE = pd.concat([E.reset_index(drop=True), X], axis=1)
    EE['skey'] = skey
    rows.append(EE)
EE = pd.concat(rows, ignore_index=True)
EE.to_pickle(f'{CV}/entries_exposures.pkl')
print(EE.groupby(['skey', 'method', 'grp']).size().unstack())
