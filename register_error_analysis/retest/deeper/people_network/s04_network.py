#!/usr/bin/env python3
"""Co-authorship network among error studies vs year-matched random samples of clean studies (cryo-EM).

Nodes are studies (papers); entry-level variant uses entries with within-study links removed.
Edge variants (person = ORCID-merged strict name key from the PDB audit-author list):
  any        share >= 1 person
  cross_lab  share >= 1 person and the last-author labs differ
  cross_mid  share >= 1 person who is a non-last author in both studies, labs differ ("middle-author connector")
  *_noamb    same, ignoring name-only keys with a common (collision-prone) surname
  paper_*    same using Europe PMC paper author lists (published studies only)
Statistics of the induced subgraph: edges, share of nodes with >= 1 link, non-singleton components, largest component.
Null: 2,000 samples of clean studies matched 1:1 to the error studies on deposit half-year (primary), and on
half-year x author-count bin (secondary). One-sided p = share of null samples >= observed.
"""
import sys, re, unicodedata
sys.path.insert(0, '/home/user/SAbR/register_error_analysis/retest/deeper/people_network')
import numpy as np, pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from common import OUT, CACHE, load_studies, load_entries

rng = np.random.default_rng(11)
NSAMP = 2000
A = pd.read_csv(f'{CACHE}/audit_authors_keyed.csv.gz')
amb = set(A[A.common & ~A.pkey.str.startswith('o:')].pkey)


def incidence(sets):
    keys = sorted(set().union(*sets))
    kx = {k: i for i, k in enumerate(keys)}
    r, c = [], []
    for i, s in enumerate(sets):
        for k in s:
            r.append(i); c.append(kx[k])
    return csr_matrix((np.ones(len(r)), (r, c)), shape=(len(sets), len(keys)))


def adjacency(sets, labs=None, lab2=None):
    P = incidence(sets)
    M = (P @ P.T).toarray() > 0
    np.fill_diagonal(M, False)
    if labs is not None:
        la = np.asarray(labs, object); lb = np.asarray(lab2, object)
        diff = (la[:, None] != la[None, :]) & (lb[:, None] != lb[None, :])
        M &= diff
    return M


def stats(M, ix):
    sub = M[np.ix_(ix, ix)]
    e = sub.sum() / 2
    deg = sub.sum(1)
    nc, lab = connected_components(csr_matrix(sub), directed=False)
    sizes = np.bincount(lab)
    return dict(edges=e, frac_linked=(deg > 0).mean(), n_comp_nonsingleton=int((sizes > 1).sum()), largest=int(sizes.max()))


def matched_samples(df, err_ix, clean_ix, keycols):
    key = df[keycols].astype(str).agg('|'.join, axis=1).values
    pools = {}
    for i in clean_ix:
        pools.setdefault(key[i], []).append(i)
    need = pd.Series([key[i] for i in err_ix]).value_counts()
    # fall back to half-year only, then year, if a cell is short
    out = []
    for _ in range(NSAMP):
        samp = []
        for k, n in need.items():
            pool = pools.get(k, [])
            if len(pool) >= n:
                samp += list(rng.choice(pool, n, replace=False))
            else:
                samp += list(pool)
                yr = k.split('|')[0][:4]
                alt = [i for i in clean_ix if key[i].startswith(yr) and i not in pool]
                samp += list(rng.choice(alt, n - len(pool), replace=False))
        out.append(np.array(samp))
    return out


def run_network(df, label, variants):
    err_ix = np.where(df.y == 1)[0]; clean_ix = np.where(df.y == 0)[0]
    df = df.copy()
    df['authbin'] = pd.cut(df.n_auth, [-1, 3, 6, 10, 1000]).astype(str)
    nulls = {'half-year matched': matched_samples(df, err_ix, clean_ix, ['half']),
             'half-year x author-count matched': matched_samples(df, err_ix, clean_ix, ['half', 'authbin'])}
    rows = []
    for vname, M in variants.items():
        obs = stats(M, err_ix)
        for nname, samples in nulls.items():
            null = pd.DataFrame([stats(M, s) for s in samples])
            for st, v in obs.items():
                rows.append(dict(level=label, variant=vname, null=nname, stat=st, observed=v, null_mean=null[st].mean(),
                                 null_sd=null[st].std(), ratio=v / null[st].mean() if null[st].mean() > 0 else np.nan,
                                 p_upper=(1 + (null[st] >= v).sum()) / (1 + len(null)),
                                 p_lower=(1 + (null[st] <= v).sum()) / (1 + len(null)), n_nodes=len(err_ix)))
    return rows


S = load_studies('EM')
S['half'] = pd.to_datetime(S.dep0).dt.year.astype(str) + 'H' + np.where(pd.to_datetime(S.dep0).dt.month <= 6, '1', '2')
sets = S.akeys.map(lambda s: set(s.split('|')) - {''}).tolist()
sets_nl = S.akeys_nonlast.map(lambda s: set(s.split('|')) - {''}).tolist()
sets_na = [s - amb for s in sets]; sets_nl_na = [s - amb for s in sets_nl]
labs = S.lab.fillna('?').values; labp = S.lab_pkey.fillna('?').values
V = {'any': adjacency(sets), 'cross_lab': adjacency(sets, labs, labp), 'cross_mid': adjacency(sets_nl, labs, labp),
     'cross_lab_noamb': adjacency(sets_na, labs, labp), 'cross_mid_noamb': adjacency(sets_nl_na, labs, labp)}

# paper-author variant (Europe PMC author lists; published studies only)
PA = pd.read_csv(f'{OUT}/../paper_meta/paper_authors.csv.gz')


def nrm(s):
    return re.sub(r'[^a-z]', '', unicodedata.normalize('NFKD', str(s)).encode('ascii', 'ignore').decode().lower())


PA['k'] = np.where(PA.orcid.notna(), 'o:' + PA.orcid.astype(str), 'n:' + PA['last'].map(nrm) + '_' + PA.initials.fillna('').map(nrm))
PA['nonlast'] = PA.pos < PA.n - 1
pa_sets = PA.groupby('doi').k.apply(set).to_dict()
pa_nl = PA[PA.nonlast].groupby('doi').k.apply(set).to_dict()
SP_ = S[S.published & S.doi.isin(pa_sets.keys())].reset_index(drop=True)
ps = [pa_sets[d] for d in SP_.doi]; pn = [pa_nl.get(d, set()) for d in SP_.doi]
# paper last author key for lab difference
plast = PA[~PA.nonlast].set_index('doi').k
labs2 = SP_.lab.fillna('?').values; labp2 = SP_.doi.map(plast).fillna('?').values
V2 = {'paper_any': adjacency(ps), 'paper_cross_lab': adjacency(ps, labs2, labp2), 'paper_cross_mid': adjacency(pn, labs2, labp2)}

rows = run_network(S, 'study', V)
rows += run_network(SP_.assign(n_auth=[len(x) for x in ps]), 'study (published, paper authors)', V2)

# entry level, within-study links removed
E = load_entries('EM')
E['half'] = E.half
es = E.akeys.map(lambda s: set(s.split('|')) - {''}).tolist()
esn = E.akeys_nonlast.map(lambda s: set(s.split('|')) - {''}).tolist()
st = np.asarray(E.study.astype(str).tolist(), dtype=object)
same_study = st[:, None] == st[None, :]
VE = {'any (other study)': adjacency(es) & ~same_study,
      'cross_lab': adjacency(es, E.lab.fillna('?').values, E.lab_pkey.fillna('?').values),
      'cross_mid': adjacency(esn, E.lab.fillna('?').values, E.lab_pkey.fillna('?').values)}
rows += run_network(E, 'entry', VE)

R = pd.DataFrame(rows)
R.to_csv(f'{OUT}/tables/network_vs_matched_clean.csv', index=False)
pd.set_option('display.width', 250)
print(R[R.stat.isin(['edges', 'frac_linked', 'largest', 'n_comp_nonsingleton'])].round(3).to_string())

# describe the observed error-study components (cross-lab), aggregated
err_ix = np.where(S.y == 1)[0]
for vn in ['cross_lab', 'cross_mid']:
    sub = V[vn][np.ix_(err_ix, err_ix)]
    nc, lab = connected_components(csr_matrix(sub), directed=False)
    sizes = pd.Series(lab).value_counts()
    big = sizes[sizes > 1]
    print(f'\n{vn}: error studies in non-singleton components {big.sum()}/{len(err_ix)}; component sizes {big.tolist()}')
    for c in big.index[:5]:
        mem = S.iloc[err_ix[lab == c]]
        print(f'  component of {len(mem)}: {mem.lab.nunique()} labs, years {sorted(mem.year.unique().tolist())}')
