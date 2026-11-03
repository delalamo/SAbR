#!/usr/bin/env python3
"""Post-hoc follow-up of the one country-level signal (Belgium, any author; q = 0.31 in s03).
Hypothesis: a shared nanobody supplier (co-authors affiliated with VUB / VIB Brussels, where a large nanobody
discovery core is based) links error studies across labs.
  * within nanobody-containing cryo-EM studies: VUB/VIB co-authorship vs not (stratified exact test by year);
  * entity scan of institutions within nanobody studies (BH);
  * unit level (nanobody V region in a paper): CDR lengths, species, germline identity, error rate and modes;
  * is it the reagent or the people?  Auxiliary (reused) reagent vs subject nanobody; last author Belgian or not.
Treat as hypothesis-generating: selected after looking at the country scan.
"""
import sys, re, collections, unicodedata, warnings
sys.path.insert(0, '/home/user/SAbR/register_error_analysis/retest/deeper/people_network')
import numpy as np, pandas as pd
import statsmodels.formula.api as smf, statsmodels.api as sm
from scipy.stats import fisher_exact, mannwhitneyu
from common import OUT, SP, CACHE, load_studies, load_entries, strat_exact, bh
warnings.filterwarnings('ignore')
S = load_studies('EM'); E = load_entries('EM')
PA = pd.read_csv(f'{OUT}/../paper_meta/paper_authors.csv.gz')
PM = pd.read_csv(f'{OUT}/../paper_meta/paper_meta.csv', usecols=['doi', 'last_country']).set_index('doi').last_country
CH = pd.read_csv(f'{SP}/chains.csv.gz', usecols=['pdb', 'chain', 'vhh', 'role', 'aux_reagent', 'L1', 'L2', 'L3', 'species', 'v_gene',
                                                 'v_id', 'err_first', 'err_any', 'err_modes', 'unit', 'method', 'chain_desc'])
E['vhh'] = E.pdb.map(CH.groupby('pdb').vhh.max())
S['vhh'] = S.study.map(E.groupby('study').vhh.max()).fillna(0).astype(int)
VUBRE = r'Vrije Universiteit Brussel|VIB-VUB|VIB Center for Structural Biology|Nanobody Core|Structural Biology Brussels'
vub = set(PA[PA.aff.fillna('').str.contains(VUBRE, regex=True)].doi)
S['vub'] = S.doi.isin(vub).astype(int)
S['bel_last'] = S.doi.map(lambda d: PM.get(d) == 'Belgium' if isinstance(d, str) else False).astype(int)
N = S[S.vhh == 1].reset_index(drop=True)
out = []
o, ex, pu, p2, orr = strat_exact(N.vub.values == 1, N.y.values, N.ycat.values)
out.append(dict(test='nanobody studies: VUB/VIB co-author vs not (year-stratified exact)', n=len(N), n_exposed=int(N.vub.sum()),
                err_exposed=f'{o}/{int(N.vub.sum())}', err_unexposed=f'{int(N.y.sum()) - o}/{int(len(N) - N.vub.sum())}',
                expected=ex, OR=orr, p=p2))
NN = N[N.bel_last == 0]
o, ex, pu, p2, orr = strat_exact(NN.vub.values == 1, NN.y.values, NN.ycat.values)
out.append(dict(test='... last author not in Belgium (supplier-type collaborations)', n=len(NN), n_exposed=int(NN.vub.sum()),
                err_exposed=f'{o}/{int(NN.vub.sum())}', err_unexposed=f'{int(NN.y.sum()) - o}/{int(len(NN) - NN.vub.sum())}',
                expected=ex, OR=orr, p=p2))
Nn = S[S.vhh == 0]
o, ex, pu, p2, orr = strat_exact(Nn.vub.values == 1, Nn.y.values, Nn.ycat.values)
out.append(dict(test='non-nanobody studies: VUB/VIB co-author vs not', n=len(Nn), n_exposed=int(Nn.vub.sum()),
                err_exposed=f'{o}/{int(Nn.vub.sum())}', err_unexposed=f'{int(Nn.y.sum()) - o}/{int(len(Nn) - Nn.vub.sum())}',
                expected=ex, OR=orr, p=p2))
m = smf.glm('y ~ vub + C(ycat) + res_med + l2_nv', N, family=sm.families.Binomial()).fit()
out.append(dict(test='nanobody studies: logit adj. year, resolution, V chains', n=len(N), OR=np.exp(m.params['vub']), p=m.pvalues['vub']))
Sx = pd.read_csv(f'{CACHE}/study_experience.csv.gz')[['study', 'first_abem_team_na']]
N2 = N.merge(Sx, on='study', how='left')
m = smf.glm('y ~ vub + first_abem_team_na + C(ycat) + res_med + l2_nv', N2, family=sm.families.Binomial()).fit()
out.append(dict(test='... + new-to-antibody-cryo-EM team', n=len(N2), OR=np.exp(m.params['vub']), p=m.pvalues['vub']))

# institution scan within nanobody studies (same extraction as s03)
KW = re.compile(r'(universit|institut|cent(er|re)|college|hospital|school|laborator|academy|inc\b|ltd|gmbh|llc|corporation|'
                r'company|pharma|therapeutics|biosciences|foundation|facility|core|consortium|agency|council)', re.I)


def norm_inst(seg):
    s = unicodedata.normalize('NFKD', seg).encode('ascii', 'ignore').decode().lower()
    s = re.sub(r'\b(the|of|and|for|&)\b', ' ', s); s = re.sub(r'[^a-z ]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()


inst_of = collections.defaultdict(set)
for d, aff in zip(PA.doi, PA.aff.fillna('')):
    for a in re.split(r';|\|', aff):
        segs = [x.strip() for x in a.split(',') if x.strip()]
        f = [norm_inst(x) for x in segs if KW.search(x) and not re.search(r'department|dept|division|faculty|school of|program', x, re.I)]
        if f:
            inst_of[d].add(f[-1])
inv = collections.defaultdict(list)
for i, d in enumerate(N.doi):
    for k in (inst_of.get(d, set()) if isinstance(d, str) else set()):
        inv[k].append(i)
rows = []
for k, ii in inv.items():
    if len(ii) < 3:
        continue
    msk = np.zeros(len(N), bool); msk[ii] = True
    o, ex, pu, p2, orr = strat_exact(msk, N.y.values, N.ycat.values)
    rows.append(dict(institution=k, n=len(ii), n_err=o, expected=ex, OR_MH=orr, p_upper=pu))
I = pd.DataFrame(rows); I['q'] = bh(I.p_upper); I = I.sort_values('p_upper')
I.to_csv(f'{OUT}/tables/nanobody_institution_scan.csv', index=False)
print('institution scan within nanobody studies (top 6 of %d):' % len(I)); print(I.head(6).round(4).to_string())

# unit level: nanobody V regions
c = CH[(CH.method == 'EM') & (CH.vhh == 1)].copy()
c['study'] = c.pdb.map(dict(zip(E.pdb, E.study)))
c = c[c.study.isin(N.study)]
c['vub'] = c.study.isin(N[N.vub == 1].study)
u = c.sort_values('err_first', ascending=False).drop_duplicates('unit')
u = u[~(u.err_any & ~u.err_first)]   # drop copied/artifact-only units
print('\nnanobody units: VUB-co-authored vs other')
desc = u.groupby('vub').agg(n=('unit', 'size'), err=('err_first', 'mean'), n_err=('err_first', 'sum'), aux=('aux_reagent', 'mean'),
                            L2_med=('L2', 'median'), L3_med=('L3', 'median'), v_id=('v_id', 'median'),
                            llama=('species', lambda s: (s.astype(str).str.contains('lama|alpaca|llama', case=False)).mean()))
print(desc.round(3).to_string())
for col in ['L1', 'L2', 'L3', 'v_id']:
    a = u[u.vub][col].dropna(); b = u[~u.vub][col].dropna()
    out.append(dict(test=f'nanobody units: {col} VUB vs other (Mann-Whitney)', n=len(u), err_exposed=f'median {a.median():.2f}',
                    err_unexposed=f'median {b.median():.2f}', p=mannwhitneyu(a, b).pvalue))
print(pd.crosstab(u.vub, u.L2, normalize='index').round(3).to_string())
for aux in [False, True]:
    g = u[u.aux_reagent == aux]
    t = pd.crosstab(g.vub, g.err_first).reindex(index=[False, True], columns=[False, True]).fillna(0)
    print(f'aux_reagent={aux}:', t.values.tolist())
    out.append(dict(test=f'nanobody units (aux_reagent={aux}): error rate VUB vs other (Fisher)', n=len(g),
                    err_exposed=f'{int(t.loc[True, True])}/{int(t.loc[True].sum())}', err_unexposed=f'{int(t.loc[False, True])}/{int(t.loc[False].sum())}',
                    OR=fisher_exact(t.values)[0], p=fisher_exact(t.values)[1]))
modes = u[u.err_first].assign(m=u.err_modes.astype(str).str.split(r'[;|,]')).explode('m').reset_index(drop=True)
print('\nmodes among erroneous nanobody units:'); print(pd.crosstab(modes.m, modes.vub).to_string())
mb = u.assign(B=u.err_modes.astype(str).str.contains('B_CDR2FR3'))
t = pd.crosstab(mb.vub, mb.B).reindex(index=[False, True], columns=[False, True]).fillna(0)
out.append(dict(test='nanobody units: CDR2/FR3 (mode B) error, VUB vs other (Fisher)', n=len(mb), err_exposed=f'{int(t.loc[True, True])}/{int(t.loc[True].sum())}',
                err_unexposed=f'{int(t.loc[False, True])}/{int(t.loc[False].sum())}', OR=fisher_exact(t.values)[0], p=fisher_exact(t.values)[1]))
print('\nVUB error studies: distinct last-author labs', N[(N.vub == 1) & (N.y == 1)].lab.nunique(), 'years',
      sorted(N[(N.vub == 1) & (N.y == 1)].year.tolist()))
O = pd.DataFrame(out); O['q'] = bh(O.p)
O.to_csv(f'{OUT}/tables/nanobody_supplier_tests.csv', index=False)
pd.set_option('display.width', 250)
print(O.round(4).to_string())
