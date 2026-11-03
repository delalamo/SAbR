#!/usr/bin/env python3
"""Collect the lens's primary tests into one table with a lens-wide Benjamini-Hochberg q.
Entity scans (s03) keep their own within-class BH (thousands of entities); here they enter as the aggregate tests
(hub on study, facility mention).  Network and permutation statistics enter as two-sided p = 2 x min(upper, lower)."""
import sys
sys.path.insert(0, '/home/user/SAbR/register_error_analysis/retest/deeper/people_network')
import numpy as np, pandas as pd
from common import OUT, bh
T = f'{OUT}/tables'
rows = []
en = pd.read_csv(f'{T}/entity_enrichment.csv')
for _, r in en[en.cls.str.contains('aggregate')].iterrows():
    rows.append(dict(family='entity aggregate', test=f'{r.cls} [{r.unit}]', effect=f'obs {r.n_err} vs exp {r.expected:.1f}, MH OR {r.OR_MH:.2f}',
                     n=r.n, p=r.p_two))
nw = pd.read_csv(f'{T}/network_vs_matched_clean.csv')
nw = nw[(nw.null == 'half-year matched') & nw.variant.isin(['any', 'cross_lab', 'cross_mid', 'any (other study)', 'cross_mid_noamb'])
        & nw.level.isin(['study', 'entry'])]
for _, r in nw.iterrows():
    rows.append(dict(family='network vs year-matched clean', test=f'{r.level}: {r.variant}: {r.stat}',
                     effect=f'obs {r.observed:.2f} vs null {r.null_mean:.2f} (x{r.ratio:.2f})', n=r.n_nodes, p=min(1, 2 * min(r.p_upper, r.p_lower))))
ex = pd.read_csv(f'{T}/experience_tests.csv')
for _, r in ex[ex.unit == 'study'].iterrows():
    rows.append(dict(family='experience (study level)', test=r.exposure, effect=f'OR {r.OR:.2f} ({r.lo:.2f}-{r.hi:.2f}); err {r.err_value:.3g} vs clean {r.control_value:.3g}',
                     n=r.n, p=r.p))
lk = pd.read_csv(f'{T}/crosslab_linking_perm.csv')
for _, r in lk.iterrows():
    rows.append(dict(family='cross-lab person linking (permutation)', test=f'{r.stat} [{r["keys"]}]',
                     effect=f'obs {r.observed} vs null {r.null_mean:.1f} (x{r.ratio:.2f})', n=127, p=min(1, 2 * min(r.p_upper, r.p_lower))))
cp = pd.read_csv(f'{T}/carrier_persistence_tests.csv')
for _, r in cp.iterrows():
    rows.append(dict(family='carrier / persistence', test=r.test, effect=f'OR {r.OR:.2f} ({r.lo:.2f}-{r.hi:.2f})', n=r.n, p=r.p))
mc = pd.read_csv(f'{T}/mode_concordance_within_lab.csv').iloc[0]
rows.append(dict(family='carrier / persistence', test=mc.test, effect=f'obs {mc.observed} vs null {mc.null_mean:.1f} of {mc.pairs} pairs', n=mc.pairs, p=mc.p_upper))
nb = pd.read_csv(f'{T}/nanobody_supplier_tests.csv')
for _, r in nb.iterrows():
    rows.append(dict(family='post hoc: nanobody supplier (VUB/VIB co-authors)', test=r.test,
                     effect=f'OR {r.OR:.2f}; {r.err_exposed} vs {r.err_unexposed}' if pd.notna(r.OR) else f'{r.err_exposed} vs {r.err_unexposed}', n=r.n, p=r.p))
era = pd.read_csv(f'{T}/new_team_by_era.csv')
for _, r in era.iterrows():
    rows.append(dict(family='exploratory: new-team effect by era', test=r.stratum, effect=f'OR {r.OR:.2f} ({r.lo:.2f}-{r.hi:.2f})', n=r.n, p=r.p))
R = pd.DataFrame(rows)
R['q_lens'] = bh(R.p)
R.to_csv(f'{T}/all_tests_bh.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 90)
print(R.sort_values('p').head(30).round(4).to_string())
print(len(R), 'tests')

# pseudonymised study-level analysis frame for other lenses (no names; labs as L#### ids)
from common import CACHE
X = pd.read_csv(f'{CACHE}/study_experience.csv.gz')
C = pd.read_csv(f'{CACHE}/study_carrier_persistence.csv.gz')[['study', 'n_prior_x', 'exp_x', 'n_prior_own', 'exp_own', 'published', 'doi', 'lab_pkey']]
X = X.merge(C, on='study', how='left')
labs = {k: f'L{i:04d}' for i, k in enumerate(sorted(X.lab.dropna().unique()))}
X['lab_id'] = X.lab.map(labs)
X['study_id'] = np.where(X.doi.notna(), X.doi, 'unpublished_' + pd.factorize(X.study)[0].astype(str))
keep = ['study_id', 'lab_id', 'year', 'ycat', 'period', 'y', 'res', 'n_v', 'sars2', 'published', 'lab_prior_all', 'lab_prior_em', 'lab_prior_ab',
        'lab_prior_abem', 'first_em_lab', 'first_abem_lab', 'team_max_em_na', 'team_max_ab_na', 'team_max_abem_na', 'first_em_team_na',
        'first_abem_team_na', 'nonlast_max_abem_na', 'has_hub', 'n_prior_x', 'exp_x', 'n_prior_own', 'exp_own']
X[keep].rename(columns={'y': 'err_first_study'}).to_csv(f'{T}/study_frame_pseudonymised.csv.gz', index=False)
print('wrote study_frame_pseudonymised.csv.gz', len(X))
