#!/usr/bin/env python3
"""Build a study-level verification frame for the 'new team' finding (people_network lens), cryo-EM.
Exposure: first_abem_team_na (no audit author, ignoring collision-prone surnames, with an earlier antibody cryo-EM
PDB entry) from the investigator's cache; plus all-name-key version (team_max_abem == 0).
Adds study-level covariates from chains.csv.gz, seq_novelty chain_novelty, pdb_meta chain_features_rcsb,
paper_meta (last-author country) and antigen_competition (antigen class).
"""
import numpy as np, pandas as pd
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad/deeper'
PN = f'{SP}/cache/people_network'
D = '/home/user/SAbR/register_error_analysis/retest/deeper'
OUT = f'{SP}/cache/verify_people_newteam_conf'

X = pd.read_csv(f'{PN}/study_experience.csv.gz')
St = pd.read_csv(f'{PN}/studies.csv.gz', parse_dates=['dep0'])
St = St[St.method == 'EM']
S = X[['study', 'first_abem_team_na', 'first_em_team_na', 'team_max_abem', 'team_max_abem_na', 'l_lab_prior_ab',
       'l_n_auth', 'y', 'lab', 'lab_common', 'has_hub']].merge(
    St[['study', 'dep0', 'res_med', 'n_v', 'n_entries', 'published', 'doi', 'sars2', 'n_auth', 'grp']], on='study', how='left')
S['new_all'] = (S.team_max_abem == 0).astype(int)
S['new'] = S.first_abem_team_na.astype(int)
S['dep_dec'] = S.dep0.dt.year + (S.dep0.dt.dayofyear - 1) / 365.25
S['dep_year'] = S.dep0.dt.year
S['dep_half'] = S.dep0.dt.year.astype(str) + 'H' + np.where(S.dep0.dt.month <= 6, '1', '2')
S['ycat'] = pd.cut(S.dep_year, [0, 2019, 2020, 2021, 2022, 2023, 2024, 2100],
                   labels=['<=2019', '2020', '2021', '2022', '2023', '2024', '>=2025']).astype(str)
S['ycat2'] = pd.cut(S.dep_year, [0, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2100],
                    labels=['<=2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024', '>=2025']).astype(str)
S['resbin'] = pd.cut(S.res_med, [0, 3.2, 3.4, 3.6, 3.9, 4.3, 10], labels=['<3.2', '3.2-3.4', '3.4-3.6', '3.6-3.9', '3.9-4.3', '>4.3']).astype(str)
S['l2_nv'] = np.log2(S.n_v)
S['l2_ne'] = np.log2(S.n_entries)

# entry -> study map (cryo-EM)
E = pd.read_csv(f'{PN}/entries_people.csv.gz', usecols=['pdb', 'method', 'study', 'grp', 'lab', 'year', 'dep'])
E = E[(E.method == 'EM') & E.study.isin(S.study)]
st_of = dict(zip(E.pdb, E.study))

C = pd.read_csv(f'{SP}/chains.csv.gz', usecols=['pdb', 'chain', 'ctype', 'species', 'v_id', 'L1_atyp', 'L2_atyp', 'L3_atyp',
                                                 'vhh', 'role', 'aux_reagent', 'novel_target', 'tier', 'irm', 'fit', 'lq_z',
                                                 'err_first', 'err_any', 'unit', 'journal'])
C = C[C.pdb.isin(st_of)].copy()
C['study'] = C.pdb.map(st_of)
N = pd.read_csv(f'{D}/seq_novelty/tables/chain_novelty.csv.gz', usecols=['pdb', 'chain', 'n_prior_same_ab', 'exact_prior', 'n_prior_ge90', 'nn_any_V', 'gprior_V'])
C = C.merge(N, on=['pdb', 'chain'], how='left')
R = pd.read_csv(f'{D}/pdb_meta/tables/chain_features_rcsb.csv.gz', usecols=['pdb', 'chain', 'chain_q', 'chain_unobs_cov', 'chain_inc'])
C = C.merge(R, on=['pdb', 'chain'], how='left')
C['novel_ab'] = (C.n_prior_same_ab.fillna(0) == 0)
C['atyp'] = (C[['L1_atyp', 'L2_atyp']].fillna(0).astype(int).sum(axis=1) > 0)  # L3_atyp flags ~60% of chains, so CDR1/2 only
C['af'] = C.irm.astype(str).str.contains('AlphaFold|in-silico', case=False)
C['exp_start'] = C.irm.astype(str).str.contains('experimental', case=False)
C['rigid'] = C.fit.astype(str).eq('rigid body only')
C['flex'] = C.fit.astype(str).eq('flexible fit')
C['subject'] = C.role.eq('Antibody of interest')
g = C.groupby('study')
Z = pd.DataFrame({
    'any_K': g.ctype.apply(lambda s: (s == 'K').any()), 'any_L': g.ctype.apply(lambda s: (s == 'L').any()),
    'vhh_any': g.vhh.max(), 'vhh_only': g.vhh.min(),
    'mouse_any': g.species.apply(lambda s: (s == 'mouse').any()),
    'min_vid': g.v_id.min(), 'atyp_any': g.atyp.max(),
    'aux_any': g.aux_reagent.max(), 'subject_any': g.subject.max(),
    'novel_any': g.novel_ab.max(), 'novel_all': g.novel_ab.min(),
    'novel_target': g.novel_target.max(),
    'tier': g.tier.first(),
    'af_any': g.af.max(), 'exp_start_any': g.exp_start.max(), 'rigid_any': g.rigid.max(), 'flex_any': g.flex.max(),
    'irm_none': g.irm.apply(lambda s: (s.astype(str) == 'none recorded').all()),
    'lq_z_med': g.lq_z.median(), 'lq_z_min': g.lq_z.min(),
    'q_med': g.chain_q.median(), 'unobs_max': g.chain_unobs_cov.max(), 'unobs_med': g.chain_unobs_cov.median(),
    'nn_V_max': g.nn_any_V.max(), 'n_chains': g.size(), 'n_units': g.unit.nunique(),
    'n_err_units': g.apply(lambda d: d.loc[d.err_first == True, 'unit'].nunique()),
})
for c in ['any_K', 'any_L', 'vhh_any', 'vhh_only', 'mouse_any', 'atyp_any', 'aux_any', 'subject_any', 'novel_any', 'novel_all',
          'af_any', 'exp_start_any', 'rigid_any', 'flex_any', 'irm_none']:
    Z[c] = Z[c].fillna(False).astype(int)
Z['novel_target'] = Z.novel_target.fillna(0)
S = S.merge(Z, left_on='study', right_index=True, how='left')
# antigen class
F = pd.read_csv(f'{D}/antigen_competition/tables/focal_entries_antigen_features.csv.gz', usecols=['pdb', 'tgt_cls', 'first_fam'])
F['study'] = F.pdb.map(st_of)
F = F.dropna(subset=['study'])
fa = F.groupby('study').agg(capsid=('tgt_cls', lambda s: (s == 'viral capsid').any()),
                            enzyme=('tgt_cls', lambda s: s.isin(['enzyme', 'viral other']).any()),
                            first_fam=('first_fam', 'max'), tgt_cls=('tgt_cls', lambda s: s.mode().iloc[0] if s.notna().any() else 'unknown'))
S = S.merge(fa, left_on='study', right_index=True, how='left')
for c in ['capsid', 'enzyme', 'first_fam']:
    S[c] = S[c].fillna(0).astype(int)
S['tgt_cls'] = S.tgt_cls.fillna('unknown')
# country
PM = pd.read_csv(f'{D}/paper_meta/paper_meta.csv', usecols=['doi', 'last_country', 'n_authors']).drop_duplicates('doi').set_index('doi')
S['country'] = S.doi.map(PM.last_country).fillna('unknown')
S['china'] = (S.country == 'China').astype(int)
S['usa'] = (S.country == 'USA').astype(int)
S['ctry3'] = np.where(S.china == 1, 'China', np.where(S.usa == 1, 'USA', np.where(S.country == 'unknown', 'unknown', 'other')))
S['sars2'] = S.sars2.astype(int)
S['published'] = S.published.astype(int)
S['lab_key'] = S.lab.fillna('?' + S.study)
S.to_csv(f'{OUT}/study_frame.csv.gz', index=False)
print(len(S), S.y.sum(), S.new.sum())
print(S.groupby('new').y.agg(['size', 'sum', 'mean']))
print(S.isna().sum()[S.isna().sum() > 0])
