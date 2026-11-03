# Build an entry-level and unit-level analysis frame for verifying the "no competitive pressure" finding.
# Inputs: investigator's focal entry features (competition metrics) + shared chains.csv (antibody/modelling covariates).
import numpy as np, pandas as pd
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
INV = '/home/user/SAbR/register_error_analysis/retest/deeper/antigen_competition/tables'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/antigen_competition_no_competitive_pressure_confounding'

A = pd.read_csv(f'{INV}/pdbwide_ab_entries_antigen.csv.gz')            # all PDB antibody entries (competition metrics)
e = pd.read_csv(f'{SP}/deeper/entries.csv')                               # all focal entries (incl. copied-only)
amy = set(pd.read_csv('/home/user/SAbR/register_error_analysis/retest/deeper/pdb_meta/tables/amyloid_fibril_chains.csv').pdb)
ch = pd.read_csv(f'{SP}/deeper/chains.csv.gz')
ch['atyp'] = ((ch.L1_atyp + ch.L2_atyp + ch.L3_atyp) > 0).astype(int)
ch['mouse'] = (ch.species == 'mouse').astype(int)
ch['af'] = (ch.irm == 'AlphaFold model').astype(int)
ch['expstart'] = (ch.irm == 'experimental (PDB) model').astype(int)
ch['rigid'] = (ch.fit == 'rigid body only').astype(int)
ch['lam'] = (ch.ctype == 'L').astype(int)
ch['cdr3_atyp'] = ch.L3_atyp
agg = ch.groupby('pdb').agg(any_vhh=('vhh', 'max'), any_mouse=('mouse', 'max'), min_vid=('v_id', 'min'),
                            mean_vid=('v_id', 'mean'), any_atyp=('atyp', 'max'), n_atyp=('atyp', 'sum'),
                            lqz_min=('lq_z', 'min'), lqz_mean=('lq_z', 'mean'), af=('af', 'max'), expstart=('expstart', 'max'),
                            rigid=('rigid', 'max'), any_lam=('lam', 'max'), hold_days=('hold_days', 'first'),
                            tier=('tier', 'first'), aux=('aux_reagent', 'min'), nV=('chain', 'size'),
                            role=('role', lambda s: s.mode().iat[0])).reset_index()
m = e.merge(A.drop(columns=['date']), on='pdb', how='left').merge(agg, on='pdb', how='left')
m['amyloid'] = m.pdb.isin(amy)
m = m[~m.amyloid].copy()
m['dep'] = pd.to_datetime(m.deposit)
m['dyear'] = m.dep.dt.year
m['ycat'] = m.dyear.clip(lower=2017).astype(str)
m['half'] = (m.dyear.clip(lower=2017).astype(str) + 'H' + np.where(m.dep.dt.month <= 6, '1', '2'))
m['quarter'] = m.dep.dt.to_period('Q').astype(str)
m['tnum'] = (m.dep - pd.Timestamp('2020-01-01')).dt.days / 365.25
m['period'] = pd.cut(m.dyear, [0, 2019, 2021, 2100], labels=['<=2019', '2020-21', '2022+']).astype(str)
m['paper'] = m.doi.fillna(m.last_author.fillna('') + '_' + m.dyear.astype(str))
m['lab2'] = m.last_author.fillna('NA')
m['sars2'] = m.sars2.astype(bool)
m['ab_status2'] = m.ab_status.replace({'no contact': 'other', 'no interface data': 'other'})
m['cls2'] = m.tgt_cls.replace({'viral other': 'enzyme'})
m['capenz'] = m.cls2.isin(['viral capsid', 'enzyme']).astype(int)
m['fidonly'] = (m.ab_status2 == 'fiducial-only').astype(int)
m['resbin'] = pd.cut(m.res, [0, 3.3, 3.6, 4.0, 9], labels=['<=3.3', '3.3-3.6', '3.6-4.0', '>4.0']).astype(str)
for c in ['labs_acc_1m', 'labs_acc_6m', 'labs_acc_12m', 'labs_fam_6m', 'labs_fam_12m', 'prev12_acc', 'prev12_allpdb', 'ents_acc_12m']:
    m['l_' + c] = np.log2(1 + m[c])
m['race_1m'] = (m.labs_acc_1m > 0).astype(float).where(m.labs_acc_1m.notna())
# outcome definitions
m['y_first'] = m.err_first.astype(int)                      # investigator's error definition
m['clean'] = (~m.err_any).astype(int)
m['keep_first'] = m.err_first | ~m.err_any                  # investigator's analysis set (drops copied/artifact-only)
m['y_any'] = m.err_any.astype(int)                          # any run incl. copied -- all entries kept
m.to_csv(f'{OUT}/entries_frame.csv.gz', index=False)
print('entries', len(m), 'EM', (m.method == 'EM').sum())
print('analysis set (investigator):', m.keep_first.sum(), 'err', m[m.keep_first].y_first.sum())

# unit level (antibody-in-paper), chain-type resolved
cu = ch[~ch.pdb.isin(amy)].merge(m[['pdb', 'ycat', 'half', 'period', 'tnum', 'labs_acc_12m', 'l_labs_acc_12m', 'l_labs_fam_12m', 'labs_fam_12m',
                                     'race_1m', 'ab_status2', 'capenz', 'fidonly', 'lab2', 'tgt_cls']], on='pdb', how='left')
cu['dep'] = pd.to_datetime(cu.deposit)
U = cu.sort_values('dep').groupby(['unit', 'ctype']).agg(
    err=('err_first', 'max'), anyrun=('err_any', 'max'), method=('method', 'first'), res=('res', 'median'), ycat=('ycat', 'first'),
    half=('half', 'first'), period=('period', 'first'), tnum=('tnum', 'first'), sars2=('sars2', 'max'), vhh=('vhh', 'max'), mouse=('mouse', 'max'),
    vid=('v_id', 'min'), atyp=('atyp', 'max'), lqz=('lq_z', 'mean'), af=('af', 'max'), rigid=('rigid', 'max'), labs12=('labs_acc_12m', 'median'),
    l_labs12=('l_labs_acc_12m', 'median'), l_fam12=('l_labs_fam_12m', 'median'), race=('race_1m', 'max'), status=('ab_status2', 'first'),
    capenz=('capenz', 'max'), fidonly=('fidonly', 'max'), lab=('lab2', 'first'), vseq=('vseq', 'first'), first_pdb=('pdb', 'first')).reset_index()
U['paper'] = U.unit.str.split('|').str[0]
U.to_csv(f'{OUT}/units_frame.csv.gz', index=False)
print('units', len(U), U.ctype.value_counts().to_dict())
