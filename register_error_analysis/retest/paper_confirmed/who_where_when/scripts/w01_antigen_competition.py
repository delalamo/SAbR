"""Same-antigen competition, SARS-CoV-2, niche antigens, capsid/enzyme targets: paper-confirmed cases vs window controls.
Reuses the per-entry frame of verify/antigen_competition_no_competitive_pressure_confounding (entries_frame.csv.gz:
antigen class, other labs on the same antigen +-1/6/12 months, antibody covariates) and, at paper level, the PM1
verification table (paper_table_verify.csv, family_competition.csv)."""
import sys; sys.path.insert(0, '.')
from ww_common import *

E = entry_flags()
VF = f'{D}/verify/antigen_competition_no_competitive_pressure_confounding/entries_frame.csv.gz'
m = pd.read_csv(VF).drop(columns=['clean']).merge(E[['pdb', 'case', 'case_f', 'case_i', 'ctrl', 'gen', 'rest', 'rest_w', 'win_all', 'clean', 'win', 's1']], on='pdb', how='inner')
m = add_yc(m, 'dyear')
m['cap'] = (m.cls2 == 'viral capsid').astype(int); m['enz'] = (m.cls2 == 'enzyme').astype(int)
m['sars2'] = m.sars2.astype(int); m['zero12'] = (m.labs_acc_12m == 0).astype(float)
m['first_fam_i'] = m.first_fam.astype(float)
base = m[m.ab_status2.isin(['bound', 'fiducial-only']) & m.labs_acc_12m.notna()].copy()
AB = 'any_vhh + any_mouse + min_vid + any_atyp + any_lam'
print('entries in analysis set by method:', base.groupby('method')[['case', 'case_i', 'ctrl', 'rest_w', 'rest', 'clean']].sum().to_string())
print('case entries dropped (unbound/other or no antigen):', sorted(set(E[E.case].pdb) - set(base[base.case].pdb)))

FULL = {'l_labs_acc_12m': 'EM 0.94 (0.86-1.04) yr+res; pre-treatment adj 1.07 (0.98-1.18) [verify comp-conf]',
        'l_labs_acc_6m': 'EM pre-treatment adj 1.09 p=0.09 [verify comp-conf]',
        'l_labs_acc_1m': 'EM pre-treatment adj 1.16 (1.01-1.34) p=0.03 [verify comp-conf]',
        'l_labs_fam_12m': 'EM 0.895 p=0.02 yr+res; pre-treatment adj 1.03 [verify comp-conf]',
        'race_1m': 'EM 0.90 p=0.67; pre-treatment adj 1.50 (0.98-2.30) p=0.06 [verify comp-conf]',
        'sars2': 'EM 6.7% vs 6.4%, OR 0.91 p=0.74 [antigen_competition]; pre-treatment adj 1.72 (0.99-2.97) [verify]',
        'cap': 'EM entry OR 4.7 (2.5-8.8) -> ~1.0 after artifacts/2 groups removed [README 4.3]',
        'enz': 'EM entry OR 3.8 (1.8-8.0); EC definition 1.4-1.9 n.s. [README 4.3]',
        'capenz': 'capsid+enzyme combined (lens adjustment term)',
        'first_fam_i': 'EM first antibody structure of family OR 2.2 q=0.014 -> 1.3-1.4 n.s. adjusted [antigen_competition]',
        'fidonly': 'EM fiducial-only entries 1.0% errors, OR 0.17 [antigen_competition]'}
for meth in ['EM', 'X-ray']:
    d0 = base[base.method == meth]
    for x in ['l_labs_acc_1m', 'l_labs_acc_6m', 'l_labs_acc_12m', 'l_labs_fam_12m', 'race_1m', 'sars2', 'cap', 'enz', 'capenz', 'fidonly', 'first_fam_i']:
        record(f'competition/antigen: {x}', 'entry', meth, x, '{YC}+res, paper-clustered', run_groups(d0, x, '{YC} + res', 'paper'), FULL.get(x, ''))
    for x in ['l_labs_acc_1m', 'l_labs_acc_6m', 'l_labs_acc_12m', 'l_labs_fam_12m', 'race_1m', 'sars2']:
        record(f'competition/antigen: {x} (pre-treatment adj)', 'entry', meth, x, '{YC}+res+capenz+fidonly+antibody feats',
               run_groups(d0, x, '{YC} + res + capenz + fidonly + ' + AB, 'paper'), FULL.get(x, ''))

# ---- one per paper: aggregate entries to paper (EM)
print('\n== paper level (one row per paper), EM')
d0 = base[base.method == 'EM']
P = d0.groupby('paper').agg(**{k: (k, 'max') for k in ['l_labs_acc_1m', 'l_labs_acc_6m', 'l_labs_acc_12m', 'l_labs_fam_12m', 'race_1m', 'sars2', 'cap', 'enz', 'capenz', 'fidonly', 'first_fam_i', 'any_vhh', 'any_mouse', 'any_atyp', 'any_lam']},
                            min_vid=('min_vid', 'min'), res=('res', 'median'), dyear=('dyear', 'min'), n=('pdb', 'size'), lab=('lab2', 'first')).reset_index()
U = unit_flags(E, d0[['pdb', 'paper']], 'paper')
P = add_yc(P.merge(U, on='paper'), 'dyear'); P['ln'] = np.log2(P.n)
for x in ['l_labs_acc_1m', 'l_labs_acc_12m', 'l_labs_fam_12m', 'race_1m', 'sars2', 'cap', 'enz', 'first_fam_i']:
    record(f'competition/antigen: {x}', 'paper', 'EM', x, '{YC}+res+log2 entries, lab-clustered', run_groups(P, x, '{YC} + res + ln', 'lab'), FULL.get(x, ''))

# first antibody structure of the family, adjusted for capsid/enzyme and lab antibody experience (as antigen_competition s09)
labx = pd.read_csv(f'{D}/antigen_competition/tables/lab_experience.csv')
d2 = base[base.method == 'EM'].merge(labx, on='pdb', how='left'); d2['lexp'] = np.log2(1 + d2.lab_prior_ab)
record('competition/antigen: first_fam_i (+capenz +lab experience)', 'entry', 'EM', 'first_fam_i', '{YC}+res+capenz+lexp', run_groups(d2, 'first_fam_i', '{YC} + res + capenz + lexp', 'paper'),
       FULL['first_fam_i'])

# ---- niche antigen in 2020-21 (antigen_competition s15): paper level, non-SARS2 EM, 0 other labs +-12 m
print('\n== niche antigen 2020-21 (paper level, non-SARS-CoV-2 EM); full set OR 8.3 (2.2-31) adj capsid/enzyme + lab experience')
lab = pd.read_csv(f'{D}/antigen_competition/tables/lab_experience.csv')
d1 = base[(base.method == 'EM') & (base.sars2 == 0)].merge(lab, on='pdb', how='left')
d1 = d1[d1.dyear.isin([2020, 2021])]
P2 = d1.groupby('paper').agg(zero=('zero12', 'max'), capenz=('capenz', 'max'), res=('res', 'median'), dyear=('dyear', 'min'), n=('pdb', 'size'),
                              lexp=('lab_prior_ab', lambda s: np.log2(1 + s.min())), lab=('lab2', 'first')).reset_index()
P2 = add_yc(P2.merge(unit_flags(E, d1[['pdb', 'paper']], 'paper'), on='paper'), 'dyear'); P2['ln'] = np.log2(P2.n)
record('niche antigen 2020-21 (0 other labs +-12m), non-SARS2', 'paper', 'EM', 'zero', 'yr+res+ln+capenz+lab exp',
       run_groups(P2, 'zero', 'C(yc3) + res + ln + capenz + lexp', None), 'paper-level OR 8.3 (2.2-31), 19/89 vs 4/100 [antigen_competition s15]')
record('niche antigen 2020-21, crude (Fisher)', 'paper', 'EM', 'zero', 'MH by year', run_groups(P2, 'zero', None, None, kind='mh'), '21% vs 4%')
for g in ['case', 'rest_w']:
    t = P2[P2[g] | (P2.ctrl & ~P2.gen & ~P2.case_i)]
    print(f'  {g}: niche papers with a {g} error {int(t[t.zero == 1][g].sum())}/{int((t.zero == 1).sum())}, contested {int(t[t.zero == 0][g].sum())}/{int((t.zero == 0).sum())}')

# ---- paper-level 'fewer competitors' (paper_meta PM1): any other-lab DOI with same UniProt +-180 d; family 30% cluster
print('\n== paper_meta fewer competitors (PM1): any competitor +-180 d (accession), family 30% clusters')
PV = pd.read_csv(f'{D}/verify/paper_meta_PM1_less_competition_confounding/paper_table_verify.csv', low_memory=False)
FC = pd.read_csv(f'{D}/verify/paper_meta_PM1_less_competition_confounding/family_competition.csv')[['pkey', 'fam30_n']]
PV = PV.merge(FC, on='pkey', how='left')
PV['any_comp'] = np.where(PV.comp_dep180.notna(), (PV.comp_dep180 > 0).astype(float), np.nan)
PV['any_fam30'] = np.where(PV.fam30_n.notna(), (PV.fam30_n > 0).astype(float), np.nan)
PV['l_comp'] = np.log2(1 + PV.comp_dep180); PV['lab'] = PV.last_author
km = PV[['pkey', 'pdbs']].assign(pdb=PV.pdbs.str.split(';')).explode('pdb')[['pkey', 'pdb']]
PV = add_yc(PV.merge(unit_flags(E, km, 'pkey'), on='pkey'), 'dep_year')
for x, ref in [('any_comp', 'OR 0.47 (0.32-0.70) -> class-adj 0.60 p=0.057 [PM1]'), ('any_fam30', 'family 30%: base 0.58, parsimonious 0.75 p=0.33 [PM1 verify]'),
               ('l_comp', 'log2 count 0.895/doubling; class-adj 0.86 p=0.044 [PM1 verify]')]:
    record(f'fewer competitors (paper_meta): {x}', 'paper', 'EM', x, '{YC}+l2_units+res_median, lab-clustered',
           run_groups(PV[PV.method == 'EM'], x, '{YC} + l2_units + res_median', 'lab'), ref)
    record(f'fewer competitors (paper_meta): {x} +class', 'paper', 'EM', x, '+cap+enz+fid_only+sars2',
           run_groups(PV[PV.method == 'EM'], x, '{YC} + l2_units + res_median + cap + enz + fid_only + sars2', 'lab'), ref)
    record(f'fewer competitors (paper_meta): {x}', 'paper', 'X-ray', x, '{YC}+l2_units+res_median', run_groups(PV[PV.method == 'X-ray'], x, '{YC} + l2_units + res_median', 'lab'),
           'X-ray OR 1.26 (0.55-2.90) [PM1 verify]')

save_rows(f'{T}/w01_antigen_competition_tests.csv')
base[['pdb', 'method', 'dyear', 'paper', 'lab2', 'cls2', 'sars2', 'labs_acc_1m', 'labs_acc_12m', 'labs_fam_12m', 'race_1m', 'first_fam', 'case', 'case_i', 'ctrl', 'rest', 'rest_w', 'gen', 'clean']].rename(
    columns={'lab2': 'lab_key'}).drop(columns=['lab_key']).to_csv(f'{T}/w01_entry_frame.csv.gz', index=False)
