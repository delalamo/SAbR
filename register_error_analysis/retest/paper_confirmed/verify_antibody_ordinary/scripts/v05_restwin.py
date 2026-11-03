"""How many of the 37 'conkit-missed' rest_win units are in entries (or chains) that Table S1 does flag?"""
import pandas as pd
AO='/home/user/SAbR/register_error_analysis/retest/paper_confirmed/antibody_ordinary/tables'
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
T=pd.read_excel('/tmp/supp/epmc/m-11-00938-sup2.xlsx'); pc=T.PDB_CHAIN_ID.str.split(':',n=1,expand=True)
s1e=set(pc[0].str.lower()); s1c=set(zip(pc[0].str.lower(),pc[1]))
U=pd.read_csv(f'{AO}/units_frame.csv.gz'); CH=pd.read_csv(f'{SP}/deeper/chains.csv.gz',low_memory=False)
r=U[U.rest_win]
CH['in_s1_entry']=CH.pdb.isin(s1e); CH['in_s1_chain']=[k in s1c for k in zip(CH.pdb,CH.chain)]
g=CH[CH.unit.isin(r.unit)&CH.err_first.astype(bool)].groupby('unit')
x=pd.DataFrame({'any_err_chain_in_s1_entry':g.in_s1_entry.any(),'any_err_chain_flagged_by_s1':g.in_s1_chain.any(),'paper':g.paper.first()})
print('rest_win units',len(r),'| with an error chain in a Table S1 entry',int(x.any_err_chain_in_s1_entry.sum()),
      '| error chain itself listed in Table S1',int(x.any_err_chain_flagged_by_s1.sum()))
x=x.join(r.set_index('unit')[['f_bovine','f_rabbit','f_rare_cdr_len5','EM']])
print(x[x.any_err_chain_in_s1_entry].to_string())
print('truly unflagged (entry not in Table S1):',int((~x.any_err_chain_in_s1_entry).sum()),
      'bovine',int(x[~x.any_err_chain_in_s1_entry].f_bovine.sum()),'rabbit',int(x[~x.any_err_chain_in_s1_entry].f_rabbit.sum()),
      'rare len',int(x[~x.any_err_chain_in_s1_entry].f_rare_cdr_len5.sum()))
x.to_csv('/home/user/SAbR/register_error_analysis/retest/paper_confirmed/verify_antibody_ordinary/tables/v05_restwin_s1.csv')
