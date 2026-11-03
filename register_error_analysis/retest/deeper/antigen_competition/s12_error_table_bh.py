# Per-error-paper antigen/competition table, and one Benjamini-Hochberg correction over every test in this lens
import numpy as np, pandas as pd
from statsmodels.stats.multitest import multipletests
OUT='/home/user/SAbR/register_error_analysis/retest/deeper/antigen_competition'
m=pd.read_csv(f'{OUT}/tables/focal_entries_antigen_features.csv.gz')
lx=pd.read_csv(f'{OUT}/tables/lab_experience.csv'); m=m.merge(lx,on='pdb',how='left')
E=m[m.y==1]
T=E.groupby('paper').agg(method=('method','first'),first_deposit=('deposit','min'),last_author=('last_author','first'),n_error_entries=('pdb','size'),
    pdbs=('pdb',lambda s:' '.join(sorted(s))),modes=('modes',lambda s:';'.join(sorted(set(';'.join(s.dropna()).split(';'))-{''}))),
    antigen=('tgt_name',lambda s:s.fillna('').mode().iat[0]),antigen_desc=('tgt_desc','first'),target_class=('tgt_cls','first'),organism=('tgt_org','first'),
    binding=('ab_status','first'),sars2=('sars2','max'),other_labs_same_antigen_6m=('labs_acc_6m','median'),other_labs_same_antigen_12m=('labs_acc_12m','median'),
    other_labs_same_family_12m=('labs_fam_12m','median'),first_ab_structure_of_antigen=('first_acc','max'),first_ab_structure_of_family=('first_fam','max'),
    ab_entries_same_antigen_prev12m=('prev12_acc','median'),all_pdb_entries_antigen_prev12m=('prev12_allpdb','median'),lab_prior_antibody_entries=('lab_prior_ab','min')).reset_index().sort_values('first_deposit')
T.to_csv(f'{OUT}/tables/error_papers_antigen_competition.csv',index=False)
print('error papers',len(T)); print(T.target_class.value_counts().to_string())
print('error papers with zero other labs on the same antigen within +-12 months:',(T.other_labs_same_antigen_12m==0).sum(),'| first antibody structure of antigen:',T.first_ab_structure_of_antigen.sum())
frames=[]
for f in ['all_tests_bh.csv','followup_tests.csv','lab_experience_tests.csv','extra_checks.csv','novel_v_checks.csv','niche_by_period.csv']:
    d=pd.read_csv(f'{OUT}/tables/{f}'); d['source']=f; frames.append(d)
A=pd.concat(frames,ignore_index=True)
A=A[A.p.notna()].drop(columns=[c for c in ['q'] if c in A.columns])
A['q_lens']=multipletests(A.p,method='fdr_bh')[1]
A.sort_values('p').to_csv(f'{OUT}/tables/all_tests_bh_lens.csv',index=False)
print('total tests',len(A),'q<0.05:',(A.q_lens<0.05).sum())
