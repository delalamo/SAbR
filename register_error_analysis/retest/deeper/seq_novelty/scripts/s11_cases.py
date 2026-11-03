"""Case lists behind the main positives."""
import pandas as pd
OUT='/home/user/SAbR/register_error_analysis/retest/deeper/seq_novelty/tables'
U=pd.read_csv(f'{OUT}/units.csv.gz',low_memory=False)
E=U[(U.method_u=='EM')&U.err_first_u].copy()
c=['pdb','chain','ctype','dep_year','last_author','modes_u','q_species_all','q_v_gene_hm','q_j_gene_hm','seq_first_imgt','seq_last_imgt','gprior_V','gprior_VJ','nn_any_V','nn_diff_V','n_prior_same_ab','rare_pos']
x=E[E.seq_nt_trunc|E.seq_fr4_trunc|(E.rareins_win_A>0)|(E.rareins_win_B>0)|(E.rareins_win_C>0)|(E.gprior_V==0)|(~E.q_species_all.isin(['human','mouse']))|E.modes_u.fillna('').str.contains('E_FR4')]
x[c].sort_values(['modes_u','pdb']).to_csv(f'{OUT}/cases_positive_features.csv',index=False)
print(len(x)); print(x.groupby('q_species_all').last_author.agg(lambda s: s.value_counts().head(3).to_dict()))
a=E[E.modes_u.fillna('').str.contains('A_FR1')&E.seq_nt_trunc]
print('A with SEQRES N-trunc:',len(a),'labs',a.last_author.nunique(),'papers',a.paper.nunique(), a.dep_year.value_counts().sort_index().to_dict())
