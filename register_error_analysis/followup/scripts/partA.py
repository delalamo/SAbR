import re, pandas as pd, numpy as np
A='/tmp/analysis'; O=f'{A}/out'
em=pd.read_pickle(f'{A}/em_all_roles.pkl')
em['year']=pd.to_datetime(em.deposit).dt.year
txt=(em.ab_desc.fillna('')+' '+em.title.fillna('')).str.lower()
def sub(r):
    if r.role!='Auxiliary': return None
    t=(r.ab_desc or '')+' '+(r.title or '')
    if re.search(r'nanobody[- ]?35|\bnb[- ]?35\b|scfv[- ]?16|\bfv[- ]?16|svfv16',t,re.I): return 'G-protein stabiliser (Nb35/scFv16)'
    if re.search(r'bril|bag2|nabfab|anti[- ]?fab|fab[- ]specific|fab[- ]nanobod|megabod|legobod|trimbod|alfa|anti-?gfp|synthetic|\bsab|sybod',t,re.I): return 'Engineered fiducial/synthetic binder (BRIL-Fab, NabFab, megabody, legobody, sybody, sAB)'
    if r.formats=='VHH/nanobody': return 'Other nanobody'
    return 'Other Fab / scFv'
em['aux_type']=em.apply(sub,axis=1)
mem=em.kw.str.contains('MEMBRANE|TRANSPORT|CHANNEL',case=False,na=False)|em.title.str.contains('receptor|channel|transporter|GPCR',case=False,na=False)
em['membrane_like']=mem
yr=em[(em.year>=2012)]
g=yr.groupby('year')
T=pd.DataFrame({'EM_entries':g.size(),'with_antibody':g.role.apply(lambda x:x.notna().sum()),
  'auxiliary':g.role.apply(lambda x:(x=='Auxiliary').sum()),'antibody_of_interest':g.role.apply(lambda x:(x=='Antibody of interest').sum())})
T['pct_EM_with_aux_antibody']=100*T.auxiliary/T.EM_entries
T['pct_EM_with_antibody_of_interest']=100*T.antibody_of_interest/T.EM_entries
gm=yr[yr.membrane_like].groupby('year')
T['membrane_like_EM_entries']=gm.size(); T['pct_membrane_like_with_aux']=100*gm.role.apply(lambda x:(x=='Auxiliary').sum())/gm.size()
aux=yr[yr.role=='Auxiliary']
T['aux_median_res']=aux.groupby('year').res.median(); T['nonab_median_res']=yr[yr.role.isna()].groupby('year').res.median()
T['aux_median_antigen_kDa']=aux.groupby('year').antigen_kda.median()
T=T.fillna(0); print(T.round(1).to_string()); T.round(2).to_csv(f'{O}/A_em_antibody_use_by_year.csv')
st=pd.crosstab(aux.year,aux.aux_type); print(st.to_string()); st.to_csv(f'{O}/A_aux_types_by_year.csv')
# size of target: aux vs interest vs no antibody (non-antibody mass)
print(yr.groupby(yr.role.fillna('No antibody')).antigen_kda.describe(percentiles=[.25,.5,.75]).round(0).to_string())
# Labs: how concentrated is auxiliary use
print('distinct last authors using aux antibodies:',aux.last_author.nunique(), 'entries',len(aux))
# validation sample of classifier for report
em.to_pickle(f'{A}/em_all_roles2.pkl')
