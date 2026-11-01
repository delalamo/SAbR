import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from scipy.stats import wilcoxon, binomtest
O='/tmp/tmpl/out'
M=pd.read_pickle(f'{O}/mapfit.pkl').dropna(subset=['seg_pct'])
R=pd.read_pickle(f'{O}/register_calls.pkl'); Fm=pd.read_pickle(f'{O}/family_summary.pkl')
M['reg']=np.where(M.S,'shifted (flagged) register','other register')
M['rb']=pd.cut(M.res,[0,2.5,3.0,3.5,4.0,10],labels=['≤2.5','2.5–3','3–3.5','3.5–4','>4']).astype(str)
print('== Segment fit percentile within its own chain (50 = typical residue; lower = worse fit)')
print(M.groupby(['metric','rb','reg']).seg_pct.agg(['size','median']).round(1).unstack().to_string())
# contested families: both registers with map values
fam=M.groupby(['family','S']).seg_pct.mean().unstack()
both=fam.dropna()
diff=both[True]-both[False]
print(f'\n== Contested families (both registers deposited, map values for both): {len(both)}')
print('   shifted register fits worse in %d (%.0f%%), better in %d; median difference %.1f percentile points; Wilcoxon p=%.2g'%(
    (diff<0).sum(),100*(diff<0).mean(),(diff>0).sum(),diff.median(),wilcoxon(diff).pvalue))
# same restricted to high-res (≤3 A) entries on both sides
h=M[M.res<=3.0].groupby(['family','S']).seg_pct.mean().unstack().dropna(); dh=h[True]-h[False]
print('   using only ≤3.0 Å entries: %d families; shifted worse in %.0f%%; median diff %.1f'%(len(h),100*(dh<0).mean(),dh.median()))
# classify families
cls=pd.Series('uncontested',index=Fm.family)
cls[cls.index.isin(both.index)]=np.where(diff.reindex(cls[cls.index.isin(both.index)].index)<-10,'contested: shifted fits worse',
                                     np.where(diff.reindex(cls[cls.index.isin(both.index)].index)>10,'contested: shifted fits better','contested: no clear difference'))
Fm['arb']=Fm.family.map(cls)
# uncontested: best-resolution shifted entry's fit
best=M[M.S].sort_values('res').groupby('family').first()
Fm=Fm.merge(best[['res','seg_pct','metric']].rename(columns={'res':'best_S_res','seg_pct':'best_S_pct'}),left_on='family',right_index=True,how='left')
u=Fm[Fm.arb=='uncontested']
u['ev']=np.select([(u.best_S_res<=2.5)&(u.best_S_pct>=40),(u.best_S_pct<20)],['well-fitting ≤2.5 Å structure in flagged register (flag likely false)','flagged segment fits poorly (error likely real)'],'undetermined')
Fm.loc[u.index,'arb']='uncontested: '+u.ev
print('\n== Family verdicts'); print(Fm.arb.value_counts().to_string())
Fm.to_pickle(f'{O}/family_summary_arb.pkl')
# propagation statistics restricted to families where the shifted register is likely a real error
real=Fm[Fm.arb.isin(['contested: shifted fits worse','uncontested: flagged segment fits poorly (error likely real)'])]
print(f'\n== Families where the flagged register is likely a real error: {len(real)}')
print('   shifted register in ≥2 papers: %d (%.0f%%); in ≥2 labs: %d (%.0f%%); still deposited 2023+: %d (%.0f%%)'%(
   (real.papers_S>=2).sum(),100*(real.papers_S>=2).mean(),(real.labs_S>=2).sum(),100*(real.labs_S>=2).mean(),(real.recent_S>0).sum(),100*(real.recent_S>0).mean()))
print('   later papers on the same protein: %d; carrying the error: %d (%.0f%%)'%(real.post_papers.sum(),real.post_papers_S.sum(),100*real.post_papers_S.sum()/real.post_papers.sum()))
print('   error present in very first structure: %.0f%%'%(100*(real.first_entry_status=='S').mean()))
c=real[real.arb.str.startswith('contested')]
print('   contested subset: %d families; later papers %d, carrying error %.0f%%'%(len(c),c.post_papers.sum(),100*c.post_papers_S.sum()/max(1,c.post_papers.sum())))
I=pd.read_pickle(f'{O}/inheritance.pkl'); Ir=I[I.family.isin(real.family)]
print('   later entries by recorded starting model:'); print(Ir.groupby('cites').agg(entries=('S','size'),pct_carrying_error=('S',lambda x:round(100*x.mean(),1))).to_string())
real.to_pickle(f'{O}/families_real.pkl')
