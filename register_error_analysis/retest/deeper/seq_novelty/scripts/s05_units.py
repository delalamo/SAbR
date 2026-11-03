"""Unit-level (paper x antibody V region) table with novelty features; error modes per unit; run-level windows."""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
OUT='/home/user/SAbR/register_error_analysis/retest/deeper/seq_novelty/tables'
ch=pd.read_csv(f'{OUT}/chain_novelty.csv.gz',low_memory=False)
runs=pd.read_csv(f'{SP}/deeper/runs_first_occurrence.csv')
runs=runs.merge(ch[['pdb','chain','unit']],on=['pdb','chain'])
ch['deposit']=pd.to_datetime(ch.deposit)
# unit = paper x V region; use the earliest-deposited chain of the unit for prior-structure features
ch=ch.sort_values(['unit','deposit','pdb','chain'])
g=ch.groupby('unit')
U=g.first().copy()
U['n_chains']=g.size(); U['any_EM']=g.method.apply(lambda x:(x=='EM').any()); U['all_EM']=g.method.apply(lambda x:(x=='EM').all())
U['err_first_u']=g.err_first.any(); U['err_any_u']=g.err_any.any(); U['res_med']=g.res.median()
U['sars2_u']=g.sars2.any(); U['vhh_u']=g.vhh.max()
U['method_u']=np.where(U.any_EM,'EM','X-ray')
U['dep_year']=U.deposit.dt.year
modes=runs.groupby('unit').mode.apply(lambda x:';'.join(sorted(set(x))))
U['modes_u']=modes.reindex(U.index).fillna('')
win=runs.groupby(['unit','mode']).agg(i0=('imgt0','min'),i1=('imgt1','max'),delta=('delta',lambda x:x.dropna().mode().iat[0] if x.notna().any() else np.nan),nruns=('pdb','size')).reset_index()
win.to_csv(f'{OUT}/unit_mode_windows.csv',index=False)
U['clean']=~U.err_any_u
U.reset_index().to_csv(f'{OUT}/units.csv.gz',index=False)
print(U.shape, pd.crosstab(U.method_u,[U.err_first_u,U.clean]))
for m in sorted(runs['mode'].unique()):
    print(m, (U.modes_u.str.contains(m)&U.err_first_u).sum(), ((U.modes_u.str.contains(m))&(U.method_u=='EM')).sum())
