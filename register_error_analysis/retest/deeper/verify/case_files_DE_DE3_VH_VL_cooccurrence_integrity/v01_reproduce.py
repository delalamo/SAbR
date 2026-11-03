"""Independent reproduction of DE3 (VH D/E slip -> VL error co-occurrence), built from shared data directly."""
import pandas as pd, numpy as np
from scipy.stats import fisher_exact
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
D=f'{SP}/deeper'
OUT='/home/user/SAbR/register_error_analysis/retest/deeper/verify/case_files_DE_DE3_VH_VL_cooccurrence_integrity'
MODES={'Dplus_CDR3short','Dminus_CDR3long','E_FR4bulge'}
C=pd.read_csv(f'{D}/chains.csv.gz'); R=pd.read_csv(f'{D}/runs_first_occurrence.csv')
ER=pd.read_pickle(f'{D}/cache/case_files_DE/err_struct.pkl')
bad=set(zip(ER[ER.confirm=='not confirmed'].pdb, ER[ER.confirm=='not confirmed'].chain))
# consistency checks
k_runs=set(zip(R.pdb,R.chain)); k_errfirst=set(zip(C[C.err_first==1].pdb, C[C.err_first==1].chain))
print('chains with first-occ runs:',len(k_runs),' chains err_first==1:',len(k_errfirst),' sym diff:',len(k_runs^k_errfirst))
print('runs keys not in chains:',len(k_runs-set(zip(C.pdb,C.chain))))
# are all EM D/E chains in ER?
de=R[R['mode'].isin(MODES)]
k_de=set(zip(de.pdb,de.chain)); k_er=set(zip(ER.pdb,ER.chain))
emk=set(zip(C[C.method=='EM'].pdb,C[C.method=='EM'].chain))
print('EM D/E chains:',len(k_de&emk),' of which in err_struct:',len(k_de&emk&k_er))
print('EM D/E chains missing from err_struct:',sorted(k_de&emk-k_er))
md={}
for r in R.itertuples():
    if r.mode in MODES and (r.pdb,r.chain) in bad: continue
    md.setdefault((r.pdb,r.chain),set()).add(r.mode)
X=C[(C.method=='EM')&(C.vhh==0)].copy()
X['DE']=[bool(md.get(k,set())&MODES) for k in zip(X.pdb,X.chain)]
X['oth']=[bool(md.get(k,set())-MODES) for k in zip(X.pdb,X.chain)]
X['anyerr']=X.DE|X.oth
rows=[]
for p,x in X.groupby('pdb'):
    H=x[x.ctype=='H']; L=x[x.ctype!='H']
    if len(H) and len(L):
        rows.append(dict(pdb=p,paper=x.paper.iloc[0],year=x.year.iloc[0],res=x.res.iloc[0],nH=len(H),nL=len(L),
            H_DE=H.DE.any(),L_DE=L.DE.any(),H_oth=H.oth.any(),L_oth=L.oth.any(),L_any=L.anyerr.any(),H_any=H.anyerr.any(),
            H_modes=';'.join(sorted(set().union(*[md.get((p,c),set()) for c in H.chain])) ),
            L_modes=';'.join(sorted(set().union(*[md.get((p,c),set()) for c in L.chain])) )))
T=pd.DataFrame(rows)
print('n entries',len(T),' H_DE',T.H_DE.sum())
for a,b in [('H_DE','L_oth'),('H_DE','L_any'),('H_DE','L_DE')]:
    tab=pd.crosstab(T[a],T[b]).reindex(index=[False,True],columns=[False,True],fill_value=0)
    o,p=fisher_exact(tab.values)
    print(a,b,'given:',int(T[a].sum()),'both',int((T[a]&T[b]).sum()),'rate',round(T[T[a]][b].mean(),4),'vs',round(T[~T[a]][b].mean(),4),'OR',round(o,2),'p',p)
T.to_csv(f'{OUT}/entry_table.csv',index=False)
print(T[T.H_DE].sort_values(['L_any','paper'],ascending=False)[['pdb','paper','year','res','nH','nL','H_modes','L_modes']].to_string())
