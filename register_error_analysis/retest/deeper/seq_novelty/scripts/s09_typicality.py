"""Sequence typicality ("nativeness"-style) score: mean log10 frequency of each residue at its IMGT position, using the
profile of all unique PDB V domains of the same chain type released BEFORE the unit's deposition (leave-future-out).
Scored over framework, CDRs, and the mode-specific windows. Error vs clean EM units, adjusted (M2, M3)."""
import pickle, numpy as np, pandas as pd, warnings, statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings('ignore')
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
OUT='/home/user/SAbR/register_error_analysis/retest/deeper/seq_novelty/tables'
db=pickle.load(open(f'{SP}/deeper/cache/seq_novelty/abdb.pkl','rb')); D,M,keys=db['D'],db['M'],db['keys']
pos=np.array([k[0] for k in keys]); ins=np.array([k[1]!=' ' for k in keys])
U=pd.read_csv(f'{OUT}/units.csv.gz',low_memory=False)
U=U[(U.method_u=='EM')&(U.err_first_u|U.clean)&(U.dbrow>=0)&(U.dbrow<len(M))].copy()
REG={'FW':((pos<=26)|((pos>=39)&(pos<=55))|((pos>=66)&(pos<=104))|(pos>=118))&~ins,
     'CDR12':(((pos>=27)&(pos<=38))|((pos>=56)&(pos<=65))),'CDR3':(pos>=105)&(pos<=117),
     'win_A':(pos<=13),'win_F':(pos>=24)&(pos<=41),'win_B':(pos>=55)&(pos<=72),'win_C':(pos>=74)&(pos<=92),'win_D':(pos>=105),'win_E':(pos>=116)}
rel=pd.to_datetime(D.first_release.fillna('2999-01-01'),errors='coerce').values
U['dep']=pd.to_datetime(U.deposit)
# profiles per chain type per deposit-quarter (cumulative), pseudocount 0.5
U['q']=U.dep.dt.to_period('Q')
out=[]
for ct in 'HKL':
    sel=(D.ctype==ct).values
    for q,g in U[U.ctype==ct].groupby('q'):
        cut=np.datetime64(q.start_time)
        X=M[sel&(rel<cut)]
        F=np.stack([(X==a).sum(0) for a in range(22)]).astype(float)
        occ=F[1:].sum(0)
        P=(F+0.5)/(occ+0.5*20)[None]   # residue probability given occupied
        for _,u in g.iterrows():
            v=M[u.dbrow]; o=v>0
            lp=np.log10(P[v,np.arange(len(v))])
            r={'unit':u.unit}
            for k,m in REG.items():
                mm=m&o; r['ll_'+k]=lp[mm].mean() if mm.any() else np.nan
            out.append(r)
L=pd.DataFrame(out); U=U.merge(L,on='unit')
U.to_csv(f'{OUT}/units_typicality.csv.gz',index=False)
U['yb']=pd.cut(U.dep_year,[0,2018,2019,2020,2021,2022,2023,2100],labels=False)
U['nonhm']=(~U.q_species_all.isin(['human','mouse'])).astype(int); U['vhh_i']=U.vhh_u.fillna(0).astype(int); U['sars_i']=U.sars2_u.astype(int)
U['psa']=(U.n_prior_same_ab>0).astype(int); U['ntt']=U.seq_nt_trunc.astype(int)
rows=[]
MODES={'ANY':None,'A_FR1Nterm':'win_A','F_CDR1':'win_F','B_CDR2FR3':'win_B','C_FR3':'win_C','Dplus_CDR3short':'win_D','Dminus_CDR3long':'win_D','E_FR4bulge':'win_E'}
for m,w in MODES.items():
    cases=U.err_first_u if m=='ANY' else U.err_first_u&U.modes_u.fillna('').str.contains(m)
    ctrl=U.clean&(True if m=='ANY' else U.ctype.isin(U[cases].ctype.unique()))
    d=U[cases|ctrl].copy(); d['y']=cases[cases|ctrl].astype(int)
    for feat in ['ll_FW','ll_CDR12','ll_CDR3']+([w] and ['ll_'+w] if w else []):
        dd=d.dropna(subset=[feat]).copy(); dd['f']=(dd[feat]-dd[feat].mean())/dd[feat].std()
        r=dict(mode=m,feature=feat,n_err=int(dd.y.sum()),n_ctrl=int((dd.y==0).sum()),err_mean=dd[dd.y==1][feat].mean(),ctrl_mean=dd[dd.y==0][feat].mean())
        ctc=' + C(ctype)' if dd.ctype.nunique()>1 else ''
        for lab,cov,sub in [('M1',' + C(yb) + res_med'+ctc,dd),('M2',' + C(yb) + res_med'+ctc+' + nonhm + vhh_i + sars_i + ntt',dd),
                            ('M3',' + C(yb) + res_med'+ctc+' + sars_i + ntt',dd[(dd.psa==0)&(dd.nonhm==0)&(dd.vhh_i==0)])]:
            try:
                f=smf.logit('y ~ f'+cov,data=sub).fit(disp=0)
                r[f'OR_{lab}']=np.exp(f.params['f']); r[f'p_{lab}']=f.pvalues['f']; r[f'n_err_{lab}']=int(sub.y.sum())
            except Exception: r[f'OR_{lab}']=np.nan; r[f'p_{lab}']=np.nan
        rows.append(r)
T=pd.DataFrame(rows)
for lab in ['M1','M2','M3']:
    ok=T[f'p_{lab}'].notna(); T.loc[ok,f'q_{lab}']=multipletests(T.loc[ok,f'p_{lab}'],method='fdr_bh')[1]
T.to_csv(f'{OUT}/tests_typicality.csv',index=False)
pd.set_option('display.width',250)
print(T.round(4).to_string(index=False))
