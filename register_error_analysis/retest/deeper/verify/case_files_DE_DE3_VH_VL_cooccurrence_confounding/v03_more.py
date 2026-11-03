"""Additional checks: first-occurrence selection (any-run outcome), modelling protocol, unbuilt residues, specificity."""
import pandas as pd, numpy as np, json, warnings
from scipy.stats import fisher_exact
import statsmodels.formula.api as smf
warnings.filterwarnings('ignore')
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
CACHE=f'{SP}/deeper/cache/verify_DE3_confounding'
OUT='/home/user/SAbR/register_error_analysis/retest/deeper/verify/case_files_DE_DE3_VH_VL_cooccurrence_confounding'
CH=pd.read_pickle(f'{CACHE}/chains_flags.pkl'); T=pd.read_pickle(f'{CACHE}/T_em.pkl')
X=CH[(CH.method=='EM')&(CH.vhh==0)&CH.pdb.isin(T.pdb)]
keys=set(zip(X.pdb,X.chain))
# unbuilt residues in the V region per chain
try:
    UB=pd.read_pickle(f'{CACHE}/unbuilt.pkl')
except Exception:
    seqmap={k:(s,v) for k,s,v in zip(zip(X.pdb,X.chain),X.seq,X.vseq)}
    rows=[]
    for f in ['validation','snapshot_rest','new']:
        for l in open(f'{SP}/sabr/out/{f}.jsonl'):
            r=json.loads(l)
            k=(r['pdb'],r['chain'])
            if k not in keys or r.get('status')!='ok': continue
            s,v=seqmap[k]; st=s.find(v)
            if st<0: continue
            built=sorted(x[3] for x in r['residues'] if x[3] is not None and st<=x[3]<st+len(v))
            if not built: continue
            nb=len(set(built)); first=built[0]-st; last=built[-1]-st
            rows.append(dict(pdb=k[0],chain=k[1],vlen=len(v),unbuilt_V=len(v)-nb,unbuilt_N=first,unbuilt_C=len(v)-1-last,unbuilt_int=(last-first+1)-nb))
    UB=pd.DataFrame(rows).drop_duplicates(['pdb','chain']); UB.to_pickle(f'{CACHE}/unbuilt.pkl')
X=X.merge(UB,on=['pdb','chain'],how='left')
X['ub_int']=(X.unbuilt_int>0); X['ub_N']=(X.unbuilt_N>0)
e=X.groupby('pdb').apply(lambda x: pd.Series(dict(
    H_ub_int=x[x.ctype=='H'].ub_int.any(), L_ub_int=x[x.ctype!='H'].ub_int.any(),
    H_ub_N=x[x.ctype=='H'].ub_N.any(), L_ub_N=x[x.ctype!='H'].ub_N.any(),
    L_errany=x[x.ctype!='H'].err_any.any(), H_errany=x[x.ctype=='H'].err_any.any(),
    ub_cov=x.unbuilt_V.notna().all()))).reset_index()
T=T.merge(e,on='pdb',how='left')
mod={}
for l in open(f'{SP}/retest/modelling.jsonl'):
    r=json.loads(l); mod[r['rcsb_id'].lower()]=r
T['yb']=pd.cut(T.year,[0,2019,2021,2023,2030]).astype(str)
T['rigid']=(T.fit=='rigid body only').astype(int)
T['pdbmodel']=(T.irm=='experimental (PDB) model').astype(int)
T['sars2i']=T.sars2.astype(int)
for c in ['H_ub_int','L_ub_int','H_ub_N','L_ub_N']: T[c+'_i']=T[c].fillna(False).astype(int)
rows=[]
def run(label,df,a,b,form):
    tab=pd.crosstab(df[a],df[b]).reindex(index=[False,True],columns=[False,True],fill_value=0).values
    o,p=fisher_exact(tab)
    z=df.assign(y=df[b].astype(int),x=df[a].astype(int))
    try:
        m=smf.logit(form,data=z).fit(disp=0,maxiter=200,cov_type='cluster',cov_kwds={'groups':pd.factorize(z.paper)[0]})
        aor,ap=float(np.exp(m.params['x'])),float(m.pvalues['x']); ci=np.exp(m.conf_int().loc['x']).tolist()
        extra={k:(round(float(np.exp(m.params[k])),2),round(float(m.pvalues[k]),4)) for k in m.params.index if k not in ('Intercept','x') and not k.startswith('C(')}
    except Exception as ex:
        aor=ap=np.nan; ci=[np.nan,np.nan]; extra=str(ex)[:60]
    rows.append(dict(analysis=label,exposure=a,outcome=b,n=len(df),events_exposed=int(tab[1,1]),n_exposed=int(tab[1].sum()),rate_exposed=tab[1,1]/tab[1].sum(),
        events_unexposed=int(tab[0,1]),n_unexposed=int(tab[0].sum()),rate_unexposed=tab[0,1]/tab[0].sum(),fisher_OR=o,fisher_p=p,adj_OR=aor,adj_p=ap,adj_lo=ci[0],adj_hi=ci[1],covariate_ORs=str(extra),model=form))
base='y ~ x + C(yb) + res'
run('A any-run VL outcome (incl. copied/artifact runs)',T,'H_DE','L_errany',base)
run('B + rigid-body-only fit + PDB starting model',T,'H_DE','L_oth',base+' + rigid + pdbmodel')
Tc=T[T.ub_cov==True]
run('C + unbuilt internal V residues in H / L, N-term truncation of L',Tc,'H_DE','L_oth',base+' + H_ub_int_i + L_ub_int_i + L_ub_N_i')
run('C0 (same sample as C, base model)',Tc,'H_DE','L_oth',base)
run('D unbuilt internal residues in VH -> VL error (simpler factor alone)',Tc,'H_ub_int','L_oth',base)
# specificity: one model with both D/E-only, other-only, both
T['HDEonly']=(T.H_DE&~T.H_oth).astype(int); T['Hothonly']=(T.H_oth&~T.H_DE).astype(int); T['Hboth']=(T.H_DE&T.H_oth).astype(int)
z=T.assign(y=T.L_oth.astype(int))
m=smf.logit('y ~ HDEonly + Hothonly + Hboth + C(yb) + res',data=z).fit(disp=0,cov_type='cluster',cov_kwds={'groups':pd.factorize(z.paper)[0]})
w=m.wald_test('HDEonly = Hothonly',scalar=True)
spec=dict(analysis='E specificity: H D/E-only vs H other-only (same model)',OR_DEonly=float(np.exp(m.params['HDEonly'])),OR_othonly=float(np.exp(m.params['Hothonly'])),
          OR_both=float(np.exp(m.params['Hboth'])),p_equal=float(w.pvalue))
print(spec)
# leave-one-paper-out over exposed papers (entry level)
lo=[]
for p in T[T.H_DE].paper.unique():
    S=T[T.paper!=p]; tab=pd.crosstab(S.H_DE,S.L_oth).reindex(index=[False,True],columns=[False,True],fill_value=0).values
    lo.append((p,fisher_exact(tab)[0],fisher_exact(tab)[1]))
lo=pd.DataFrame(lo,columns=['dropped_paper','OR','p']); print('LOPO OR range',lo.OR.min(),lo.OR.max(),'max p',lo.p.max())
R=pd.DataFrame(rows); R.to_csv(f'{OUT}/tables/more_tests.csv',index=False)
pd.DataFrame([spec]).to_csv(f'{OUT}/tables/specificity.csv',index=False); lo.to_csv(f'{OUT}/tables/leave_one_paper_out.csv',index=False)
pd.set_option('display.width',250); pd.set_option('display.max_colwidth',120)
print(R.drop(columns=['model']).round(4).to_string())
