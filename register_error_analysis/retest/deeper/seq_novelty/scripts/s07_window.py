"""Exact-location tests. For every first-occurrence error unit x mode (EM), take the run window [imgt0-2, imgt1+2]
(runs longer than 25 positions are truncated to their first 25) and ask whether the antibody carries
(i) a residue that is rare (<1%) at that IMGT position among PDB V domains, (ii) a residue at a rarely occupied IMGT column,
compared with the expectation from clean EM units of the same chain type in the same window (and same deposit-year bin).
Also: J-germline identity split into the CDR3 part (IMGT 105-117) and FR4 (118-128) for D modes."""
import pickle, numpy as np, pandas as pd
from scipy import stats
from anarci.germlines import all_germlines as G
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
CACHE=f'{SP}/deeper/cache/seq_novelty'
OUT='/home/user/SAbR/register_error_analysis/retest/deeper/seq_novelty/tables'
U=pd.read_csv(f'{OUT}/units.csv.gz',low_memory=False).set_index('unit').copy()
W=pd.read_csv(f'{OUT}/unit_mode_windows.csv')
db=pickle.load(open(f'{CACHE}/abdb.pkl','rb')); D,M,keys,code,S=db['D'],db['M'],db['keys'],db['code'],db['S']
pos=np.array([k[0] for k in keys])
# rare matrices per chain type (computed on unique V domains, as in s04)
RARE={}; RINS={}
for ct in 'HKL':
    sel=(D.ctype==ct).values; X=M[sel]
    F=np.stack([(X==a).sum(0) for a in range(22)]); occ=F[1:].sum(0)
    RARE[ct]=(F,occ,sel.sum())
def flags(row,ct):
    F,occ,n=RARE[ct]; q=M[row]
    f=np.where(occ>0,F[q,np.arange(len(q))]/np.maximum(occ,1),1.0)
    return (q>0)&(f<0.01), (q>0)&(occ/n<0.01)
# dbrow for extra rows (chains not in DB) is beyond len(M) only if s04 appended; skip those
U['yb']=pd.cut(U.dep_year,[0,2018,2019,2020,2021,2022,2023,2100],labels=False)
EM=U[U.method_u=='EM']
ctrl=EM[EM.clean & (EM.dbrow<len(M))]
cache={}
def fl(row,ct):
    if row not in cache: cache[row]=flags(row,ct)
    return cache[row]
rows=[]
for _,w in W.iterrows():
    if w.unit not in EM.index or not EM.loc[w.unit,'err_first_u'] or w['mode'] in ('other',): continue
    u=EM.loc[w.unit]
    if u.dbrow>=len(M): continue
    a=int(w.i0)-2; b=min(int(w.i1),int(w.i0)+24)+2
    win=(pos>=a)&(pos<=b)
    r,ri=fl(u.dbrow,u.ctype)
    cc=ctrl[(ctrl.ctype==u.ctype)&(ctrl.yb==u.yb)]
    er=[];ei=[]
    for row in cc.dbrow.values:
        rr,rri=fl(row,u.ctype); er.append(rr[win].any()); ei.append(rri[win].any())
    rows.append(dict(unit=w.unit,mode=w['mode'],ctype=u.ctype,i0=w.i0,i1=w.i1,obs_rare=bool(r[win].any()),exp_rare=np.mean(er),
                     obs_rins=bool(ri[win].any()),exp_rins=np.mean(ei),n_ctrl=len(cc),
                     rare_res=' '.join(f'{keys[j][0]}{keys[j][1].strip()}' for j in np.where(r&win)[0])))
R=pd.DataFrame(rows); R.to_csv(f'{OUT}/window_rare_matched.csv',index=False)
def summ(x,lab):
    out=[]
    for k in ['rare','rins']:
        o=x[f'obs_{k}'].sum(); e=x[f'exp_{k}'].sum(); v=(x[f'exp_{k}']*(1-x[f'exp_{k}'])).sum()
        z=(o-e)/np.sqrt(v) if v>0 else np.nan; p=2*stats.norm.sf(abs(z)) if v>0 else np.nan
        out.append(dict(group=lab,feature='rare residue (<1%) in run window' if k=='rare' else 'residue at rarely occupied IMGT column in run window',
                        n_units=len(x),observed=int(o),expected=round(e,2),obs_frac=o/len(x),exp_frac=e/len(x),ratio=o/e if e>0 else np.nan,p=p))
    return out
S_=[]
S_+=summ(R,'all modes')
for m,x in R.groupby('mode'): S_+=summ(x,m)
S_=pd.DataFrame(S_); S_.to_csv(f'{OUT}/window_rare_summary.csv',index=False)
print(S_.round(3).to_string(index=False))
# ---- J identity split
def jsplit(row,ct,sp,jg):
    g=G['J'].get(ct,{}).get(sp,{}).get(jg)
    if g is None: return np.nan,np.nan
    s=S[row]; inv={v:k for k,v in code.items()}
    res=[]
    for a,b in ((105,117),(118,128)):
        m=n=0
        for p in range(a,b+1):
            if g[p-1]=='-': continue
            n+=1; m+=(s[p-1]>0 and inv[s[p-1]]==g[p-1])
        res.append(m/n if n else np.nan)
    return res
U2=EM[(EM.err_first_u|EM.clean)&(EM.dbrow<len(M))].copy()
U2[['j_id_cdr3part','j_id_fr4']]=[jsplit(r.dbrow,r.ctype,r.q_species_hm,r.q_j_gene_hm) for _,r in U2.iterrows()]
U2['D']=U2.err_first_u&U2.modes_u.fillna('').str.contains('Dplus|Dminus|D0')
import statsmodels.formula.api as smf
out=[]
U2['nonhm']=(~U2.q_species_all.isin(['human','mouse'])).astype(int); U2['vhh_i']=U2.vhh_u.fillna(0).astype(int); U2['sars_i']=U2.sars2_u.astype(int)
U2['psa']=(U2.n_prior_same_ab>0).astype(int)
for col in ['j_id_cdr3part','j_id_fr4','q_j_id_hm']:
    for lab,sub,cov in [('M2',lambda d:d.index==d.index,' + nonhm + vhh_i + sars_i'),('M2 excl. FR4-truncated SEQRES',lambda d:~d.seq_fr4_trunc,' + nonhm + vhh_i + sars_i'),
                        ('M3 excl. FR4-truncated',lambda d:(~d.seq_fr4_trunc)&(d.psa==0)&(d.nonhm==0)&(d.vhh_i==0),' + sars_i')]:
        d=U2[(U2.D|U2.clean)].dropna(subset=[col]).copy(); d=d[sub(d)]; d['y']=d.D.astype(int); d['f']=(d[col]-d[col].mean())/d[col].std()
        f=smf.logit('y ~ f + C(yb) + res_med + C(ctype)'+cov,data=d).fit(disp=0)
        out.append(dict(feature=col,model=lab,n_err=int(d.y.sum()),n_ctrl=int((d.y==0).sum()),err_mean=d[d.y==1][col].mean(),ctrl_mean=d[d.y==0][col].mean(),
                    p_mw=stats.mannwhitneyu(d[d.y==1][col],d[d.y==0][col]).pvalue,OR_per_SD=np.exp(f.params['f']),p=f.pvalues['f']))
print(pd.DataFrame(out).round(4).to_string(index=False))
pd.DataFrame(out).to_csv(f'{OUT}/D_jsplit.csv',index=False)
