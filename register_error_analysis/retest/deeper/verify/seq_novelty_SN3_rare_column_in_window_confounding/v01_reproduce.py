"""Reproduce SN3 (rare IMGT column in the run window, 14/199 vs 0.47% expected) and list, for every positive, which rare
column(s) are hit, where the run sits relative to them, the chain's per-residue SAbR/ANARCI numbering, and the
unit's paper/lab/year/SARS-CoV-2/resolution context. Also: alternative control definitions for the expectation."""
import pickle, numpy as np, pandas as pd
from scipy import stats
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
SN='/home/user/SAbR/register_error_analysis/retest/deeper/seq_novelty/tables'
OUT='/home/user/SAbR/register_error_analysis/retest/deeper/verify/seq_novelty_SN3_rare_column_in_window_confounding'
U=pd.read_csv(f'{SN}/units.csv.gz',low_memory=False).set_index('unit')
W=pd.read_csv(f'{SN}/unit_mode_windows.csv')
db=pickle.load(open(f'{SP}/deeper/cache/seq_novelty/abdb.pkl','rb')); D,M,keys,code=db['D'],db['M'],db['keys'],db['code']
inv={v:k for k,v in code.items()}
pos=np.array([k[0] for k in keys])
RARE={}
for ct in 'HKL':
    sel=(D.ctype==ct).values; X=M[sel]; occ=(X>0).sum(0); RARE[ct]=(occ,sel.sum())
def rins(row,ct):
    occ,n=RARE[ct]; q=M[row]; return (q>0)&(occ/n<0.01)
U['yb']=pd.cut(U.dep_year,[0,2018,2019,2020,2021,2022,2023,2100],labels=False)
EM=U[U.method_u=='EM']
ok=lambda d: d[(d.dbrow>=0)&(d.dbrow<len(M))]
CTRL={'clean_EM (investigator)':ok(EM[EM.clean]),
      'all_EM_not_first_error':ok(EM[~EM.err_first_u]),
      'all_EM_units':ok(EM),
      'clean_Xray':ok(U[(U.method_u=='X-ray')&U.clean]),
      'all_units_any_method_not_first':ok(U[~U.err_first_u])}
FL={}
def fl(row,ct):
    if row not in FL: FL[row]=rins(row,ct)
    return FL[row]
rows=[]
for _,w in W.iterrows():
    if w.unit not in EM.index or not EM.loc[w.unit,'err_first_u'] or w['mode']=='other': continue
    u=EM.loc[w.unit]
    if u.dbrow<0 or u.dbrow>=len(M): continue
    a=int(w.i0)-2; b=min(int(w.i1),int(w.i0)+24)+2; win=(pos>=a)&(pos<=b)
    ri=fl(u.dbrow,u.ctype)
    d=dict(unit=w.unit,mode=w['mode'],ctype=u.ctype,i0=w.i0,i1=w.i1,obs=bool(ri[win].any()),
           cols=' '.join(f'{keys[j][0]}{keys[j][1].strip()}' for j in np.where(ri&win)[0]),
           cols_anywhere=' '.join(f'{keys[j][0]}{keys[j][1].strip()}' for j in np.where(ri)[0]),
           paper=u.paper,last_author=u.last_author,pdb=u.pdb,dep_year=u.dep_year,res=u.res_med,sars2=bool(u.sars2_u),
           vhh=u.vhh_u,species=u.q_species_all,yb=u.yb)
    for nm,C in CTRL.items():
        cc=C[(C.ctype==u.ctype)&(C.yb==u.yb)]
        if nm.startswith('all_units') or 'Xray' in nm: cc=C[C.ctype==u.ctype]  # few X-ray in late bins: chain type only
        d['exp_'+nm]=np.mean([fl(r,u.ctype)[win].any() for r in cc.dbrow.values]) if len(cc) else np.nan
        d['n_'+nm]=len(cc)
    rows.append(d)
R=pd.DataFrame(rows); R.to_csv(f'{OUT}/v01_unit_modes.csv',index=False)
print('unit-modes',len(R),'positives',R.obs.sum())
for nm in CTRL:
    e=R['exp_'+nm]; o=R.obs.sum(); E=e.sum(); v=(e*(1-e)).sum()
    # exact Poisson-binomial tail via normal approx and via Poisson on E
    print(f'{nm:35s} expected {E:.2f} ({E/len(R):.4f}) O/E {o/E:.1f}  poisson p={stats.poisson.sf(o-1,E):.2e}')
pd.set_option('display.width',250); pd.set_option('display.max_colwidth',40)
P=R[R.obs]
print(P[['mode','ctype','i0','i1','cols','cols_anywhere','pdb','paper','last_author','dep_year','res','sars2','vhh','species']].to_string())
# column occupancy of the hit columns
print('\nOccupancy (% of PDB unique V domains of that chain type) of each hit column:')
for ct in 'HKL':
    occ,n=RARE[ct]
    for c in sorted(set(' '.join(P[P.ctype==ct].cols).split())):
        p=int(''.join(ch for ch in c if ch.isdigit())); i=''.join(ch for ch in c if ch.isalpha()) or ' '
        j=keys.index((p,i)); print(ct,c,f'{100*occ[j]/n:.2f}%')
