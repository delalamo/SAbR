"""Robustness tests for DE3: leave-one-paper-out, unit/paper-level collapse, within-paper strata, extra adjustment,
specificity vs non-D/E VH errors, outcome definitions (drop chain-end/next-to-gap VL runs, drop unconfirmed VL mode-A),
X-ray replication. Writes robustness.csv and lopo.csv."""
import pandas as pd, numpy as np, warnings
from scipy.stats import fisher_exact
import statsmodels.api as sm, statsmodels.formula.api as smf
from statsmodels.stats.contingency_tables import StratifiedTable
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings('ignore')
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
D=f'{SP}/deeper'
OUT='/home/user/SAbR/register_error_analysis/retest/deeper/verify/case_files_DE_DE3_VH_VL_cooccurrence_integrity'
MODES={'Dplus_CDR3short','Dminus_CDR3long','E_FR4bulge'}
CH=pd.read_csv(f'{D}/chains.csv.gz'); R=pd.read_csv(f'{D}/runs_first_occurrence.csv')
ER=pd.read_pickle(f'{D}/cache/case_files_DE/err_struct.pkl')
bad=set(zip(ER[ER.confirm=='not confirmed'].pdb, ER[ER.confirm=='not confirmed'].chain))
MA=pd.read_csv('/home/user/SAbR/register_error_analysis/retest/deeper/case_files_DE/tables/modeA_nterm_check.csv')
MA=MA[MA.label=='A error']
# mode-A not confirmed: propagate to all chains with the same V sequence in the same paper (same unit)
unit=dict(zip(zip(CH.pdb,CH.chain),CH.unit))
badA_units={unit[k] for k in zip(MA[MA.confirm=='not confirmed'].pdb, MA[MA.confirm=='not confirmed'].chain) if k in unit}

def build(method, outcome_filter=None, vhh0=True):
    md={}
    for r in R.itertuples():
        k=(r.pdb,r.chain)
        if r.mode in MODES and k in bad: continue
        if outcome_filter is not None and r.ctype!='H' and not outcome_filter(r): continue
        md.setdefault(k,set()).add(r.mode)
    X=CH[(CH.method==method)&((CH.vhh==0) if vhh0 else True)].copy()
    X['DE']=[bool(md.get(k,set())&MODES) for k in zip(X.pdb,X.chain)]
    X['oth']=[bool(md.get(k,set())-MODES) for k in zip(X.pdb,X.chain)]
    X['err']=X.DE|X.oth
    rows=[]
    for p,x in X.groupby('pdb'):
        H=x[x.ctype=='H']; L=x[x.ctype!='H']
        if len(H) and len(L):
            rows.append(dict(pdb=p,paper=x.paper.iloc[0],year=x.year.iloc[0],res=x.res.iloc[0],nH=len(H),nL=len(L),
                lqz=x.lq_z.mean(), H_DE=H.DE.any(),H_oth=H.oth.any(),H_any=H.err.any(),L_DE=L.DE.any(),L_oth=L.oth.any(),L_any=L.err.any(),
                Hunits='|'.join(sorted(set(H.unit))), Lunits='|'.join(sorted(set(L.unit)))))
    T=pd.DataFrame(rows); T['yb']=pd.cut(T.year,[0,2019,2021,2023,2030]).astype(str); return T

res=[]
def fisher(T,a,b,label):
    tab=pd.crosstab(T[a],T[b]).reindex(index=[False,True],columns=[False,True],fill_value=0)
    o,p=fisher_exact(tab.values)
    d=dict(test=label,given=a,outcome=b,n=len(T),n_given=int(T[a].sum()),events_given=int((T[a]&T[b]).sum()),
           rate_given=T[T[a]][b].mean(),rate_not=T[~T[a]][b].mean(),OR=o,p=p,
           n_papers_given=T[T[a]].paper.nunique(),n_papers_event=T[T[a]&T[b]].paper.nunique())
    res.append(d); return d

T=build('EM')
fisher(T,'H_DE','L_oth','entry: reproduce investigator (L non-D/E error)')
fisher(T,'H_DE','L_any','entry: L any first-occ error')
fisher(T,'H_oth','L_any','entry: VH non-D/E error -> L any')
fisher(T,'H_any','L_any','entry: VH any error -> L any')
# specificity: is D/E a stronger predictor than other VH errors? among entries with any VH error
Ta=T[T.H_any]
fisher(Ta.assign(H_DEonly=Ta.H_DE),'H_DEonly','L_any','entry, VH-error entries only: D/E vs other VH error')

# LOPO
lopo=[]
for pap in sorted(T[T.H_DE].paper.unique()):
    t=T[T.paper!=pap]; tab=pd.crosstab(t.H_DE,t.L_oth).reindex(index=[False,True],columns=[False,True],fill_value=0)
    o,p=fisher_exact(tab.values)
    lopo.append(dict(dropped=pap,n_dropped=int((T.paper==pap).sum()),given=int(t.H_DE.sum()),events=int((t.H_DE&t.L_oth).sum()),OR=o,p=p))
L=pd.DataFrame(lopo).sort_values('p',ascending=False); L.to_csv(f'{OUT}/lopo.csv',index=False)
print('LOPO worst cases:\n',L.head(5).to_string())
# drop the Cell Res paper (5 of 11 events) and drop the top-2 contributing papers
cr='10.1038/s41422-021-00487-9'
fisher(T[T.paper!=cr],'H_DE','L_oth','entry: drop Cell Res 2021 paper (5/11 events)')
fisher(T[T.paper!=cr],'H_DE','L_any','entry: drop Cell Res 2021, L any')

# collapse: one row per paper (max over entries)
P=T.groupby('paper').agg(H_DE=('H_DE','max'),L_any=('L_any','max'),L_oth=('L_oth','max'),H_oth=('H_oth','max'),year=('year','min'),res=('res','mean')).reset_index()
P['paper']=P['paper']
fisher(P,'H_DE','L_any','paper level (any entry)')
fisher(P,'H_DE','L_oth','paper level (any entry), L non-D/E')
# H-unit level: each VH unit; exposure = any entry of this unit has H_DE; outcome = any L in those entries has error
rows=[]
for p,x in T.groupby('paper'):
    for u in set('|'.join(x.Hunits).split('|')) if False else set():
        pass
# simpler: explode by H unit using chain-level table
md={}
for r in R.itertuples():
    k=(r.pdb,r.chain)
    if r.mode in MODES and k in bad: continue
    md.setdefault(k,set()).add(r.mode)
X=CH[(CH.method=="EM")&(CH.vhh==0)].copy()
X['DE']=[bool(md.get(k,set())&MODES) for k in zip(X.pdb,X.chain)]
X['err']=[bool(md.get(k,set())) for k in zip(X.pdb,X.chain)]
ent=set(T.pdb)
HX=X[(X.ctype=='H')&X.pdb.isin(ent)]; LX=X[(X.ctype!='H')&X.pdb.isin(ent)]
Lerr=LX.groupby('pdb').err.any()
U=HX.groupby('unit').agg(paper=('paper','first'),H_DE=('DE','max'),pdbs=('pdb',lambda s:set(s))).reset_index()
U['L_any']=[any(Lerr.get(p,False) for p in s) for s in U.pdbs]
fisher(U,'H_DE','L_any','VH-unit level (antibody-in-paper)')

# within-paper: Mantel-Haenszel over papers with both exposed and unexposed entries
strata=[]
for p,x in T.groupby('paper'):
    tab=pd.crosstab(x.H_DE,x.L_any).reindex(index=[True,False],columns=[True,False],fill_value=0).values
    if tab[0].sum()>0 and tab[1].sum()>0: strata.append(tab.astype(float)+0.0)
st=StratifiedTable(strata)
mh=dict(test='within-paper Mantel-Haenszel (entry, L any)',given='H_DE',outcome='L_any',n=int(sum(t.sum() for t in strata)),
        n_given=int(sum(t[0].sum() for t in strata)),events_given=int(sum(t[0,0] for t in strata)),OR=st.oddsratio_pooled,
        p=st.test_null_odds(correction=True).pvalue,n_papers_given=len(strata))
res.append(mh); print('MH strata',len(strata),mh)
for t in strata: print(t.astype(int).tolist())

# adjusted models: GEE exchangeable by paper, with and without map-fit (lq_z) and n chains
for label,form,dat in [('GEE yb+res','L_any ~ H_DE + C(yb) + res',T),
                       ('GEE yb+res+lqz+nL','L_any ~ H_DE + C(yb) + res + lqz + nL',T.dropna(subset=['lqz'])),
                       ('GEE yb+res+lqz+nL, drop Cell Res','L_any ~ H_DE + C(yb) + res + lqz + nL',T[T.paper!=cr].dropna(subset=['lqz'])),
                       ('GEE H_DE + H_oth (specificity)','L_any ~ H_DE + H_oth + C(yb) + res + lqz + nL',T.dropna(subset=['lqz']))]:
    d=dat.assign(L_any=dat.L_any.astype(int),H_DE=dat.H_DE.astype(int),H_oth=dat.H_oth.astype(int)).sort_values('paper')
    try:
        m=smf.gee(form,groups=pd.factorize(d.paper)[0],data=d,family=sm.families.Binomial(),cov_struct=sm.cov_struct.Exchangeable()).fit()
        res.append(dict(test=label,given='H_DE',outcome='L_any',n=len(d),n_given=int(d.H_DE.sum()),events_given=int((d.H_DE&d.L_any).sum()),
                        OR=float(np.exp(m.params['H_DE'])),p=float(m.pvalues['H_DE']),
                        OR_H_oth=float(np.exp(m.params['H_oth'])) if 'H_oth' in m.params else np.nan,
                        p_H_oth=float(m.pvalues['H_oth']) if 'H_oth' in m.params else np.nan))
    except Exception as e: print(label,e)

# paper-cluster permutation: shuffle L_any outcome vectors across papers within year bin (keeps within-paper correlation)
rng=np.random.default_rng(1)
obs=(T.H_DE&T.L_any).sum()
# permutation at paper level: permute paper-level outcome among papers within year bin, statistic = #exposed papers w/ event
Pp=T.groupby('paper').agg(H_DE=('H_DE','max'),L_any=('L_any','max'),yb=('yb','first')).reset_index()
obsP=int((Pp.H_DE&Pp.L_any).sum()); cnt=0; NP=20000
for i in range(NP):
    perm=Pp.groupby('yb').L_any.transform(lambda s: rng.permutation(s.values))
    cnt+= int((Pp.H_DE & perm.astype(bool)).sum())>=obsP
res.append(dict(test='paper-level permutation within year bin',given='H_DE',outcome='L_any',n=len(Pp),n_given=int(Pp.H_DE.sum()),
                events_given=obsP,p=(cnt+1)/(NP+1),rate_given=Pp[Pp.H_DE].L_any.mean(),rate_not=Pp[~Pp.H_DE].L_any.mean()))

# outcome definitions
T3=build('EM',outcome_filter=lambda r: not str(r.origin).startswith('3'))
fisher(T3,'H_DE','L_any','entry: VL runs at chain end/next to unbuilt loop (origin 3) removed')
TA=build('EM',outcome_filter=lambda r: not (r.mode=='A_FR1Nterm' and unit.get((r.pdb,r.chain)) in badA_units))
fisher(TA,'H_DE','L_any','entry: VL mode-A runs not confirmed by investigator check removed')
TA2=build('EM',outcome_filter=lambda r: not (r.mode=='A_FR1Nterm' and unit.get((r.pdb,r.chain)) in badA_units))
fisher(TA2[TA2.paper!=cr],'H_DE','L_any','entry: unconfirmed VL A removed + Cell Res dropped')
# X-ray replication (no confirmation filter available outside ER; bad set only covers tested chains)
TX=build('X-ray')
fisher(TX,'H_DE','L_any','X-ray replication: entry')
fisher(TX,'H_any','L_any','X-ray: VH any -> VL any')

O=pd.DataFrame(res)
O['q_BH']=multipletests(O.p.fillna(1),method='fdr_bh')[1]
O.to_csv(f'{OUT}/robustness.csv',index=False)
pd.set_option('display.width',260); pd.set_option('display.max_colwidth',70)
print(O[['test','given','outcome','n','n_given','events_given','rate_given','rate_not','OR','p','q_BH','n_papers_given','n_papers_event','OR_H_oth','p_H_oth']].round(4).to_string())
