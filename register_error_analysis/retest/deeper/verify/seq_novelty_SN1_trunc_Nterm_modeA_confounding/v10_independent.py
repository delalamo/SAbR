"""Independent re-checks for SN1 (mode A <- SEQRES starts after IMGT 1), confounding lens, second pass.
  A. deposited entity sequence (RCSB entity_poly) for the 19 truncated cases: truly truncated? modified N-terminal residues?
  B. model start recomputed from SAbR output (sequence start S = ANARCI IMGT of first built residue; structural start = SAbR IMGT)
  C. within-lab conditional logit; one-row-per-paper analysis; X-ray replication
  D. adjustment for the known factor as previously coded (deepdive: unbuilt residues within window or first built > 3)
  E. direction check (does the shift itself create the 'unbuilt' residue 1?) and bounding with structural start
"""
import json, numpy as np, pandas as pd, warnings, statsmodels.formula.api as smf
from statsmodels.discrete.conditional_models import ConditionalLogit
from scipy import stats
warnings.filterwarnings('ignore')
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
SN='/home/user/SAbR/register_error_analysis/retest/deeper/seq_novelty/tables'
OUT='/home/user/SAbR/register_error_analysis/retest/deeper/verify/seq_novelty_SN1_trunc_Nterm_modeA_confounding'
CACHE=f'{SP}/deeper/cache/verify_SN1_confounding'
L=[]; ROWS=[]
def P(*a):
    s=' '.join(str(x) for x in a); print(s); L.append(s)

U=pd.read_csv(f'{SN}/units.csv.gz',low_memory=False)
ch=pd.read_csv(f'{SP}/deeper/chains.csv.gz',low_memory=False,usecols=['pdb','chain','unit','ctype','method','deposit','seq','vseq','err_first','err_any'])
runs=pd.read_csv(f'{SP}/deeper/runs_first_occurrence.csv')
Akeys=set(zip(runs.loc[runs['mode']=='A_FR1Nterm','pdb'],runs.loc[runs['mode']=='A_FR1Nterm','chain']))
# ---- per-chain N-terminal features recomputed from SAbR per-residue output
REC={}
for fn in ['new.jsonl','snapshot_rest.jsonl','validation.jsonl']:
    for line in open(f'{SP}/sabr/out/{fn}'):
        r=json.loads(line)
        if r.get('status')=='ok' and r.get('residues'): REC[(r['pdb'],r['chain'])]=r['residues']
S=[];SS=[];FI=[];UNB=[]
for p,c in zip(ch.pdb,ch.chain):
    res=REC.get((p,c))
    a=[x for x in res if x[4] is not None] if res else []
    if not a: S.append(np.nan); SS.append(np.nan); FI.append(np.nan); UNB.append(np.nan); continue
    f=min(a,key=lambda x:(x[4],x[3]))
    S.append(f[4]); SS.append(f[6] if f[6] is not None else np.nan); FI.append(f[3])
    UNB.append(f[3])  # SEQRES residues before the first built V residue (incl. tags)
ch['S']=S; ch['S_sabr']=SS; ch['first_seqidx']=FI
ch['v_off']=[s.find(v) if isinstance(s,str) and isinstance(v,str) and v else -1 for s,v in zip(ch.seq,ch.vseq)]
ch['deposit']=pd.to_datetime(ch.deposit); ch['isA']=[k in Akeys for k in zip(ch.pdb,ch.chain)]
# SEQRES first IMGT from the investigator's unit table is per earliest chain; recompute an independent T:
# T_alt = first built V residue is the first V residue in SEQRES (no unmodelled V residues before it) AND S>1
ch['T_alt']=((ch.first_seqidx==ch.v_off)&(ch.S>1)).astype(float)
ch.loc[ch.S.isna(),'T_alt']=np.nan
EM=U.method_u=='EM'
U['caseA']=EM&U.err_first_u&U.modes_u.str.contains('A_FR1Nterm')
U['ctrl']=EM&U.clean
rep=ch.sort_values(['unit','deposit','pdb','chain'])
rA=rep[rep.isA&rep.err_first].groupby('unit').first(); r0=rep.groupby('unit').first()
D=U[U.caseA|U.ctrl].copy()
cA=D.caseA.values
for c in ['pdb','chain','S','S_sabr','T_alt','first_seqidx','v_off']:
    D['x_'+c]=np.where(cA,D.unit.map(rA[c]),D.unit.map(r0[c]))
D['y']=D.caseA.astype(int)
D['TR']=D.seq_nt_trunc.astype(int)
D=D[D.ctype.isin(D[D.y==1].ctype.unique())]
D['yb']=pd.cut(D.dep_year,[0,2018,2019,2020,2021,2022,2023,2100],labels=['<=2018','2019','2020','2021','2022','2023','>=2024']).astype(str)
D['nonhm']=(~D.q_species_all.isin(['human','mouse'])).astype(int); D['vhh_i']=D.vhh_u.fillna(0).astype(int); D['sars_i']=D.sars2_u.astype(int)
D['lab']=D.last_author.fillna(D.paper.astype(str))
P(f'cases {D.y.sum()} controls {(D.y==0).sum()}; T cases {D[D.y==1]['TR'].mean():.3f} ({D[D.y==1]['TR'].sum()}) controls {D[D.y==0]['TR'].mean():.4f}')
P('T (investigator, ANARCI on SEQRES) vs T_alt (SEQRES index of first built residue == V start and S>1):')
P(pd.crosstab(D['TR'],D.x_T_alt.fillna(-1)).to_string())

def fit(formula,d,label,feat='TR',cluster=None,note=''):
    try:
        if cluster is not None:
            import patsy
            vars_=[v for v in patsy.ModelDesc.from_formula(formula).rhs_termlist for v in v.factors]
            d=d.dropna(subset=[c for c in d.columns if any(c in str(x) for x in vars_)]+['y'])
        m=smf.logit(formula,data=d)
        f=m.fit(disp=0,maxiter=300) if cluster is None else m.fit(disp=0,maxiter=300,cov_type='cluster',cov_kwds={'groups':pd.factorize(d[cluster])[0]})
        OR=np.exp(f.params[feat]); lo,hi=np.exp(f.conf_int().loc[feat]); p=f.pvalues[feat]
    except Exception as e:
        OR=lo=hi=p=np.nan; P('  fail',label,str(e)[:80])
    n1=int(d.y.sum()); n0=int((d.y==0).sum()); e1=d.loc[d.y==1,feat].mean(); e0=d.loc[d.y==0,feat].mean()
    ROWS.append(dict(test=label,feature=feat,n_err=n1,n_ctrl=n0,err_value=e1,ctrl_value=e0,OR=OR,lo=lo,hi=hi,p=p,note=note))
    P(f'{label:78s} n={n1}/{n0} {e1:.3f} vs {e0:.3f} OR {OR:.2f} [{lo:.2f}-{hi:.2f}] p={p:.2g}')
M1=' + C(yb) + res_med + C(ctype)'; M2=M1+' + vhh_i + nonhm + sars_i'
P('\n=== B. reproduction')
fit('y ~ TR'+M2,D,'M2 (investigator covariates), EM units')
fit('y ~ TR'+M2,D,'M2, cluster-robust by last author',cluster='lab')
fit('y ~ T_alt'+M2,D.assign(T_alt=D.x_T_alt).dropna(subset=['T_alt']),'M2 with independent T_alt (from SAbR seq_index)',feat='T_alt')

P('\n=== C1. within-lab conditional logit (strata = last author; only labs with >=1 case contribute)')
d=D.copy(); g=d.groupby('lab').y.transform('sum'); d=d[g>0]
d=d[d.groupby('lab')['TR'].transform('nunique')>1]
try:
    cl=ConditionalLogit(d.y.values,d[['TR']].astype(float).values,groups=pd.factorize(d.lab)[0]).fit(disp=0)
    OR=np.exp(cl.params[0]); lo,hi=np.exp(cl.conf_int()[0]); p=cl.pvalues[0]
except Exception as e:
    OR=lo=hi=p=np.nan; P('fail',e)
n1=int(d.y.sum()); n0=int((d.y==0).sum())
P(f'within-lab: labs {d.lab.nunique()}, units {len(d)}, cases {n1}; T cases {d[d.y==1]["TR"].mean():.3f} vs controls {d[d.y==0]["TR"].mean():.3f}; OR {OR:.2f} [{lo:.2f}-{hi:.2f}] p={p:.2g}')
ROWS.append(dict(test='within-lab conditional logit (labs with a case and T variation)',feature='TR',n_err=n1,n_ctrl=n0,err_value=d[d.y==1]['TR'].mean(),ctrl_value=d[d.y==0]['TR'].mean(),OR=OR,lo=lo,hi=hi,p=p))
# lab habit: do labs with a mode-A case truncate SEQRES more in their OTHER (clean) units?
labs_case=set(D[D.y==1].lab)
c0=D[D.y==0]
a=c0[c0.lab.isin(labs_case)]['TR']; b=c0[~c0.lab.isin(labs_case)]['TR']
t=[[a.sum(),len(a)-a.sum()],[b.sum(),len(b)-b.sum()]]; orr,p=stats.fisher_exact(t)
P(f'lab habit: clean units of labs with a mode-A case truncated {a.mean():.3f} ({a.sum()}/{len(a)}) vs other labs {b.mean():.3f} ({b.sum()}/{len(b)}); OR {orr:.2f} p={p:.2g}')
ROWS.append(dict(test='lab habit: clean units of case labs vs other labs',feature='TR',n_err=len(a),n_ctrl=len(b),err_value=a.mean(),ctrl_value=b.mean(),OR=orr,p=p,note='Fisher'))

P('\n=== C2. one row per paper (case paper = >=1 mode-A unit; control paper = all units clean)')
Ue=U[EM].copy(); Ue['TR']=Ue.seq_nt_trunc.astype(int)
pp=Ue.groupby('paper').agg(case=('caseA','any'),allclean=('clean','all'),Tany=('TR','max'),Tfrac=('TR','mean'),n=('unit','size'),
                           yr=('dep_year','min'),res=('res_med','median'),sars=('sars2_u','max'),vhh=('vhh_u','max')).reset_index()
pp=pp[pp.case|pp.allclean]; pp['y']=pp.case.astype(int); pp['logn']=np.log(pp.n)
pp['yb']=pd.cut(pp.yr,[0,2018,2019,2020,2021,2022,2023,2100]).astype(str); pp['sars']=pp.sars.astype(int); pp['vhh']=pp.vhh.fillna(0).astype(int)
fit('y ~ Tany + C(yb) + res + logn + sars + vhh',pp.rename(columns={}),'paper level: any truncated V chain in paper',feat='Tany')
# within case papers: is the truncated unit the mode-A unit?  (unit-level within paper)
cp=set(pp[pp.y==1].paper)
w=U[EM&U.paper.isin(cp)&(U.caseA|U.clean)].copy(); w['TR']=w.seq_nt_trunc.astype(int); w['y']=w.caseA.astype(int)
w=w[w.groupby('paper')['TR'].transform('nunique')>1]
try:
    cl=ConditionalLogit(w.y.values,w[['TR']].astype(float).values,groups=pd.factorize(w.paper)[0]).fit(disp=0)
    OR=np.exp(cl.params[0]); lo,hi=np.exp(cl.conf_int()[0]); p=cl.pvalues[0]
except Exception as e:
    OR=lo=hi=p=np.nan
P(f'within-paper conditional logit: papers {w.paper.nunique()}, units {len(w)}, cases {w.y.sum()}; OR {OR:.2f} [{lo:.2f}-{hi:.2f}] p={p:.2g}')
ROWS.append(dict(test='within-paper conditional logit',feature='TR',n_err=int(w.y.sum()),n_ctrl=int((w.y==0).sum()),err_value=w[w.y==1]['TR'].mean() if len(w) else np.nan,ctrl_value=w[w.y==0]['TR'].mean() if len(w) else np.nan,OR=OR,lo=lo,hi=hi,p=p))

P('\n=== C3. X-ray replication (X-ray mode-A first-occurrence units vs clean X-ray units)')
X=U[(U.method_u=='X-ray')&((U.err_first_u&U.modes_u.str.contains('A_FR1Nterm'))|U.clean)].copy()
X['y']=(X.err_first_u&X.modes_u.str.contains('A_FR1Nterm')).astype(int); X['TR']=X.seq_nt_trunc.astype(int)
X=X[X.ctype.isin(X[X.y==1].ctype.unique())]
t=pd.crosstab(X.y,X['TR']).reindex(index=[0,1],columns=[0,1],fill_value=0); orr,p=stats.fisher_exact(t.values)
P(f'X-ray: cases {t.loc[1,1]}/{t.loc[1].sum()} vs controls {t.loc[0,1]}/{t.loc[0].sum()}  OR {orr:.2f} p={p:.2g}')
ROWS.append(dict(test='X-ray replication (Fisher)',feature='TR',n_err=int(t.loc[1].sum()),n_ctrl=int(t.loc[0].sum()),err_value=t.loc[1,1]/max(1,t.loc[1].sum()),ctrl_value=t.loc[0,1]/t.loc[0].sum(),OR=orr,p=p))

P('\n=== D. the known factor as previously coded (deepdive): unbuilt V residue in IMGT 1-19 (in SEQRES numbering) or first built > 3')
# unbuilt residues within window A (1..14) +-5 using SEQRES numbering: count V residues before the first built one = first_seqidx - v_off
D['unb_seqres']=(D.x_first_seqidx-D.x_v_off).clip(lower=0)
D['dd_gap']=((D.unb_seqres>0)|(D.x_S>3)).astype(int)
P(pd.crosstab([D.y,D['TR']],D.dd_gap).to_string())
fit('y ~ TR'+M2+' + dd_gap',D,'M2 + deepdive gap flag (unbuilt-in-SEQRES or start>3)')
D['ms']=pd.cut(D.x_S,[0,1,2,3,100],labels=['1','2','3','4+']).astype(str)
fit('y ~ TR'+M2+' + C(ms)',D,'M2 + sequence model start S (1/2/3/4+)')
fit('y ~ TR'+M2+' + C(ms)',D,'M2 + S, cluster-robust by last author',cluster='lab')
d=D[(D.x_S>1)&(D.x_S<10)]
fit('y ~ TR + C(ms) + res_med + C(ctype) + C(yb)',d,'S in 2-9 only: truncated record vs unbuilt-but-in-record')

P('\n=== E. direction: does the shift itself remove residue 1 from the model?')
c=D[(D.y==1)]
P('cases: sequence start S vs structural start (SAbR IMGT of the same residue):')
P(pd.crosstab(c.x_S.clip(0,6),c.x_S_sabr.clip(-1,6)).to_string())
fT=c[c['TR']==0]; z=fT[fT.x_S>1]
P(f'full-SEQRES cases with S>1: {len(z)}; of these, structural slot 1 (or 0) occupied by the shifted residue: {(z.x_S_sabr<=1).sum()}')
zt=c[c['TR']==1]; P(f'truncated cases: {len(zt)}; structural slot 1 (or 0) occupied: {(zt.x_S_sabr<=1).sum()}')
# bounding: adjust for structural start instead of sequence start
D['ms_str']=pd.cut(D.x_S_sabr,[-100,1,2,3,100],labels=['<=1','2','3','4+']).astype(str)
fit('y ~ TR'+M2+' + C(ms_str)',D,'M2 + structural start (SAbR) instead of S [upper bound]')
# sensitivity: drop full-SEQRES cases whose S>1 may be caused by the shift (structural slot <=1 occupied)
drop=(D.y==1)&(D['TR']==0)&(D.x_S>1)&(D.x_S_sabr<=1)
fit('y ~ TR'+M2+' + C(ms)',D[~drop],'M2 + S, excluding full-SEQRES cases with shift-filled slot 1')

P('\n=== A. deposited entity sequences (RCSB entity_poly) for the 19 truncated cases')
ep=json.load(open(f'{CACHE}/case_entity_poly.json'))['data']['entries']
EP={}
for e in ep:
    for pe in e['polymer_entities']:
        for s in (pe['entity_poly']['pdbx_strand_id'] or '').split(','):
            EP[(e['rcsb_id'].lower(),s.strip())]=pe['entity_poly']
tc=D[(D.y==1)&(D['TR']==1)]
arows=[]
for r in tc.itertuples():
    e=EP.get((r.x_pdb,r.x_chain))
    can=e['pdbx_seq_one_letter_code_can'].replace('\n','') if e else ''
    raw=e['pdbx_seq_one_letter_code'].replace('\n','') if e else ''
    vs=ch[(ch.pdb==r.x_pdb)&(ch.chain==r.x_chain)].vseq.iat[0]
    off=can.find(vs[:15])
    arows.append(dict(pdb=r.x_pdb,chain=r.x_chain,seq_first_imgt=r.seq_first_imgt,S=r.x_S,S_sabr=r.x_S_sabr,entity_start=can[:12],
                      residues_before_V=off,modified_Nterm=raw[:12].startswith('('),author=r.last_author,year=r.dep_year))
A=pd.DataFrame(arows); A.to_csv(f'{OUT}/v10_case_entity_poly.csv',index=False); P(A.to_string())
P(f'entity sequence starts exactly at the V start (no tag/leader): {(A.residues_before_V==0).sum()}/{len(A)}; modified N-terminal residue: {A.modified_Nterm.sum()}')
from statsmodels.stats.multitest import multipletests
T_=pd.DataFrame(ROWS); ok=T_.p.notna(); T_.loc[ok,'q_BH_v10']=multipletests(T_.loc[ok,'p'],method='fdr_bh')[1]
T_.to_csv(f'{OUT}/v10_tests.csv',index=False)
P('\n=== BH over v10 tests'); P(T_[['test','n_err','n_ctrl','err_value','ctrl_value','OR','lo','hi','p','q_BH_v10']].round(4).to_string())
open(f'{OUT}/v10_independent.log','w').write('\n'.join(L)+'\n')
