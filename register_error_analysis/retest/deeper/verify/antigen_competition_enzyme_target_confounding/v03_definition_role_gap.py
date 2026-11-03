# Further checks for the enzyme_target finding: outcome definition (any run vs first-occurrence), antibody role,
# chain-level unbuilt residues (confounder or mediator?), known starting-model/rigid-body mechanism, mode mix by antibody.
import warnings, numpy as np, pandas as pd, statsmodels.formula.api as smf, statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings('ignore')
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
AC='/home/user/SAbR/register_error_analysis/retest/deeper/antigen_competition/tables'
OUT='/home/user/SAbR/register_error_analysis/retest/deeper/verify/antigen_competition_enzyme_target_confounding'
pd.set_option('display.width',250)
A=pd.read_csv(f'{AC}/pdbwide_ab_entries_antigen.csv.gz',usecols=['pdb','tgt_cls','ab_status'])
e=pd.read_csv(f'{SP}/deeper/entries.csv')
amy=set(pd.read_csv('/home/user/SAbR/register_error_analysis/retest/deeper/pdb_meta/tables/amyloid_fibril_chains.csv').pdb)
e=e.merge(A,on='pdb',how='left'); e=e[~e.pdb.isin(amy)].copy()
e['enz']=e.tgt_cls.isin(['enzyme','viral other']).astype(int)
e['dyear']=pd.to_datetime(e.deposit).dt.year; e['ycat']=e.dyear.clip(lower=2017).astype(str)
e['paper']=e.doi.fillna(e.last_author.fillna('')+'_'+e.dyear.astype(str))
TESTS=[]
def logit(d,term,extra,label,yv='y',cl='paper'):
    d=d.copy(); d['y']=d[yv].astype(int)
    a=int(((d[term]==1)&(d.y==1)).sum()); b=int(((d[term]==1)&(d.y==0)).sum()); c_=int(((d[term]==0)&(d.y==1)).sum()); dd=int(((d[term]==0)&(d.y==0)).sum())
    row=dict(test=label,rate_in=f'{a}/{a+b} ({100*a/max(a+b,1):.1f}%)',rate_out=f'{c_}/{c_+dd} ({100*c_/max(c_+dd,1):.1f}%)',n=len(d),n_err=int(d.y.sum()))
    if a<3 or b<3:
        p=stats.fisher_exact([[a,b],[c_,dd]])[1]; row.update(OR=((a+.5)*(dd+.5))/((b+.5)*(c_+.5)),lo=np.nan,hi=np.nan,p=p,model='Fisher')
    else:
        mod=smf.glm(f'y ~ {term} + {extra}',d,family=sm.families.Binomial()).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(d[cl])[0]})
        ci=mod.conf_int().loc[term]; row.update(OR=np.exp(mod.params[term]),lo=np.exp(ci[0]),hi=np.exp(ci[1]),p=mod.pvalues[term],model=extra)
    TESTS.append(row)
    print(f"  {label[:90]:90s} {row['rate_in']:>16s} vs {row['rate_out']:>18s} OR {row['OR']:.2f} ({row['lo']:.2f}-{row['hi']:.2f}) p={row['p']:.2g}")
    return row
EM=e[e.method=='EM']
print('== 1. Outcome definition: any disagreement run (copied/artifact runs included), all EM entries')
logit(EM,'enz','C(ycat) + res','1a any run (err_any), all EM entries',yv='err_any')
x=EM[EM.err_first|~EM.err_any]; logit(x,'enz','C(ycat) + res','1b first-occurrence (lens definition, reproduced from entries.csv)',yv='err_first')
x=EM[~EM.err_first]; logit(x,'enz','C(ycat) + res','1c copied/artifact-only runs vs clean (entries without a first-occurrence error)',yv='err_any')

# ---- chain level
ch=pd.read_csv(f'{SP}/deeper/chains.csv.gz').merge(pd.read_csv(f'{SP}/deeper/cache/verify_enzyme_confounding/chain_gaps.csv.gz'),on=['pdb','chain'],how='left')
ch=ch[(ch.method=='EM')&~ch.pdb.isin(amy)&~ch.err_copied_or_artifact_only].merge(e[['pdb','enz','ycat','tgt_cls','ab_status']],on='pdb',how='inner')
ch['y']=ch.err_first.astype(int); ch['gap']=ch.any_gap.fillna(False).astype(int); ch['rigid']=(ch.fit=='rigid body only').astype(int)
ch['expmodel']=(ch.irm=='experimental (PDB) model').astype(int); ch['aux']=(ch.role=='Auxiliary').astype(int)
ch['kind']=np.where(ch.vhh==1,'VHH',np.where(ch.ctype=='H','Fab heavy','light'))
print('\n== 2. Antibody role (chains.csv role; EM chains)')
print(ch.groupby(['enz','role']).y.agg(['mean','sum','size']).round(3).to_string())
logit(ch,'enz','aux + C(kind) + C(ycat) + res','2a chain level: enzyme adjusted for auxiliary role + chain kind')
logit(ch[ch.aux==1],'enz','C(kind) + C(ycat) + res','2b chain level within auxiliary-role antibodies')
logit(ch[ch.role=='Antibody of interest'],'enz','C(kind) + C(ycat) + res','2c chain level within antibody-of-interest')
print('\n== 3. Unbuilt residues in the same chain: confounder or mediator?')
print(ch.groupby(['enz','gap']).y.agg(['mean','sum','size']).round(3).to_string())
cl=ch[ch.y==0]
print(f'  clean chains with an unbuilt V stretch: enzyme {cl[cl.enz==1].gap.mean():.3f} (n={int((cl.enz==1).sum())}) vs other {cl[cl.enz==0].gap.mean():.3f}; Fisher p={stats.fisher_exact(pd.crosstab(cl.enz,cl.gap).values)[1]:.2g}')
ce=e[(e.method=='EM')&~e.err_any]
logit(ch[ch.gap==0],'enz','C(kind) + C(ycat) + res','3a chain level, chains with NO unbuilt V residue')
logit(ch[ch.gap==1],'enz','C(kind) + C(ycat) + res','3b chain level, chains WITH an unbuilt V stretch or N-terminal truncation')
mod=smf.glm('y ~ enz*gap + C(kind) + C(ycat) + res',ch,family=sm.families.Binomial()).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(ch.paper)[0]})
print(f"  interaction enz:gap OR {np.exp(mod.params['enz:gap']):.2f} p={mod.pvalues['enz:gap']:.2g}")
TESTS.append(dict(test='3c chain level enzyme x unbuilt interaction',OR=np.exp(mod.params['enz:gap']),p=mod.pvalues['enz:gap'],n=len(ch),n_err=int(ch.y.sum()),model='interaction'))
# run level: fraction of first-occurrence runs whose origin is 'chain end or next to an unbuilt loop'
r=pd.read_csv(f'{SP}/deeper/runs_first_occurrence.csv'); r=r[(r.method=='EM')&~r.pdb.isin(amy)].merge(e[['pdb','enz']],on='pdb',how='inner')
t=pd.crosstab(r.enz,r.next_to_gap_origin); print('  runs next to gap/chain end (rows enz):'); print(t)
TESTS.append(dict(test='3d run level: next-to-gap origin, enzyme vs other EM runs',OR=stats.fisher_exact(t.values)[0],p=stats.fisher_exact(t.values)[1],n=int(t.values.sum()),n_err=int(t.values[1].sum()),model='Fisher',rate_in=f'{t.values[1,1]}/{t.values[1].sum()}',rate_out=f'{t.values[0,1]}/{t.values[0].sum()}'))
print('\n== 4. Known starting-model / rigid-body mechanism')
print(ch.groupby(['enz','rigid']).y.agg(['mean','sum','size']).round(3).to_string())
logit(ch[ch.rigid==0],'enz','C(kind) + C(ycat) + res','4a chain level, excluding rigid-body-only fitted entries')
logit(ch[(ch.rigid==0)&(ch.expmodel==0)],'enz','C(kind) + C(ycat) + res','4b chain level, excluding rigid-body-only and experimental-starting-model entries')
# entry level with CDR3/FR4 junction modes (D+/D-) removed from the outcome (those entries dropped)
x=EM[(EM.err_first|~EM.err_any)].copy(); x['y']=x.err_first.astype(int)
dmode=x.modes.fillna('').str.contains('Dplus|Dminus')&~x.modes.fillna('').str.contains('A_|B_|C_|E_|F_|other')
logit(x[~dmode],'enz','C(ycat) + res','4c entry level, entries whose only errors are CDR3/FR4 junction slips removed')
print('\n== 5. Mode mix by distinct antibody (enzyme error chains)')
ee=ch[(ch.enz==1)&(ch.y==1)]
print(ee.groupby('err_modes').agg(entries=('pdb','nunique'),antibodies=('vseq','nunique'),labs=('last_author','nunique')).to_string())
oth=ch[(ch.enz==0)&(ch.y==1)]
for md in ['B_CDR2FR3','A_FR1Nterm','Dplus|Dminus']:
    a=ee.vseq[ee.err_modes.fillna('').str.contains(md)].nunique(); A_=ee.vseq.nunique(); b=oth.vseq[oth.err_modes.fillna('').str.contains(md)].nunique(); B_=oth.vseq.nunique()
    p=stats.fisher_exact([[a,A_-a],[b,B_-b]])[1]
    print(f'  {md}: enzyme {a}/{A_} distinct erroneous V regions vs other {b}/{B_}; Fisher p={p:.2g}')
    TESTS.append(dict(test=f'5 mode {md} share among distinct erroneous V regions, enzyme vs other EM',rate_in=f'{a}/{A_}',rate_out=f'{b}/{B_}',OR=((a+.5)*(B_-b+.5))/((A_-a+.5)*(b+.5)),p=p,n=A_+B_,n_err=a+b,model='Fisher'))
R=pd.DataFrame(TESTS); R['q_BH_within_file']=multipletests(R.p.fillna(1),method='fdr_bh')[1]
R.to_csv(f'{OUT}/tests_definition_role_gap.csv',index=False)
