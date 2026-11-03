"""'Canonical pull' test. For each chain, CDR lengths implied by the deposited structure (SAbR structural IMGT numbering,
modelled residues) vs by the sequence (ANARCI IMGT, same modelled residues). Two parts:
 (a) detector-bias control: in chains with NO disagreement run (err_any False) do SAbR and ANARCI lengths agree at
     every length, including atypical ones?  (If SAbR itself pulled atypical loops to common lengths, it would show here.)
 (b) in first-occurrence error chains, for fully modelled CDRs touched by a run with |apparent-true| in {1,2}: is the
     apparent length more frequent in the curated reference (X-ray <=3.0 A unique V domains, <=2019) than the true one
     more often than a random direction of the same size would give?"""
import json, pickle, numpy as np, pandas as pd
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
OUT='/home/user/SAbR/register_error_analysis/retest/deeper/literature_dean/tables'
R=pickle.load(open(f'{SP}/deeper/cache/literature_dean/ref.pkl','rb'))
ref=R[(R.hq30)&(R.first_year<=2019)]
F={}
for ct in 'HKL':
    for k,col in [('1','len_CDR1'),('2','len_CDR2'),('3','len_CDR3')]:
        v=ref[ref.ctype==ct][col]; F[(ct,k)]=(v.value_counts()/len(v)).to_dict(); F[(ct,k,'n')]=len(v)
def fr(ct,k,L): return F[(ct,k)].get(L,0.5/F[(ct,k,'n')])
runs=pd.read_csv(f'{SP}/deeper/runs_first_occurrence.csv')
c=pd.read_pickle(f'{SP}/deeper/cache/literature_dean/chains_aug.pkl')[['pdb','chain','ctype','method','unit','err_first','err_any','rL1','rL2','rL3','year']]
key=set(zip(c.pdb,c.chain))
REG={'1':(27,38),'2':(56,65),'3':(105,117)}
rows=[]
for fn in ['new','snapshot_rest','validation']:
    for l in open(f'{SP}/sabr/out/{fn}.jsonl'):
        if '"status": "ok"' not in l: continue
        d=json.loads(l)
        if (d['pdb'],d['chain']) not in key: continue
        res=d['residues']; o=dict(pdb=d['pdb'],chain=d['chain'])
        for k,(a,b) in REG.items():
            o['an'+k]=sum(1 for r in res if r[4] is not None and a<=r[4]<=b)
            o['sa'+k]=sum(1 for r in res if r[6] is not None and a<=r[6]<=b)
        rows.append(o)
L=pd.DataFrame(rows).drop_duplicates(['pdb','chain']).merge(c,on=['pdb','chain'])
L.to_csv(f'{OUT}/apparent_cdr_lengths_all_chains.csv.gz',index=False)
out=open(f'{OUT}/apparent_length.log','w')
def P(*a):
    s=' '.join(str(x) for x in a); print(s); out.write(s+'\n')
# (a) detector bias control
P('(a) Clean chains (no disagreement run of any kind), fully modelled CDR: SAbR length == ANARCI length')
bias=[]
for k in '123':
    cl=L[(~L.err_any)&(L['an'+k]==L['rL'+k])]
    for (ct,ln),g in cl.groupby(['ctype','an'+k]):
        if len(g)<15: continue
        agree=(g['sa'+k]==g['an'+k]).mean()
        bias.append(dict(ctype=ct,cdr=k,length=ln,n_clean=len(g),agree=agree,ref_freq=fr(ct,k,ln),
                         toward_more_common=((g['sa'+k]!=g['an'+k]).values&np.array([fr(ct,k,s)>fr(ct,k,ln) for s in g['sa'+k]])).mean()))
B=pd.DataFrame(bias); B.to_csv(f'{OUT}/sabr_length_bias_clean_chains.csv',index=False)
P(B.round(3).to_string())
P('overall agreement in clean chains (weighted):',round((B.agree*B.n_clean).sum()/B.n_clean.sum(),4),
  '| at reference-rare lengths (<5%):',round((B[B.ref_freq<0.05].agree*B[B.ref_freq<0.05].n_clean).sum()/max(1,B[B.ref_freq<0.05].n_clean.sum()),4))
# (b) error chains
E=L[L.err_first].copy()
rr=[]
for _,r in E.iterrows():
    g=runs[(runs.pdb==r.pdb)&(runs.chain==r.chain)]
    for k,(a,b) in REG.items():
        if not ((g.imgt0<=b+2)&(g.imgt1>=a-2)).any(): continue
        an,sa,tru=r['an'+k],r['sa'+k],r['rL'+k]
        if an!=tru: continue               # CDR not fully modelled
        dlt=sa-an
        if abs(dlt) not in (1,2): continue
        rr.append(dict(pdb=r.pdb,chain=r.chain,unit=r.unit,method=r.method,year=r.year,ctype=r.ctype,cdr=k,true_len=an,app_len=sa,diff=dlt,
                       modes=';'.join(sorted(set(g['mode']))),f_true=fr(r.ctype,k,an),f_app=fr(r.ctype,k,sa),
                       f_other=fr(r.ctype,k,an-dlt)))
A=pd.DataFrame(rr).sort_values(['unit','cdr']).drop_duplicates(['unit','cdr'])
A['more_common']=(A.f_app>A.f_true).astype(int)
A['p_exp']=0.5*((A.f_app>A.f_true).astype(int)+(A.f_other>A.f_true).astype(int))
A['app_is_mode']=[int(r.app_len==max(F[(r.ctype,r.cdr)],key=F[(r.ctype,r.cdr)].get)) for r in A.itertuples()]
A['other_is_mode']=[int(r.true_len-r['diff']==max(F[(r.ctype,r.cdr)],key=F[(r.ctype,r.cdr)].get)) if False else int((r.true_len-r.diff)==max(F[(r.ctype,r.cdr)],key=F[(r.ctype,r.cdr)].get)) for r in A.itertuples()]
A.to_csv(f'{OUT}/apparent_length_error_units.csv',index=False)
P('\n(b) Error units x CDR (fully modelled CDR touched by a run, |apparent-true| 1-2):',len(A))
rng=np.random.default_rng(1)
summ=[]
for lab,sub in [('all',A),('EM only',A[A.method=='EM']),('X-ray only',A[A.method!='EM'])]+[(f'CDR{k}',A[A.cdr==k]) for k in '123']+[(f'chain {t}',A[A.ctype==t]) for t in 'HKL']+[('H CDR3',A[(A.ctype=='H')&(A.cdr=='3')]),('light CDR1',A[(A.ctype!='H')&(A.cdr=='1')]),('heavy CDR2',A[(A.ctype=='H')&(A.cdr=='2')])]:
    if len(sub)<3: continue
    o=int(sub.more_common.sum()); e=sub.p_exp.sum(); n=len(sub)
    sims=rng.binomial(1,np.tile(sub.p_exp.values,(50000,1))).sum(1)
    p=min(1.0,2*min((sims>=o).mean(),(sims<=o).mean())); p=max(p,1/50000)
    # modal snapping: apparent length is the reference mode
    om=int(sub.app_is_mode.sum()); em=0.5*(sub.app_is_mode+sub.other_is_mode).sum()
    sims2=rng.binomial(1,np.tile((0.5*(sub.app_is_mode+sub.other_is_mode)).values,(50000,1))).sum(1)
    p2=max(min(1.0,2*min((sims2>=om).mean(),(sims2<=om).mean())),1/50000)
    summ.append(dict(subset=lab,n=n,obs_more_common=o,exp_more_common=round(e,1),p_more_common=p,obs_to_mode=om,exp_to_mode=round(em,1),p_to_mode=p2))
S=pd.DataFrame(summ); S.to_csv(f'{OUT}/canonical_pull_tests.csv',index=False)
P(S.to_string())
P('\nTransitions (true -> apparent) with n>=2:')
t=A.groupby(['ctype','cdr','true_len','app_len']).agg(n=('unit','size'),f_true=('f_true','first'),f_app=('f_app','first')).reset_index()
P(t[t.n>=2].round(3).to_string())
out.close()
