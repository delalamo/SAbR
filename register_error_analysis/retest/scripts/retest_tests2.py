import json, re, collections, numpy as np, pandas as pd, statsmodels.formula.api as smf, warnings, sys
warnings.filterwarnings('ignore'); pd.set_option('display.width',250); pd.set_option('display.max_colwidth',70)
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
sys.path.insert(0,f'{SP}/scripts'); from sabr_eval import segments, region
M=pd.read_pickle(f'{SP}/retest/master2.pkl')
def rate(x): return f'{100*x.mean():.1f}% (n={len(x)})'
def fit(f,d):
    try: return smf.logit(f,data=d).fit(disp=0,cov_type='cluster',cov_kwds={'groups':pd.factorize(d.pdb)[0]})
    except Exception as e: print('   (not estimable:',str(e)[:70],')'); return None
def ors(m,filt):
    if m is None: return pd.DataFrame()
    p=m.params[m.params.index.str.contains(filt)]; c=m.conf_int().loc[p.index]
    return pd.DataFrame({'OR':np.exp(p),'lo':np.exp(c[0]),'hi':np.exp(c[1]),'p':m.pvalues[p.index]}).round(3)
def H(t): print('\n'+'='*100+'\n'+t+'\n'+'='*100)
ent=pd.read_pickle(f'{SP}/sabr/out/chains_flags.pkl')[['pdb','method','res','deposit','release','era','year']].drop_duplicates('pdb')
# ---------------------------------------------------------------- T1 TCR comparator
H('T1. Same fold, different practice: TCR V domains vs antibody V domains (SAbR TCR mode vs ANARCI)')
tcr=[json.loads(l) for l in open(f'{SP}/retest/tcr.jsonl')]
print(pd.Series([r['status'] if not r['status'].startswith('error') else 'error' for r in tcr]).value_counts().to_string())
trow=[]
for r in tcr:
    if r.get('status')!='ok': continue
    v=[x for x in r['residues'] if x[4] is not None]
    if len(v)<80: continue
    trow.append(dict(pdb=r['pdb'],chain=r['chain'],ctype='TCR-'+r['chain_type'],flag=int(len(segments(r['residues']))>0),nV=len(v)))
T=pd.DataFrame(trow)
meta={}
for l in open('/tmp/supp/meta2/meta_35.jsonl'):
    e=json.loads(l); meta[e['rcsb_id'].lower()]=(('EM' if 'ELECTRON MICROSCOPY' in [m['method'] for m in e['exptl']] else 'X-ray'),(e['rcsb_entry_info']['resolution_combined'] or [np.nan])[0],e['rcsb_accession_info']['deposit_date'][:10])
for l in open(f'{SP}/new/meta.jsonl'):
    e=json.loads(l); meta[e['rcsb_id'].lower()]=(('EM' if 'ELECTRON MICROSCOPY' in [m['method'] for m in (e.get('exptl') or [])] else 'X-ray'),(e['rcsb_entry_info']['resolution_combined'] or [np.nan])[0],e['rcsb_accession_info']['deposit_date'][:10])
T['method']=[meta.get(p,(None,))[0] for p in T.pdb]; T['res']=[meta.get(p,(None,np.nan))[1] for p in T.pdb]; T['year']=[int(meta.get(p,(None,None,'1900'))[2][:4]) for p in T.pdb]
A=M[M.single][['pdb','chain','ctype','flag','nV','method','res','year']]
B=pd.concat([A.assign(kind='antibody'),T.assign(kind='TCR')]); B['rb']=pd.cut(B.res,[2.999,3.25,3.5,3.75,4.0,5.0]).astype(str); B['logV']=np.log(B.nV)
B['ybin']=pd.cut(B.year,[0,2017,2019,2021,2023,2026]).astype(str)
print(B.groupby(['kind','method']).flag.agg(rate).unstack().to_string())
print(ors(fit("flag ~ C(kind) + C(method)*C(rb) + C(ybin) + logV",B.dropna(subset=['res'])),'kind').to_string())
# ---------------------------------------------------------------- T14 template status
H('T14. Does the error rate depend on whether the prior high-resolution template itself disagrees?')
tp=[json.loads(l) for l in open(f'{SP}/retest/templates.jsonl')]
tstat={}
for r in tp:
    if r.get('status')!='ok': continue
    S=segments(r['residues']); v=[x for x in r['residues'] if x[4] is not None]
    if len(v)<80: continue
    tstat[(r['pdb'],r['chain'])]=(int(len(S)>0),[(s['start'],s['delta']) for s in S],r['residues'])
print('templates scanned OK:',len(tstat),'| template flag rate:',f'{100*np.mean([v[0] for v in tstat.values()]):.1f}%')
import gzip
info={}
for l in open('/tmp/supp/pdbwide/entries.idx').read().splitlines()[2:]:
    f=l.split('\t')
    if len(f)<8: continue
    try: dt=pd.to_datetime(f[2],format='%m/%d/%y'); rr=float(f[6].split(',')[0])
    except Exception: continue
    info[f[0].lower()]=(dt,rr)
keymap=collections.defaultdict(list)
with gzip.open('/tmp/supp/pdbwide/pdb_seqres.txt.gz','rt') as fh:
    h=None; mol=None
    for line in fh:
        if line.startswith('>'): h=line[1:].split()[0]; mol=line.split()[1]
        elif mol=='mol:protein' and len(line)>105:
            p,c=h.split('_',1)
            if (p,c) in tstat: keymap[line[:105]].append((info.get(p,(pd.NaT,))[0],p,c))
segs=pd.read_pickle(f'{SP}/retest/segs.pkl'); segmap=dict(zip(zip(segs.pdb,segs.chain),segs.segs))
rows=[]
for r in M[(M.prior_hires==1)&M.single].itertuples():
    cands=sorted([x for x in keymap.get(r.seq[:105],[]) if x[0]<pd.Timestamp(r.deposit)])
    if not cands: continue
    tf=[tstat[(p,c)][0] for _,p,c in cands]
    # same error: any template run overlapping the same IMGT start (±2) with the same delta
    def runs_imgt(res):
        out=[];cur=[]
        for x in [x for x in res if x[4] is not None]+[None]:
            if x is not None and (x[4],x[5] or '')!=(x[6],x[7] or ''): cur.append(x)
            else:
                if len(cur)>=3: out.append((cur[0][4],int(np.median([y[6]-y[4] for y in cur if not (y[5] or y[7])] or [0]))))
                cur=[]
        return out
    trun=set(); [trun.update(runs_imgt(tstat[(p,c)][2])) for _,p,c in cands]
    rows.append(dict(pdb=r.pdb,chain=r.chain,flag=r.flag,era=r.era,method=r.method,rb=r.rb,ybin=r.ybin,ctype=r.ctype,logV=r.logV,lq_z=r.lq_z,
        template_flag=int(any(tf)),n_templates=len(cands),pdbkey=r.seq[:105]))
TT=pd.DataFrame(rows)
print('chains with scanned prior template:',len(TT))
print(TT.groupby(['era','template_flag']).flag.agg(rate).unstack().to_string())
print(ors(fit("flag ~ template_flag + C(method)*C(rb) + C(ybin) + C(ctype) + logV",TT),'template').to_string())
# chain flagged with same error (position & shift) as template
recs={}
for f in ['validation','snapshot_rest','new']:
    for l in open(f'{SP}/sabr/out/{f}.jsonl'):
        r=json.loads(l)
        if r.get('status')=='ok': recs[(r['pdb'],r['chain'])]=r['residues']
def runs_imgt(res):
    out=[];cur=[]
    for x in [x for x in res if x[4] is not None]+[None]:
        if x is not None and (x[4],x[5] or '')!=(x[6],x[7] or ''): cur.append(x)
        else:
            if len(cur)>=3: out.append((cur[0][4],int(np.median([y[6]-y[4] for y in cur if not (y[5] or y[7])] or [0]))))
            cur=[]
    return out
same=0; tot=0
for r in TT[TT.flag==1].itertuples():
    cands=sorted([x for x in keymap.get(r.pdbkey,[])])
    tr=set(); [tr.update(runs_imgt(tstat[(p,c)][2])) for _,p,c in cands]
    cr=runs_imgt(recs.get((r.pdb,r.chain),[]))
    tot+=1; same+=int(any(abs(a-b)<=2 and d==e for a,d in cr for b,e in tr))
print('flagged chains with a scanned template: %d; same error (IMGT start ±2, same shift) present in the template: %d (%.0f%%)'%(tot,same,100*same/max(1,tot)))
# ---------------------------------------------------------------- T15 recorded starting model = flagged template?
H('T15. Entries that record a starting model: is the cited model a flagged structure of the same antibody?')
mod={}
for l in open(f'{SP}/retest/modelling.jsonl'):
    e=json.loads(l); mod[e['rcsb_id'].lower()]=[(x.get('accession_code') or '').lower() for x in (e.get('pdbx_initial_refinement_model') or [])]
flag_entry_key={}
for r in M.itertuples(): flag_entry_key[(r.pdb,r.seq[:105])]=max(flag_entry_key.get((r.pdb,r.seq[:105]),0),r.flag)
for (p,c),v in tstat.items(): pass
tkey={}
for k,lst in keymap.items():
    for _,p,c in lst: tkey[(p,k)]=max(tkey.get((p,k),0),tstat[(p,c)][0])
rows=[]
for r in M[M.single].itertuples():
    k=r.seq[:105]; cited=mod.get(r.pdb,[])
    st=[flag_entry_key.get((a,k),tkey.get((a,k))) for a in cited]; st=[s for s in st if s is not None]
    if st: rows.append(dict(flag=r.flag,cited_flagged=int(max(st)),era=r.era))
CI=pd.DataFrame(rows); print('chains whose entry cites an earlier structure of the same V region as starting model:',len(CI))
print(CI.groupby('cited_flagged').flag.agg(rate).to_string())
# ---------------------------------------------------------------- T16 mechanism
H('T16. Mechanism: where and next to what do the disagreements occur?')
rows=[]; ctrl=[]
rng=np.random.default_rng(1)
for r in M[M.single].itertuples():
    res=recs.get((r.pdb,r.chain))
    if not res: continue
    v=[x for x in res if x[4] is not None and x[3] is not None]
    if len(v)<80: continue
    si=[x[3] for x in v]; gaps={i for i in range(len(si)-1) if si[i+1]-si[i]>1}
    nterm_missing=v[0][4]>3   # first modelled residue is beyond IMGT 3
    def near_gap(i0,i1): return any(i0-3<=g<=i1+3 for g in gaps)
    S=segments(res)
    if S:
        idx={x[0]:i for i,x in enumerate(v)}
        for s in S:
            i0=idx.get(s['start']); i1=idx.get(s['end'])
            if i0 is None or i1 is None: continue
            rows.append(dict(kind='disagreement run',near_gap=near_gap(i0,i1),at_nterm=i0<=2,nterm_missing=nterm_missing,loc=region(v[i0][4]) if True else None,imgt0=v[i0][4],length=i1-i0+1,chain_type=r.ctype))
    else:
        L=rng.integers(3,12); i0=int(rng.integers(0,len(v)-L)); i1=i0+L-1
        ctrl.append(dict(kind='random window, clean V domain',near_gap=near_gap(i0,i1),at_nterm=i0<=2,nterm_missing=nterm_missing,imgt0=v[i0][4],length=L,chain_type=r.ctype))
D=pd.DataFrame(rows+ctrl)
print(D.groupby('kind')[['near_gap','at_nterm']].mean().round(3).to_string())
# matched on IMGT start: control windows starting within ±5 IMGT of each run
runs=D[D.kind=='disagreement run']; cw=D[D.kind!='disagreement run']
mt=[cw[(cw.imgt0-r.imgt0).abs()<=5].near_gap.mean() for r in runs.itertuples()]
print('near an unmodelled gap (±3 residues): runs %.1f%% vs position-matched clean windows %.1f%%'%(100*runs.near_gap.mean(),100*np.nanmean(mt)))
print('V domains whose N-terminus is unmodelled (first residue beyond IMGT 3): flag rate %s vs complete N-terminus %s'%(rate(M[M.single].flag[[ (recs.get((p,c)) and [x for x in recs[(p,c)] if x[4] is not None] and [x for x in recs[(p,c)] if x[4] is not None][0][4]>3) or False for p,c in zip(M[M.single].pdb,M[M.single].chain)]]),
      rate(M[M.single].flag[[ not ((recs.get((p,c)) and [x for x in recs[(p,c)] if x[4] is not None] and [x for x in recs[(p,c)] if x[4] is not None][0][4]>3) or False) for p,c in zip(M[M.single].pdb,M[M.single].chain)]])))
print('\n-- CDR3 length (heavy) and CDR1 length (light), flag rate by length tertile')
Hh=M[(M.ctype=='H')&M.cdr3_len.notna()]; Hh['cdr3_bin']=pd.qcut(Hh.cdr3_len,3,duplicates='drop').astype(str)
print(Hh.groupby('cdr3_bin').flag.agg(rate).to_string())
Ll=M[(M.ctype.isin(['K','L']))&M.cdr1_len.notna()]; Ll['cdr1_bin']=Ll.cdr1_len.astype(int).astype(str)
print(Ll.groupby(['ctype','cdr1_bin']).flag.agg(rate).unstack(0).to_string())
SG=pd.read_pickle(f'{SP}/sabr/out/segments.pkl')
fr4=SG[(SG.chain_type=='H')]
Hh2=Hh.copy(); Hh2['fr4_run']=Hh2.set_index(['pdb','chain']).index.isin(set(zip(fr4[fr4['loc'].isin(['FR4','CDR3'])].pdb,fr4[fr4['loc'].isin(['FR4','CDR3'])].chain)))
print('heavy chains with a CDR3/FR4 run, by CDR3 length tertile:'); print(Hh2.groupby('cdr3_bin').fr4_run.agg(rate).to_string())
# ---------------------------------------------------------------- T18 conkit artifact
H('T18. Was the low pre-2018 conkit rate an AlphaFold2 training-set artifact? conkit sensitivity relative to SAbR by release date')
t=pd.read_excel('/tmp/supp/epmc/m-11-00938-sup2.xlsx'); t[['pdb','chain']]=t.PDB_CHAIN_ID.str.split(':',expand=True); t['pdb']=t.pdb.str.lower()
rr=t.RESIDUE_RANGE.str.extract(r'(-?\d+)\s*-\s*(-?\d+)').astype(int); t['rs'],t['re']=rr[0],rr[1]
ck=t.groupby(['pdb','chain']).apply(lambda g: set().union(*[set(range(a,b+1)) for a,b in zip(g.rs,g.re)])).to_dict()
snap=M[M.era.str.startswith('snap')&M.single].copy()
snap['release']=pd.to_datetime(snap.release)
fr=[]
for r in snap[snap.flag==1].itertuples():
    for s in segmap.get((r.pdb,r.chain),[]):
        if s['delta'] is None: continue
        fr.append(dict(pdb=r.pdb,chain=r.chain,method=r.method,pre=r.release<pd.Timestamp('2018-04-30'),conkit=bool(ck.get((r.pdb,r.chain),set()) & set(range(s['start'],s['end']+1)))))
F=pd.DataFrame(fr)
print('SAbR framework-shift runs in the snapshot also flagged by conkit, by release relative to the AF2 training cutoff (2018-04-30):')
print(F.groupby(['method','pre']).conkit.agg(rate).unstack().to_string())
print(F.groupby('pre').conkit.agg(rate).to_string())
print('SAbR flag rate (single-domain, X-ray) released before vs after cutoff:',rate(snap[(snap.method=='X-ray')&(snap.release<'2018-04-30')].flag),rate(snap[(snap.method=='X-ray')&(snap.release>='2018-04-30')].flag))
