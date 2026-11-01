import json, sys, re, gzip, collections, numpy as np, pandas as pd, statsmodels.formula.api as smf, warnings
warnings.filterwarnings('ignore'); pd.set_option('display.width',250); pd.set_option('display.max_colwidth',60)
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
sys.path.insert(0,f'{SP}/scripts'); sys.path.insert(0,'/tmp/scripts')
from sabr_eval import segments, region
from abrole import classify
from tiers import tier as TIER
O=f'{SP}/sabr/out'
import os
def load(fn): return [json.loads(l) for l in open(fn)] if os.path.exists(fn) else []
old=[r for f in ['validation.jsonl','snapshot_rest.jsonl','validation_multi.jsonl','snapshot_rest_multi.jsonl'] for r in load(f'{O}/{f}')]
new=load(f'{O}/new.jsonl')+load(f'{O}/new_multi.jsonl')
# ---------- chain-level table
an_all=json.load(open('/tmp/supp/meta2/anarci_all.json')); an_new=json.load(open(f'{SP}/new/anarci_new_screen.json'))
def isab(s): return any(d['chain_type'] in 'HKL' for d in (an_all.get(s) or an_new.get(s) or []))
def chain_rows(recs,era):
    out=[]
    for r in recs:
        if r.get('status')!='ok': continue
        S=segments(r['residues'],minlen=3)
        v=[x for x in r['residues'] if x[4] is not None]
        if len(v)<80: continue   # require a mostly modelled V domain
        out.append(dict(pdb=r['pdb'],chain=r['chain'],era=era,chain_type=r['chain_type'],nV=len(v),flag=int(len(S)>0),nseg=len(S),
            segs=S,seg_regions='+'.join(sorted({g for s in S for g in s['regions']}))))
    return out
CH=pd.DataFrame(chain_rows(old,'snapshot (released ≤ Apr 2022)')+chain_rows(new,'new (released after Apr 2022)'))
# ---------- entry metadata
ent_old=pd.read_pickle('/tmp/analysis/entries35.pkl')[['pdb','method','res','deposit','release','role','sars2','journal','last_author','cite_key','title']]
rows=[]
for l in open(f'{SP}/new/meta.jsonl'):
    e=json.loads(l); p=e['rcsb_id'].lower(); ents=[]; sars=False
    for pe in e['polymer_entities'] or []:
        if pe['entity_poly']['type']!='polypeptide(L)': continue
        s=(pe['entity_poly']['pdbx_seq_one_letter_code_can'] or '').replace('\n','')
        orgs=' '.join(o['scientific_name'] or '' for o in (pe['rcsb_entity_source_organism'] or []))
        tax={o['ncbi_taxonomy_id'] for o in (pe['rcsb_entity_source_organism'] or [])}
        ab=isab(s)
        if not ab and 2697049 in tax: sars=True
        ents.append(dict(desc=pe['rcsb_polymer_entity']['pdbx_description'],ab=ab,tcr=False,length=len(s),orgs=orgs))
    kw=(e['struct_keywords'] or {}).get('pdbx_keywords') or ''
    meth=[m['method'] for m in (e.get('exptl') or [])]
    c=e['rcsb_primary_citation'] or {}
    last=(sorted(e['audit_author'] or [],key=lambda a:a['pdbx_ordinal'])[-1]['name'] if e.get('audit_author') else None)
    rows.append(dict(pdb=p,method='EM' if 'ELECTRON MICROSCOPY' in meth else 'X-ray',res=(e['rcsb_entry_info']['resolution_combined'] or [np.nan])[0],
        deposit=pd.to_datetime(e['rcsb_accession_info']['deposit_date'][:10]),release=pd.to_datetime((e['rcsb_accession_info']['initial_release_date'] or '')[:10] or None),
        role=classify(e['struct']['title'],kw,ents),sars2=sars,journal=c.get('journal_abbrev'),last_author=last,
        cite_key=c.get('pdbx_database_id_DOI') or (('PMID'+str(c['pdbx_database_id_PubMed'])) if c.get('pdbx_database_id_PubMed') else p),title=e['struct']['title']))
ent=pd.concat([ent_old,pd.DataFrame(rows)],ignore_index=True).drop_duplicates('pdb')
ent['year']=pd.to_datetime(ent.deposit).dt.year; ent['tier']=ent.journal.map(TIER)
CH=CH.merge(ent,on='pdb',how='left')
CH['rb']=pd.cut(CH.res,[2.999,3.25,3.5,3.75,4.0,5.0]).astype(str)
CH['ybin']=pd.cut(CH.year,[0,2017,2019,2021,2023,2026],labels=['≤2017','2018–19','2020–21','2022–23','2024–26']).astype(str)
# ---------- prior high-res template (identical first 105 aa of the chain, deposited earlier, <3 A)
seqs={}
for f in ['/tmp/supp/rcsb/seqs.jsonl',f'{SP}/new/meta.jsonl']:
    for l in open(f):
        e=json.loads(l); p=e['rcsb_id'].lower()
        for pe in e['polymer_entities'] or []:
            s=(pe['entity_poly'].get('pdbx_seq_one_letter_code_can') or '').replace('\n','')
            for c in pe['rcsb_polymer_entity_container_identifiers']['auth_asym_ids'] or []: seqs[(p,c)]=s
info={}
for l in open('/tmp/supp/pdbwide/entries.idx').read().splitlines()[2:]:
    f=l.split('\t')
    if len(f)<8: continue
    try: dt=pd.to_datetime(f[2],format='%m/%d/%y'); r=float(f[6].split(',')[0])
    except Exception: continue
    info[f[0].lower()]=(dt,r)
best=collections.defaultdict(lambda: pd.Timestamp('2100-01-01'))
with gzip.open('/tmp/supp/pdbwide/pdb_seqres.txt.gz','rt') as fh:
    h=None; mol=None
    for line in fh:
        if line.startswith('>'): h=line[1:].split()[0].split('_')[0]; mol=line.split()[1]
        elif mol=='mol:protein' and len(line)>105 and h in info and info[h][1]<3.0:
            k=line[:105]
            if info[h][0]<best[k]: best[k]=info[h][0]
CH['seq']=[seqs.get((p,c),'') for p,c in zip(CH.pdb,CH.chain)]
CH['prior_hires']=[int(len(s)>=105 and best.get(s[:105],pd.Timestamp('2100-01-01'))<pd.Timestamp(d)) for s,d in zip(CH.seq,CH.deposit)]
CH.to_pickle(f'{O}/chains_flags.pkl')
def rate(s): return f'{100*s.flag.mean():.1f}% ({int(s.flag.sum())}/{len(s)})'
print('== V domains analysed:',len(CH),' entries:',CH.pdb.nunique()); print(CH.groupby('era').apply(rate).to_string())
print('\n== By deposition period x method (V domains with ≥1 SAbR/ANARCI disagreement run)')
print(CH.pivot_table(index='ybin',columns='method',values='flag',aggfunc=lambda x: f'{100*x.mean():.1f}% (n={len(x)})').to_string())
print('\n== By method x resolution, new vs snapshot')
print(CH.pivot_table(index=['method','rb'],columns='era',values='flag',aggfunc=lambda x: f'{100*x.mean():.1f}% (n={len(x)})').to_string())
def fit(f,d):
  try:
    return smf.logit(f,data=d).fit(disp=0,cov_type='cluster',cov_kwds={'groups':pd.factorize(d.pdb)[0]})
  except Exception as e:
    print('   (model not estimable:',str(e)[:80],')'); return None
def ors(m,filt):
    if m is None: return pd.DataFrame()
    p=m.params[m.params.index.str.contains(filt)]; c=m.conf_int().loc[p.index]
    return pd.DataFrame({'OR':np.exp(p),'lo':np.exp(c[0]),'hi':np.exp(c[1]),'p':m.pvalues[p.index]}).round(3)
CH['ctype']=CH.chain_type.where(CH.chain_type.isin(['H','K','L']),'multi-domain (scFv etc.)'); CH['logV']=np.log(CH.nV); D=CH.dropna(subset=['res','year']).copy()
print('\n== Multi-domain (scFv etc.) vs single-domain chains, by era x method')
print(D.assign(md=D.ctype.str.startswith('multi')).pivot_table(index=['era','method'],columns='md',values='flag',aggfunc=lambda x: f'{100*x.mean():.1f}% (n={len(x)})').to_string())
print('\n== Adjusted model, all V domains (period vs 2018–19; method x resolution, chain type, V length)')
m=fit("flag ~ C(ybin,Treatment('2018–19')) + C(method)*C(rb) + C(ctype) + logV + prior_hires",D); print(ors(m,'ybin|prior|ctype').to_string())
print('\n== EM vs X-ray at matched resolution, by era')
for era,s in D.groupby('era'):
    m=fit("flag ~ C(method,Treatment('X-ray')) + C(rb) + C(ctype) + logV",s); print(era); print(ors(m,'method').to_string())
print('\n== SARS-CoV-2 complexes vs other antibody entries, by period (EM only)')
E=D[D.method=='EM']
print(E.pivot_table(index='ybin',columns='sars2',values='flag',aggfunc=lambda x: f'{100*x.mean():.1f}% (n={len(x)})').to_string())
print('\n== Antibody role (EM), by era')
R=D[D.role.isin(['Auxiliary','Antibody of interest'])]
print(R.pivot_table(index=['era','method'],columns='role',values='flag',aggfunc=lambda x: f'{100*x.mean():.1f}% (n={len(x)})').to_string())
for era,s in R.groupby('era'):
    s=s.copy(); s['aux']=(s.role=='Auxiliary').astype(int)
    m=fit("flag ~ aux + prior_hires + C(method)*C(rb) + C(ctype) + logV",s); print(era,'adjusted:'); print(ors(m,'aux|prior').to_string())
print('\n== Prior high-resolution model of the same V region, by era')
print(D.pivot_table(index='era',columns='prior_hires',values='flag',aggfunc=lambda x: f'{100*x.mean():.1f}% (n={len(x)})').to_string())
print('\n== Journal tier (new era)')
print(D[D.era.str.startswith('new')].groupby('tier').flag.agg(lambda x: f'{100*x.mean():.1f}% (n={len(x)})').to_string())
# ---------- where: IMGT region of each disagreement run
def seg_imgt(res):
    out=[]; cur=[]
    v=[x for x in res if x[4] is not None]
    for x in v+[None]:
        dev=x is not None and (x[4],x[5] or '')!=(x[6],x[7] or '')
        if dev: cur.append(x)
        else:
            if len(cur)>=3: out.append(cur)
            cur=[]
    return out
recs={(r['pdb'],r['chain']):r for r in old+new if r.get('status')=='ok'}
segrows=[]
for row in CH[CH.flag==1].itertuples():
    for s in seg_imgt(recs[(row.pdb,row.chain)]['residues']):
        a0=s[0][4]; a1=s[-1][4]; d=[x[6]-x[4] for x in s if region(x[4])=='FR' and not (x[5] or x[7])]
        segrows.append(dict(pdb=row.pdb,chain=row.chain,era=row.era,chain_type=row.chain_type,imgt_start=a0,imgt_end=a1,start_region=region(a0),
            loc=('FR1' if a0<=26 else 'CDR1' if a0<=38 else 'FR2' if a0<=55 else 'CDR2' if a0<=65 else 'FR3' if a0<=104 else 'CDR3' if a0<=117 else 'FR4'),
            delta=int(np.median(d)) if d else None,length=len(s),seqkey=row.seq[:105],last_author=row.last_author,paper=row.cite_key,deposit=row.deposit,year=row.year,res=row.res,method=row.method))
SG=pd.DataFrame(segrows); SG.to_pickle(f'{O}/segments.pkl')
print('\n== Where the disagreement runs start (IMGT region), by era and chain type')
print(pd.crosstab([SG.era,SG.chain_type.map({'H':'heavy','K':'light','L':'light'})],SG['loc'],normalize='index').round(2).to_string())
print('shift size |Δ|:'); print(pd.crosstab(SG.era,SG.delta.abs(),normalize='index').round(2).to_string())
# ---------- recurrence of identical errors (same V sequence, same IMGT start ±2, same Δ)
SG=SG[SG.seqkey.str.len()>=50].copy(); SG['k']=SG.seqkey.astype(object)+'|'+SG.imgt_start.astype(int).astype(str).astype(object)+'|'+SG.delta.fillna(-999).astype(int).astype(str).astype(object)
fam=SG.groupby('k').agg(entries=('pdb','nunique'),labs=('last_author','nunique'),papers=('paper','nunique'),first=('deposit','min'),last=('deposit','max'),
     eras=('era',lambda x:'+'.join(sorted(set(x)))),loc=('loc','first'),delta=('delta','first'),example=('pdb','first')).reset_index()
print('\n== Identical errors (same V sequence, same position, same shift)')
print(' error families:',len(fam),'| in ≥2 entries:',(fam.entries>=2).sum(),'| ≥2 papers:',(fam.papers>=2).sum(),'| ≥2 labs:',(fam.labs>=2).sum())
print(' families seen in both snapshot and new entries:',fam.eras.str.contains('\\+').sum())
nw=SG[SG.era.str.startswith('new')]; nk=nw.drop_duplicates(['pdb','chain','k'])
earlier=SG.groupby('k').apply(lambda g: g[['paper','deposit']].values.tolist()).to_dict()
def inherited(r):
    return any(p!=r.paper and d<r.deposit for p,d in earlier[r.k])
nk['inherited']=nk.apply(inherited,axis=1)
print(' new-era disagreement runs: %d; same error already deposited earlier by a different paper: %d (%.0f%%)'%(len(nk),nk.inherited.sum(),100*nk.inherited.mean()))
top=fam.sort_values(['labs','entries'],ascending=False).head(12)
names={}
with gzip.open('/tmp/supp/pdbwide/pdb_seqres.txt.gz','rt') as fh:
    for line in fh:
        if line.startswith('>'):
            h=line[1:].split(None,3); names[h[0].split('_')[0]+'|'+h[0].split('_')[1]]=h[3].strip() if len(h)>3 else ''
ex=SG.drop_duplicates('k').set_index('k')
top['name']=[names.get(ex.loc[k].pdb+'|'+ex.loc[k].chain,'') for k in top.k]
print(top[['name','example','entries','papers','labs','first','last','eras','loc','delta']].to_string(index=False))
fam.to_csv(f'{O}/identical_error_families.csv',index=False)
