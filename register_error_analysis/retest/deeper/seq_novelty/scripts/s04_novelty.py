"""Per V chain in chains.csv: nearest prior PDB antibody (by region), germline novelty, SHM per region,
rare residues, nanobody hallmarks, humanisation signature. Output: chain_novelty.csv.gz"""
import json, pickle, collections, re, numpy as np, pandas as pd
from anarci.germlines import all_germlines as G
from anarci import run_anarci
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
C=f'{SP}/deeper/cache/seq_novelty'
OUT='/home/user/SAbR/register_error_analysis/retest/deeper/seq_novelty/tables'
import os; os.makedirs(OUT,exist_ok=True)
db=pickle.load(open(f'{C}/abdb.pkl','rb')); D,M,keys,code,S=db['D'],db['M'],db['keys'],db['code'],db['S']
num=pickle.load(open(f'{C}/anarci_num.pkl','rb'))
k2i=json.load(open(f'{C}/kabat2imgt.json'))
kidx={k:i for i,k in enumerate(keys)}
pos=np.array([k[0] for k in keys]); isins=np.array([k[1]!=' ' for k in keys])
REG={'NT':(1,10),'FR1':(1,26),'CDR1':(27,38),'FR2':(39,55),'CDR2':(56,65),'FR3':(66,104),'CDR3':(105,117),'FR4':(118,128)}
RN=list(REG)+['FW','V']
R=np.zeros((len(keys),len(RN)),dtype=np.float32)
for j,r in enumerate(RN):
    if r=='FW': sel=np.isin(np.arange(len(keys)),np.where(((pos<=26))|((pos>=39)&(pos<=55))|((pos>=66)&(pos<=104))|(pos>=118))[0])
    elif r=='V': sel=np.ones(len(keys),bool)
    else: sel=(pos>=REG[r][0])&(pos<=REG[r][1])
    R[sel,j]=1
ri={r:j for j,r in enumerate(RN)}
ch=pd.read_csv(f'{SP}/deeper/chains.csv.gz')
# --- map each chain to its V domain in the DB
dbkey={(c,v):i for i,(c,v) in enumerate(zip(D.ctype,D.vdom))}
missing=[s for s in ch.seq.unique() if s not in num]
if missing:
    _,nn,aa,_=run_anarci([(str(i),s) for i,s in enumerate(missing)],scheme='imgt',ncpu=2)
    for s,n,a in zip(missing,nn,aa):
        num[s]=[(d['chain_type'],st,en,d.get('evalue'),[(p,ins,x) for (p,ins),x in nb if x!='-']) for (nb,st,en),d in zip(n,a)] if n else []
def vdom_of(seq,ctype,vseq):
    best=None
    for (ct,st,en,ev,nb) in num.get(seq,[]):
        if ct not in 'HKL': continue
        vs=''.join(a for _,_,a in nb)
        sc=(ct==ctype)*1000+len(os.path.commonprefix([vs,vseq]))
        if best is None or sc>best[0]: best=(sc,ct,vs,nb)
    return best
extra_rows=[]; Mx=[]
qrow=[]
for s,ct,v in zip(ch.seq,ch.ctype,ch.vseq):
    b=vdom_of(s,ct,v)
    if b is None: qrow.append(-1); continue
    key=(b[1],b[2])
    if key in dbkey: qrow.append(dbkey[key]); continue
    # not in DB (should be rare): append a row with no entries
    vec=np.zeros(len(keys),dtype=np.uint8)
    for p,i,a in b[3]:
        if (p,i) in kidx: vec[kidx[(p,i)]]=code.get(a,21)
    dbkey[key]=len(D)+len(extra_rows); extra_rows.append({'ctype':b[1],'vdom':b[2],'n_entries':0,'first_release':'2999-01-01'}); Mx.append(vec)
    qrow.append(dbkey[key])
if extra_rows:
    print('chains whose V domain not in DB:',len(extra_rows))
    D=pd.concat([D,pd.DataFrame(extra_rows)],ignore_index=True); M=np.vstack([M,np.array(Mx)])
    mc=np.array([kidx.get((p,' '),-1) for p in range(1,129)]); S2=np.zeros((len(Mx),128),np.uint8); S2[:,mc>=0]=np.array(Mx)[:,mc[mc>=0]]; S=np.vstack([S,S2])
ch['dbrow']=qrow
print('unmapped chains', (ch.dbrow<0).sum(), 'ctype mismatch', (D.ctype.reindex(ch.dbrow.clip(0)).values!=ch.ctype.values).sum())
D['first_release']=D.first_release.fillna('2999-01-01')
# --- position frequencies (unique V domains per chain type)
freq={}
for ct in 'HKL':
    X=M[(D.ctype==ct).values]
    F=np.zeros((22,len(keys)))
    for a in range(22): F[a]=(X==a).sum(0)
    occ=F[1:].sum(0); freq[ct]=(F,occ)
# kabat sets -> imgt keys
KAB={'vernier':{'H':[2,27,28,29,30,47,48,49,67,69,71,73,78,93,94,103],'K':[2,4,35,36,46,47,48,49,64,66,68,69,71,98]},
     'canonical':{'H':[24,26,27,29,34,54,55,71,94],'K':[2,25,29,30,33,48,51,64,71,90,95,97]},
     'interface':{'H':[35,37,39,44,45,47,91,93,103],'K':[34,36,38,44,46,87,89,98]}}
SETS={}
for nm,d in KAB.items():
    for ct in 'HKL':
        kl=d['H' if ct=='H' else 'K']
        SETS[(nm,ct)]=[kidx[tuple([k2i[ct][str(k)][0][0], k2i[ct][str(k)][0][1] or ' '])] for k in kl if str(k) in k2i[ct] and tuple([k2i[ct][str(k)][0][0], k2i[ct][str(k)][0][1] or ' ']) in kidx]
print({k:len(v) for k,v in SETS.items()})
# --- germline tables
def gstr(seg,ct,sp,g):
    try: return G[seg][ct][sp][g]
    except KeyError: return None
def gene(g): return None if not isinstance(g,str) else g.split('*')[0]
D['vg']=D.v_gene_hm.map(gene); D['jg']=D.j_gene_hm.map(gene)
# entry-level germline table for "first time in PDB" counts
E=[]
for i,r in D.iterrows():
    if r.n_entries==0: continue
    for e in r.entries.split():
        E.append((e,r.ctype,r.vg,r.jg,r.len_CDR1,r.len_CDR2,r.len_CDR3))
E=pd.DataFrame(E,columns=['e','ctype','vg','jg','L1','L2','L3']).drop_duplicates()
rel={}
for l in open('/tmp/supp/pdbwide/cite_all.jsonl'):
    d=json.loads(l); a=d.get('rcsb_accession_info') or {}
    if a.get('initial_release_date'): rel[d['rcsb_id'].lower()]=a['initial_release_date'][:10]
E['rel']=E.e.map(rel).fillna('2999-01-01')
def first_date_index(cols):
    g=E.groupby(cols).rel.apply(lambda x: np.sort(x.drop_duplicates().values) if False else np.sort(x.values))
    return g
# count distinct entries with release < date sharing the key
def prior_counter(cols):
    sub=E[cols+['e','rel']].drop_duplicates(cols+['e'])
    d={k:np.sort(g.rel.values) for k,g in sub.groupby(cols)}
    return d
CNT={nm:prior_counter(cols) for nm,cols in {'V':['ctype','vg'],'J':['ctype','jg'],'VJ':['ctype','vg','jg'],
      'L123':['ctype','L1','L2','L3'],'VL12':['ctype','vg','L1','L2'],'VJL3':['ctype','vg','jg','L3'],'JL3':['ctype','jg','L3'],'L3':['ctype','L3']}.items()}
COLS={'V':['ctype','vg'],'J':['ctype','jg'],'VJ':['ctype','vg','jg'],'L123':['ctype','len_CDR1','len_CDR2','len_CDR3'],'VL12':['ctype','vg','len_CDR1','len_CDR2'],
      'VJL3':['ctype','vg','jg','len_CDR3'],'JL3':['ctype','jg','len_CDR3'],'L3':['ctype','len_CDR3']}
# --- nearest prior per unique (dbrow, deposit)
Nt={}; qs=ch[ch.dbrow>=0][['dbrow','deposit']].drop_duplicates()
res={}
for ct in 'HKL':
    rows=np.where((D.ctype==ct).values)[0]
    Mc=M[rows]; Ntc=(Mc>0).astype(np.float32)@R
    relc=D.first_release.values[rows]
    l3c=Mc[:, (pos>=105)&(pos<=117)]
    sub=qs[D.ctype.values[qs.dbrow.values]==ct]
    for qi,g in sub.groupby('dbrow'):
        q=M[qi]; nq=(q>0).astype(np.float32)@R
        eq=((Mc==q[None])&(q[None]>0)).astype(np.float32)@R
        den=np.maximum(Ntc,nq[None]); idn=np.where(den>0,eq/np.where(den>0,den,1),np.nan)
        qcdr3=q[(pos>=105)&(pos<=117)]
        same_cdr3=(l3c==qcdr3[None]).all(1)
        same_ab=same_cdr3&(idn[:,ri['V']]>=0.95)
        same_l3=Ntc[:,ri['CDR3']]==nq[ri['CDR3']]
        for dep in g.deposit:
            prior=relc<dep
            out={}
            for nm,msk in (('any',prior),('diff',prior&~same_ab)):
                if msk.sum()==0:
                    for r in RN: out[f'nn_{nm}_{r}']=np.nan
                    out[f'nn_{nm}_hit']=None; continue
                I=idn[msk]
                for r in RN: out[f'nn_{nm}_{r}']=np.nanmax(I[:,ri[r]]) if np.isfinite(I[:,ri[r]]).any() else np.nan
                b=np.nanargmax(I[:,ri['V']]); bi=rows[np.where(msk)[0][b]]
                out[f'nn_{nm}_hit']=D.entries.values[bi].split()[0]; out[f'nn_{nm}_hit_rel']=D.first_release.values[bi]
                # framework of the best whole-V hit, and region identities of that same hit
                for r in RN: out[f'nnhit_{nm}_{r}']=I[b,ri[r]]
            m3=prior&~same_ab&same_l3
            out['nn_diff_CDR3_samelen']=np.nanmax(idn[m3,ri['CDR3']]) if m3.any() else 0.0
            out['n_prior_same_ab']=int((prior&same_ab).sum())
            out['n_prior_samelen_cdr3']=int((prior&same_l3).sum())
            out['exact_prior']=bool((prior&(idn[:,ri['V']]==1)&same_cdr3).any())
            out['n_prior_V_rows']=int(prior.sum())
            out['n_prior_ge90']=int((prior&~same_ab&(idn[:,ri['V']]>=0.9)).sum())
            out['n_prior_ge80']=int((prior&~same_ab&(idn[:,ri['V']]>=0.8)).sum())
            res[(qi,dep)]=out
NV=pd.DataFrame([dict(dbrow=k[0],deposit=k[1],**v) for k,v in res.items()])
ch=ch.merge(NV,on=['dbrow','deposit'],how='left')
# --- germline features per chain
qD=D.reindex(ch.dbrow.clip(0)).reset_index(drop=True)
for c in ['v_gene_hm','j_gene_hm','v_id_hm','j_id_hm','species_hm','v_gene_all','species_all','v_id_all','vg','jg','len_CDR1','len_CDR2','len_CDR3','len_FR1','len_FR4']:
    ch['q_'+c]=qD[c].values
for nm,cols in COLS.items():
    d=CNT[nm]
    vals=[]
    for i,r in qD.iterrows():
        k=tuple(r[c] for c in cols)
        arr=d.get(k if len(k)>1 else k[0])
        vals.append(0 if arr is None else int(np.searchsorted(arr,ch.deposit.iat[i])))
    ch['gprior_'+nm]=vals
# SHM per region vs assigned (human/mouse) germline; CDR1/2 indels vs germline
regs128={'FR1':(1,26),'CDR1':(27,38),'FR2':(39,55),'CDR2':(56,65),'FR3':(66,104),'FR4':(118,128)}
AAinv={v:k for k,v in code.items()}
for r in regs128: ch['mut_'+r]=np.nan
ch['cdr1_len_vs_gl']=np.nan; ch['cdr2_len_vs_gl']=np.nan
for i,r in qD.iterrows():
    if ch.dbrow.iat[i]<0: continue
    s=S[ch.dbrow.iat[i]]
    gv=gstr('V',r.ctype,r.species_hm,r.v_gene_hm); gj=gstr('J',r.ctype,r.species_hm,r.j_gene_hm)
    for reg,(a,b) in regs128.items():
        gl=gj if reg=='FR4' else gv
        if gl is None: continue
        m=0
        for p in range(a,b+1):
            x=s[p-1]; y=gl[p-1]
            if x and y!='-' and AAinv[x]!=y: m+=1
        ch.at[i,'mut_'+reg]=m
    if gv is not None:
        ch.at[i,'cdr1_len_vs_gl']=r.len_CDR1-sum(1 for p in range(27,39) if gv[p-1]!='-')
        ch.at[i,'cdr2_len_vs_gl']=r.len_CDR2-sum(1 for p in range(56,66) if gv[p-1]!='-')
# rare residues per position set and per IMGT window
def rare_flags(row,ct):
    q=M[row]; F,occ=freq[ct]
    f=np.where(occ>0,F[q,np.arange(len(q))]/np.maximum(occ,1),1.0)
    rare=(q>0)&(f<0.01)
    rare_ins=(q>0)&(occ/ (D.ctype==ct).sum()<0.01)   # residue at a column that <1% of V domains occupy
    return rare,rare_ins
WIN={'A':(1,13),'F':(24,41),'B':(55,72),'C':(74,92),'D':(105,128),'E':(116,128),'FR4':(118,128),'CDR3':(105,117)}
RARE_STORE={}
for i in range(len(ch)):
    row=ch.dbrow.iat[i]
    if row<0: continue
    ct=ch.ctype.iat[i]
    if row in RARE_STORE: continue
    rare,ri_=rare_flags(row,ct)
    d={}
    for nm in ('vernier','canonical','interface'):
        idx=SETS[(nm,ct)]; d['rare_'+nm]=int(rare[idx].sum())
    for w,(a,b) in WIN.items():
        sel=(pos>=a)&(pos<=b); d['rare_win_'+w]=int(rare[sel].sum()); d['rareins_win_'+w]=int(ri_[sel].sum())
    d['rare_FW']=int(rare[R[:,ri['FW']]>0].sum()); d['rare_V']=int(rare.sum())
    # list of rare positions for run-level lookups
    d['rare_pos']=' '.join(f'{keys[j][0]}{keys[j][1].strip()}' for j in np.where(rare)[0])
    # nanobody hallmarks (IMGT 42,49,50,52)
    if ct=='H':
        aa={p:AAinv.get(M[row][kidx[(p,' ')]],'-') for p in (42,49,50,52)}
        d['hallmark']=''.join(aa[p] for p in (42,49,50,52))
        d['vhh_hallmarks']=int(aa[42] in 'FY')+int(aa[49] in 'EQ')+int(aa[50] in 'RC')+int(aa[52] in 'FGLY')
    RARE_STORE[row]=d
RS=pd.DataFrame.from_dict(RARE_STORE,orient='index'); RS.index.name='dbrow'
ch=ch.merge(RS,left_on='dbrow',right_index=True,how='left')
# humanisation signature: framework near human germline, CDR1+CDR2 nearer mouse germline
def best_region_id(row,ct,species,lo_hi_list):
    s=S[row]; best=0
    for sp in species:
        for g,sq in G['V'].get(ct,{}).get(sp,{}).items():
            m=n=0
            for a,b in lo_hi_list:
                for p in range(a,b+1):
                    if sq[p-1]=='-' and not s[p-1]: continue
                    n+=1; m+= (s[p-1]>0 and AAinv[s[p-1]]==sq[p-1])
            best=max(best,m/max(n,1))
    return best
HU={}
for row,ct in ch[ch.dbrow>=0][['dbrow','ctype']].drop_duplicates().values:
    cdr=[(27,38),(56,65)]; fw=[(1,26),(39,55),(66,104)]
    HU[row]=dict(hu_fw=best_region_id(row,ct,['human'],fw),hu_cdr=best_region_id(row,ct,['human'],cdr),mo_cdr=best_region_id(row,ct,['mouse'],cdr),mo_fw=best_region_id(row,ct,['mouse'],fw))
HU=pd.DataFrame.from_dict(HU,orient='index'); HU.index.name='dbrow'
ch=ch.merge(HU,left_on='dbrow',right_index=True,how='left')
ch['humanised_sig']=(ch.hu_fw>=0.85)&(ch.mo_cdr>=ch.hu_cdr+0.15)&(ch.hu_fw>ch.mo_fw)
ch['humanised_kw']=(ch.chain_desc.fillna('')+' '+ch.title.fillna('')).str.contains('humani[sz]',case=False)
ch['scfv_linker']=ch.seq.str.contains('GGGGSGGGGS|GGGSGGGS|GSTSGSGKPGSGEGS|GGSSRSS')
ch['engineered_multidomain']=D.multi_domain_chain.reindex(ch.dbrow.clip(0)).values
keep=[c for c in ch.columns if c not in ('seq',)]
ch[keep].to_csv(f'{OUT}/chain_novelty.csv.gz',index=False)
print(ch[['nn_any_V','nn_diff_V','nn_diff_FW','nn_diff_CDR3','exact_prior','gprior_V','gprior_VJ','rare_FW']].describe().T)
