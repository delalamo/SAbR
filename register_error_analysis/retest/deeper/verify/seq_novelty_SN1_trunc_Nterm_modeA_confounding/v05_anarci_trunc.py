"""Sequence-arm check. (1) Does ANARCI keep the correct IMGT numbering of the remaining residues when the first k V
residues are removed from a full SEQRES (control chains)?  (2) For the 19 truncated mode-A error chains: if the
missing germline N-terminal residues are added back, does ANARCI's numbering of the deposited residues change?"""
import json, pandas as pd, numpy as np
from anarci import run_anarci
from anarci.germlines import all_germlines as G
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
OUT='/home/user/SAbR/register_error_analysis/retest/deeper/verify/seq_novelty_SN1_trunc_Nterm_modeA_confounding'
D=pd.read_csv(f'{OUT}/units_modeA_analysis.csv.gz',low_memory=False)
ch=pd.read_csv(f'{SP}/deeper/chains.csv.gz',low_memory=False,usecols=['pdb','chain','seq','vseq','v_gene','species'])
def number(seq,ct):
    _,num,det,_=run_anarci([('q',seq)],scheme='imgt',allow={'H','K','L'})
    if not num[0]: return None
    for (nb,st,en),d in zip(num[0],det[0]):
        if (d['chain_type'] in 'KL')==(ct in 'KL'):
            out=[];i=st
            for (p,ins),a in nb:
                if a=='-': continue
                out.append((i,p,ins.strip())); i+=1
            return out
    return None
# (1) control chains
c=D[(D.y==0)&(D.x_model_first_anarci==1)].sample(150,random_state=3)
c=c.merge(ch,left_on=['r0_pdb','r0_chain'],right_on=['pdb','chain'],suffixes=('','_c'))
rows=[]
for r in c.itertuples():
    full=number(r.seq,r.ctype)
    if not full: continue
    off=full[0][0]
    fmap={i:(p,ins) for i,p,ins in full}
    for k in (1,2,3):
        tseq=r.seq[:off]+r.seq[off+k:]
        t=number(tseq,r.ctype)
        if not t: rows.append(dict(pdb=r.pdb,chain=r.chain,ctype=r.ctype,k=k,ok=False)); continue
        # map truncated index back to original index
        back=lambda j: j if j<off else j+k
        diff=sum(1 for j,p,ins in t if fmap.get(back(j))!=(p,ins))
        first=t[0]
        rows.append(dict(pdb=r.pdb,chain=r.chain,ctype=r.ctype,k=k,ok=True,n_changed=diff,first_pos=first[1],expected_first=fmap[off+k][0]))
T=pd.DataFrame(rows); T.to_csv(f'{OUT}/v05_anarci_trunc_controls.csv',index=False)
T['any_change']=T.n_changed>0; T['first_ok']=T.first_pos==T.expected_first
print('(1) ANARCI on in-silico truncated control SEQRES:')
print(T.groupby(['k','ctype']).agg(n=('ok','size'),any_change=('any_change','mean'),changed3=('n_changed',lambda x:(x>=3).mean()),first_ok=('first_ok','mean')))
# (2) truncated error chains: add back germline residues
case=D[(D.y==1)&(D.seq_trunc==1)].merge(ch,left_on=['rA_pdb','rA_chain'],right_on=['pdb','chain'],suffixes=('','_c'))
rows2=[]
for r in case.itertuples():
    full=number(r.seq,r.ctype); off=full[0][0]; first=full[0][1]
    ctk={'H':'H','K':'K','L':'L'}[r.ctype]
    sp='mouse' if r.species=='mouse' else 'human'
    gene=str(r.v_gene)
    gl=G['V'][ctk].get(sp,{}).get(gene)
    if gl is None:
        # fall back: any species
        for s2 in G['V'][ctk]:
            if gene in G['V'][ctk][s2]: gl=G['V'][ctk][s2][gene]; break
    add=''.join(a for a in gl[:first-1] if a!='-') if gl else ''
    t=number(r.seq[:off]+add+r.seq[off:],r.ctype)
    tmap={j-len(add) if j>=off+len(add) else None:(p,ins) for j,p,ins in t}
    fmap={i:(p,ins) for i,p,ins in full}
    changed=sum(1 for i in fmap if i in tmap and tmap[i]!=fmap[i])
    rows2.append(dict(pdb=r.pdb,chain=r.chain,ctype=r.ctype,v_gene=gene,seq_start=r.seq[off:off+12],anarci_first=first,
                      germline_prefix_added=add,n_residues_renumbered_after_adding=changed,
                      germline_head=''.join(a for a in gl[:12] if a!='-') if gl else None))
T2=pd.DataFrame(rows2); T2.to_csv(f'{OUT}/v05_case_addback.csv',index=False)
print('\n(2) truncated mode-A error chains, germline N-terminus added back:'); print(T2.to_string())
